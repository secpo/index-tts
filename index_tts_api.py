import os
import uuid
import wave
import urllib.request
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from indextts.infer_v2 import IndexTTS2


def chunk_text(text: str, max_len: int) -> List[str]:
    seps = {".", "!", "?", "。", "！", "？", "；", ";", "，", ",", "\n", " "}
    res: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_len, n)
        k = j
        while k > i and text[k - 1] not in seps:
            k -= 1
        if k == i:
            k = j
        res.append(text[i:k].strip())
        i = k
    return [s for s in res if s]


def merge_wavs(paths: List[str], out_path: str) -> None:
    if not paths:
        raise ValueError("no input wavs")
    with wave.open(paths[0], "rb") as w0:
        nch = w0.getnchannels()
        sampwidth = w0.getsampwidth()
        fr = w0.getframerate()
    with wave.open(out_path, "wb") as out:
        out.setnchannels(nch)
        out.setsampwidth(sampwidth)
        out.setframerate(fr)
        for p in paths:
            with wave.open(p, "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))


class TTSService:
    def __init__(
        self,
        model_dir: str,
        cfg_path: str,
        use_fp16: bool = True,
        use_cuda_kernel: bool = False,
    ) -> None:
        self.tts = IndexTTS2(
            model_dir=model_dir,
            cfg_path=cfg_path,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
        )
        self._sr = int(self.tts.cfg.s2mel['preprocess_params']['sr'])
        self._hop = int(self.tts.cfg.s2mel['preprocess_params']['spect_params']['hop_length'])
        self._code_to_frame = 1.72

    def infer_chunk(
        self,
        text: str,
        voice_path: str,
        out_path: str,
        verbose: bool = False,
        max_text_tokens_per_segment: Optional[int] = None,
        duration_sec: Optional[float] = None,
    ) -> str:
        kwargs = {}
        if max_text_tokens_per_segment is not None:
            kwargs["max_text_tokens_per_segment"] = max_text_tokens_per_segment
        if duration_sec is not None and duration_sec > 0:
            frames = duration_sec * (self._sr / self._hop)
            code_len = int(max(32, round(frames / self._code_to_frame)))
            kwargs["max_mel_tokens"] = code_len
        return self.tts.infer(
            spk_audio_prompt=voice_path,
            text=text,
            output_path=out_path,
            verbose=verbose,
            **kwargs,
        )


app = FastAPI()

MODEL_DIR = os.getenv("INDEXTTS_MODEL_DIR", "checkpoints")
CFG_PATH = os.getenv("INDEXTTS_CFG_PATH", os.path.join(MODEL_DIR, "config.yaml"))
CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", "300"))
service = TTSService(MODEL_DIR, CFG_PATH)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/synthesize")
async def synthesize(
    file: UploadFile | None = File(None),
    text: Optional[str] = Form(None),
    text_url: Optional[str] = Form(None),
    voice_path: str = Form(...),
    drive_dir: Optional[str] = Form(None),
    chunk_length: int = Form(CHUNK_LENGTH),
    session_id: Optional[str] = Form(None),
    duration_sec: Optional[float] = Form(None),
) -> JSONResponse:
    if file is not None:
        content = await file.read()
        text = content.decode("utf-8")
    elif text is not None:
        pass
    elif text_url is not None:
        try:
            with urllib.request.urlopen(text_url) as resp:
                text = resp.read().decode("utf-8")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to fetch text_url: {e}")
    else:
        raise HTTPException(status_code=400, detail="one of file | text | text_url is required")
    cps_env = os.getenv("CN_CHAR_PER_SEC", "5.0")
    try:
        cps = float(cps_env)
    except Exception:
        cps = 5.0
    text_len = len("".join(text.split()))
    if text_len == 0:
        raise HTTPException(status_code=400, detail="empty text")
    est_sec = max(0.2, text_len / cps)
    if duration_sec is not None:
        if duration_sec <= 0:
            raise HTTPException(status_code=400, detail="duration_sec must be > 0")
        low = 0.5 * est_sec
        high = 1.5 * est_sec
        if not (low <= duration_sec <= high):
            raise HTTPException(status_code=400, detail=f"duration_sec out of allowed range [{low:.2f}, {high:.2f}] for estimated {est_sec:.2f}s")
    sid = session_id or str(uuid.uuid4())
    base_dir = drive_dir or os.getenv("GOOGLE_DRIVE_DIR", "outputs")
    out_dir = os.path.join(base_dir, sid)
    os.makedirs(out_dir, exist_ok=True)
    chunks = chunk_text(text, chunk_length)
    chunk_files: List[str] = []
    total_chars = sum(len(c) for c in chunks) or 1
    for idx, chunk in enumerate(chunks, start=1):
        name = f"chunk_{idx:04d}.wav"
        path = os.path.join(out_dir, name)
        dsec = None
        if duration_sec is not None and duration_sec > 0:
            ratio = len(chunk) / total_chars
            dsec = max(0.2, duration_sec * ratio)
        out = service.infer_chunk(
            text=chunk,
            voice_path=voice_path,
            out_path=path,
            verbose=False,
            duration_sec=dsec,
        )
        chunk_files.append(out or path)
    final_path = os.path.join(out_dir, "final.wav")
    merge_wavs(chunk_files, final_path)
    return JSONResponse(
        {
            "session_id": sid,
            "chunk_count": len(chunk_files),
            "chunk_files": chunk_files,
            "final_file": final_path,
        }
    )


def main() -> None:
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() in ("1", "true", "yes")
    if share:
        import gradio as gr
        print(f"IndexTTS API running: http://{host}:{port}/api")
        print(f"Docs: http://{host}:{port}/api/docs")
        print(
            f"curl -F \"file=@/path/text.txt\" -F \"voice_path=/path/voice.wav\" -F \"drive_dir=/content/drive/MyDrive/IndexTTS/outputs\" http://{host}:{port}/api/synthesize"
        )
        print(
            f"curl -F \"text=你好，世界\" -F \"voice_path=/path/voice.wav\" http://{host}:{port}/api/synthesize"
        )
        print(
            f"curl -F \"text_url=https://example.com/input.txt\" -F \"voice_path=/path/voice.wav\" http://{host}:{port}/api/synthesize"
        )
        print(
            f"curl -F \"file=@C:\\tts.txt\" -F \"voice_path=C:\\sample.wav\" http://{host}:{port}/api/synthesize"
        )
        demo = gr.Blocks()
        demo.app.mount("/api", app)
        demo.queue(16)
        demo.launch(server_name=host, server_port=port, share=True)
    else:
        print(f"IndexTTS API running: http://{host}:{port}")
        print(f"Docs: http://127.0.0.1:{port}/docs")
        print(
            f"curl -F \"file=@/path/text.txt\" -F \"voice_path=/path/voice.wav\" -F \"drive_dir=/content/drive/MyDrive/IndexTTS/outputs\" http://127.0.0.1:{port}/synthesize"
        )
        print(
            f"curl -F \"text=你好，世界\" -F \"voice_path=/path/voice.wav\" http://127.0.0.1:{port}/synthesize"
        )
        print(
            f"curl -F \"text_url=https://example.com/input.txt\" -F \"voice_path=/path/voice.wav\" http://127.0.0.1:{port}/synthesize"
        )
        print(
            f"curl -F \"file=@C:\\tts.txt\" -F \"voice_path=C:\\sample.wav\" http://127.0.0.1:{port}/synthesize"
        )
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
