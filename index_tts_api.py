import os
import uuid
import wave
import shutil
import uvicorn
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pyngrok import ngrok

from indextts.infer_v2 import IndexTTS2

# --- Helper Functions ---

def chunk_text(text: str, max_len: int) -> List[str]:
    """Splits text into chunks of a maximum length."""
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
    """Merges multiple WAV files into a single file."""
    if not paths:
        raise ValueError("No input WAV files to merge.")
    with wave.open(paths[0], "rb") as w0:
        params = w0.getparams()
    with wave.open(out_path, "wb") as out:
        out.setparams(params)
        for p in paths:
            with wave.open(p, "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))

# --- TTS Service Class ---

class TTSService:
    """A service class to handle Text-to-Speech inference."""
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
        duration_sec: Optional[float] = None,
    ) -> str:
        kwargs = {}
        if duration_sec is not None and duration_sec > 0:
            frames = duration_sec * (self._sr / self._hop)
            code_len = int(max(32, round(frames / self._code_to_frame)))
            kwargs["max_mel_tokens"] = code_len
        return self.tts.infer(
            spk_audio_prompt=voice_path,
            text=text,
            output_path=out_path,
            **kwargs,
        )

# --- FastAPI Application ---

app = FastAPI(title="IndexTTS API", description="A pure API for IndexTTS2 using FastAPI and ngrok.", lifespan=lifespan)

tts_service: Optional[TTSService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the TTS model on application startup and handle shutdown.
    """
    global tts_service
    print("--- Lifespan startup ---")
    model_dir = os.environ.get("MODEL_DIR", "/content/index-tts/checkpoints/IndexTTS-2")
    config_path = os.path.join(model_dir, "config.json")
    
    if not os.path.exists(model_dir) or not os.path.exists(config_path):
        print(f"\033[91mWarning: Model directory or config not found at {model_dir}. The API will not work.\033[0m")
    else:
        try:
            print("Loading TTS model...")
            tts_service = TTSService(model_dir=model_dir, cfg_path=config_path, use_fp16=True)
            print("\033[92mTTS Service loaded successfully.\033[0m")
        except Exception as e:
            print(f"\033[91mError loading TTS Service: {e}\033[0m")
    
    yield
    
    print("--- Lifespan shutdown ---")
    tts_service = None

tts_service: Optional[TTSService] = None

@app.post("/api/synthesize")
async def synthesize(
    voice_path: str = Form(...),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    duration_sec: Optional[float] = Form(None),
):
    """Synthesize speech from text or a text file with chunking."""
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service is not available. Check model path.")

    if not text and not file:
        raise HTTPException(status_code=400, detail="Either 'text' (form field) or 'file' (upload) must be provided.")

    input_text = ""
    if text:
        input_text = text
    elif file:
        contents = await file.read()
        input_text = contents.decode("utf-8")

    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    temp_dir = f"/tmp/{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        chunks = chunk_text(input_text, max_len=200)
        wav_paths = []

        print(f"Starting synthesis for {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            out_path = os.path.join(temp_dir, f"{i}.wav")
            try:
                tts_service.infer_chunk(
                    text=chunk,
                    voice_path=voice_path,
                    out_path=out_path,
                    duration_sec=duration_sec if i == 0 else None
                )
                wav_paths.append(out_path)
                print(f"  - Chunk {i+1}/{len(chunks)} synthesized.")
            except Exception as e:
                print(f"\033[91mError processing chunk {i+1}: {e}\033[0m")
                raise HTTPException(status_code=500, detail=f"Error processing chunk {i+1}: {str(e)}")

        if not wav_paths:
            raise HTTPException(status_code=500, detail="Synthesis failed, no audio chunks were generated.")

        output_filename = f"{uuid.uuid4()}_final.wav"
        # Hardcode the output directory to a specific Google Drive path
        output_dir = "/content/drive/MyDrive/IndexTTS/outputs"
        os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists
        final_out_path = os.path.join(output_dir, output_filename)
        
        merge_wavs(wav_paths, final_out_path)
        print(f"\033[92mSynthesis complete. Final audio at: {final_out_path}\033[0m")

        return {"output_path": final_out_path, "message": "Synthesis successful."}

    finally:
        shutil.rmtree(temp_dir)

def main():
    """Sets up ngrok tunnel and starts the FastAPI server."""
    # Get config from environment variables
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", 7890))
    ngrok_authtoken = os.environ.get("NGROK_AUTHTOKEN")

    if ngrok_authtoken:
        ngrok.set_auth_token(ngrok_authtoken)
    
    # Set up ngrok tunnel
    public_url = ngrok.connect(port, "http")
    
    print("\n" + "="*60)
    print("🚀 IndexTTS API is running!")
    print(f"✅ Public Docs URL (ngrok): {public_url}/docs")
    print(f"✅ Local Docs URL: http://{host}:{port}/docs")
    print("="*60 + "\n")

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
