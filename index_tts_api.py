import os
import uuid
import wave
import argparse
import re
import shutil
import sys
import uvicorn
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pyngrok import ngrok

# --- Command Line Arguments ---
parser = argparse.ArgumentParser(
    description="IndexTTS API Server",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 for inference if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
cmd_args = parser.parse_args()

from indextts.infer_v2 import IndexTTS2

# --- Helper Functions ---

def move_punctuation_for_english_quotes(text: str) -> str:
    """
    专门处理英文引号(" " 和 ' ')，不使用正则表达式，将引号内末尾的标点移动到引号外
    """
    punctuations = {'.', '。', '!', '！', '?', '？'}
    quote_chars = {'"', "'"}
    
    char_list = list(text)
    n = len(char_list)
    
    i = 0
    while i < n:
        current_char = char_list[i]
        if current_char in quote_chars:
            left_quote_index = i
            right_quote_index = -1
            for j in range(left_quote_index + 1, n):
                if char_list[j] == current_char:
                    right_quote_index = j
                    break # 找到后立即停止
            if right_quote_index != -1:
                char_before_right_quote_index = right_quote_index - 1
                if char_before_right_quote_index > left_quote_index and char_list[char_before_right_quote_index] in punctuations:
                    punc = char_list[char_before_right_quote_index]
                    quote = char_list[right_quote_index]
                    char_list[char_before_right_quote_index] = quote
                    char_list[right_quote_index] = punc
                i = right_quote_index + 1
            else:
                i += 1
        else:
            i += 1
    return "".join(char_list)

def chunk_text(text, max_length=250, min_length=15) -> List[str]:
    """
    Splits the text into chunks with robust handling for the final chunk.
    """
    text = text.replace("’", "'").replace("‘", "'").replace("”", '"').replace("“", '"').replace("·", "").replace("…", "，")
    text = re.sub(r'《([^》]+)》', lambda m: f"《{m.group(1).replace(',', ' ')}》", text)
    # 去除所有空白符（空格、tab、换行符等）
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'([。！？!?，,])+', r'\1', text)
    text = move_punctuation_for_english_quotes(text)
    # 在连续英文与数字之间插入连字符，例如 U2 -> U-2，3M -> 3-M；纯数字或纯英文不处理
    text = re.sub(r'([A-Za-z])(?=\d)', r'\1-', text)
    text = re.sub(r'(\d)(?=[A-Za-z])', r'\1-', text)
    sentences = re.split(r'(?<=[。！？!?])', text)

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if current_chunk and len(current_chunk) + len(sentence) > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk:
        if chunks and len(current_chunk.strip()) < min_length:
            chunks[-1] += " " + current_chunk.strip()
        else:
            chunks.append(current_chunk.strip()) 

    return chunks

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the TTS model on application startup and handle shutdown.
    Aligns with webui.py model loading logic.
    """
    global tts_service
    print("--- Lifespan startup ---")

    # 1. Check if model directory exists
    if not os.path.isdir(cmd_args.model_dir):
        print(f"\033[91mError: Model directory '{cmd_args.model_dir}' not found. Please specify a valid path with --model_dir.\033[0m")
        sys.exit(1)

    # 2. Check for required model files
    config_path = os.path.join(cmd_args.model_dir, "config.yaml")
    required_files = ["bpe.model", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt", "config.yaml"]
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(cmd_args.model_dir, f))]

    if missing_files:
        print(f"\033[91mError: The following required files are missing from '{cmd_args.model_dir}':\033[0m")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    # 3. Load the TTS model
    try:
        print("Loading TTS model...")
        tts_service = TTSService(
            model_dir=cmd_args.model_dir,
            cfg_path=config_path,
            use_fp16=cmd_args.fp16,
            use_cuda_kernel=cmd_args.cuda_kernel,
        )
        print("\033[92mTTS Service loaded successfully.\033[0m")
    except Exception as e:
        print(f"\033[91mError loading TTS Service: {e}\033[0m")
        sys.exit(1)
        
    yield
    
    print("--- Lifespan shutdown ---")
    # Clean up the ML models and release the resources
    tts_service = None

app = FastAPI(title="IndexTTS API", description="A pure API for IndexTTS2 using FastAPI and ngrok.", lifespan=lifespan)

tts_service: Optional[TTSService] = None

@app.post("/api/synthesize")
async def synthesize(
    voice_path: str = Form(...),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    duration_sec: Optional[float] = Form(None),
):
    """
    Synthesize speech from text or a text file with chunking.
    Saves intermediate and final files to a specified output directory
    with a filename based on the input file.
    """
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service is not available. Check model path.")

    if not text and not file:
        raise HTTPException(status_code=400, detail="Either 'text' (form field) or 'file' (upload) must be provided.")

    input_text = ""
    base_filename = ""
    if file and file.filename:
        input_text = (await file.read()).decode("utf-8")
        base_filename = os.path.splitext(file.filename)[0]
    elif text:
        input_text = text
        base_filename = f"synthesis_{uuid.uuid4()}"

    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    # Use the specified final output directory for all files
    output_dir = "/content/drive/MyDrive/IndexTTS/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    wav_paths = []
    try:
        chunks = chunk_text(input_text, max_length=250)
        print(f"Starting synthesis for {len(chunks)} chunks, using base name: '{base_filename}'")

        for i, chunk in enumerate(chunks):
            # Intermediate files are named based on the base filename and chunk index
            out_path = os.path.join(output_dir, f"{base_filename}_{i+1}.wav")
            try:
                tts_service.infer_chunk(
                    text=chunk,
                    voice_path=voice_path,
                    out_path=out_path,
                    duration_sec=duration_sec if i == 0 else None
                )
                wav_paths.append(out_path)
                print(f"  - Chunk {i+1}/{len(chunks)} saved to: {out_path}")
            except Exception as e:
                print(f"\033[91mError processing chunk {i+1}: {e}\033[0m")
                raise HTTPException(status_code=500, detail=f"Error processing chunk {i+1}: {str(e)}")

        if not wav_paths:
            raise HTTPException(status_code=500, detail="Synthesis failed, no audio chunks were generated.")

        # Final merged file is named based on the base filename
        final_out_path = os.path.join(output_dir, f"{base_filename}.wav")
        
        merge_wavs(wav_paths, final_out_path)
        print(f"\033[92mSynthesis complete. Final audio at: {final_out_path}\033[0m")

        return {"output_path": final_out_path, "message": "Synthesis successful."}

    finally:
        # Clean up intermediate chunk files after merging
        if wav_paths:
            print("Cleaning up intermediate files...")
            for p in wav_paths:
                try:
                    os.remove(p)
                    print(f"  - Removed {p}")
                except OSError as e:
                    print(f"\033[91mError removing intermediate file {p}: {e}\033[0m")

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
    print(f"✅ Command line example:")
    print(f"  curl -X POST -F \"voice_path=/content/drive/MyDrive/Index-TTS/samples/sample1.wav\" -F \"file=@/path/on/your/local/machine/input.txt\" \\\n"
          f"  -F \"duration_sec=15\" {public_url}/api/synthesize")
    print("="*60 + "\n")

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
