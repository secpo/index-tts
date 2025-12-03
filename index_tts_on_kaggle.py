import os
import argparse
import re
import wave
import torch
from tqdm import tqdm
from typing import List

from indextts.infer_v2 import IndexTTS2

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
                    break
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
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'([。！？!?，,])+', r'\1', text)
    text = move_punctuation_for_english_quotes(text)
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
    
    print(f"\nMerging {len(paths)} chunks into {out_path}...")
    with wave.open(paths[0], "rb") as w0:
        params = w0.getparams()
    with wave.open(out_path, "wb") as out:
        out.setparams(params)
        for p in tqdm(paths, desc="Merging"):
            with wave.open(p, "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))

def main():
    parser = argparse.ArgumentParser(
        description="Batch Text-to-Speech Synthesis for Kaggle using Index-TTS 2 with deepspeed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_dir", type=str, default="/kaggle/working/index-tts/checkpoints/IndexTTS-2", help="Path to the absolute model checkpoints directory.")
    parser.add_argument("--config", type=str, default="/kaggle/working/index-tts/checkpoints/IndexTTS-2/config.yaml", help="Path to the absolute model's config.yaml file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--voice_prompt", type=str, required=True, help="Path to the reference voice audio file (WAV).")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/", help="Directory to save the output audio file.")

    args = parser.parse_args()

    print(f"Using model_dir: {args.model_dir}")
    print(f"Using config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    # --- 1. Sanity Checks ---
    if not os.path.isdir(args.model_dir):
        print(f"Error: Model directory not found at '{args.model_dir}'")
        return
    if not os.path.isfile(args.config):
        print(f"Error: Config file not found at '{args.config}'")
        return
    if not os.path.isfile(args.input_file):
        print(f"Error: Input text file not found at '{args.input_file}'")
        return
    if not os.path.isfile(args.voice_prompt):
        print(f"Error: Voice prompt file not found at '{args.voice_prompt}'")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. Load Model ---
    print("Loading TTS model...")
    try:
        tts_model = IndexTTS2(
            model_dir=args.model_dir,
            cfg_path=args.config,
            use_fp16=True,
            use_cuda_kernel=False
        )
        print(f"TTS model loaded successfully. DeepSpeed: {use_deepspeed}, Accel: {use_accel}")
    except Exception as e:
        print(f"Error loading TTS model: {e}")
        return

    # --- 3. Process Text ---
    print(f"Reading text from '{args.input_file}'...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = chunk_text(text)
    print(f"Text split into {len(chunks)} chunks.")

    # --- 4. Synthesize ---
    wav_paths = []
    base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    
    # Overall progress bar for chunks
    for i, chunk in enumerate(tqdm(chunks, desc="Total Synthesis Progress")):
        chunk_out_path = os.path.join(args.output_dir, f"{base_filename}_chunk_{i+1}.wav")
        
        try:
            tts_model.tts.infer_v2(
                text=chunk,
                spk_audio_prompt=args.voice_prompt,
                output_path=chunk_out_path,
            )
            wav_paths.append(chunk_out_path)
        except Exception as e:
            print(f"\nError processing chunk {i+1}: {e}")
            # Clean up and exit on error
            for p in wav_paths:
                os.remove(p)
            return

    # --- 5. Merge and Clean up ---
    if wav_paths:
        final_out_path = os.path.join(args.output_dir, f"{base_filename}.wav")
        merge_wavs(wav_paths, final_out_path)
        
        print("\nCleaning up intermediate chunk files...")
        for p in wav_paths:
            os.remove(p)
            
        print(f"\n✨ Synthesis complete! Final audio saved to: {final_out_path}")
    else:
        print("Synthesis failed. No audio chunks were generated.")


if __name__ == "__main__":
    main()
