"""Inference adapter for MiniCPM-o 4.5.

Streams 1 s audio chunks through the duplex model and assembles a sentence-level
alignment from emitted text + audio chunks.

Example::

    python models/minicpm_o45.py \\
        --data-root  ./data \\
        --ckpt-dir   ~/models/openbmb/MiniCPM-o-4_5 \\
        --output-dir ./results/minicpm_o45/offline_output \\
        --ref-audio  /path/to/assistant_ref.wav \\
        --use-detailed-sys

The reference audio (any short clip of the assistant voice) is required by the
model's TTS prefill. Supply your own via ``--ref-audio``.
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModel

from . import adapter as _adapter


IN_SAMPLE_RATE = 16000
OUT_SAMPLE_RATE = 24000
DEFAULT_CHUNK_SECONDS = 1.0
SPEAKER = "SPEAKER_MAIN"
PIECE_PATTERN = re.compile(r"\s+|[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", flags=re.UNICODE)


def parse_args():
    parser = _adapter.common_parser(__doc__)
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Override AutoModel path. Default: --ckpt-dir.")
    parser.add_argument("--ref-audio", type=Path, required=True,
                        help="Reference audio for TTS prefill.")
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument("--user-channel", type=int, default=1)
    parser.add_argument("--decode-mode", default="sampling", choices=["sampling", "greedy"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2"])
    return parser.parse_args()


def load_mono_audio(audio_path, sample_rate):
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=False)
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            audio = audio[0]
        else:
            audio = audio.mean(axis=0)
    return audio.astype(np.float32, copy=False)


def load_user_audio(audio_path, sample_rate, user_channel):
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=False)
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False), {
            "input_num_channels": 1,
            "selected_user_channel": 0,
            "rms_per_input_channel": [float(np.sqrt(np.mean(audio * audio)))],
        }

    num_channels = int(audio.shape[0])
    if not 0 <= user_channel < num_channels:
        raise ValueError(
            f"--user-channel={user_channel} is out of range for {num_channels} input channel(s): {audio_path}"
        )

    rms_per_input_channel = [
        float(np.sqrt(np.mean(audio[channel] * audio[channel])))
        for channel in range(num_channels)
    ]
    selected = audio[user_channel].astype(np.float32, copy=False)
    metadata = {
        "input_num_channels": num_channels,
        "selected_user_channel": user_channel,
        "rms_per_input_channel": rms_per_input_channel,
        "selected_user_rms": float(np.sqrt(np.mean(selected * selected))),
    }
    return selected, metadata


def ensure_prompt_wav(ref_audio_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_wav_path = output_dir / "minicpmo45_ref_24k.wav"
    if prompt_wav_path.exists():
        return prompt_wav_path
    ref_audio = load_mono_audio(ref_audio_path, OUT_SAMPLE_RATE)
    sf.write(prompt_wav_path, ref_audio, OUT_SAMPLE_RATE)
    return prompt_wav_path


def get_chunk_seconds(model):
    chunk_ms = getattr(model, "CHUNK_MS", None)
    if chunk_ms is None:
        return DEFAULT_CHUNK_SECONDS
    return float(chunk_ms) / 1000.0


def iter_audio_chunks(audio, chunk_samples):
    total = len(audio)
    if total == 0:
        return
    num_chunks = (total + chunk_samples - 1) // chunk_samples
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_samples
        end = min((chunk_idx + 1) * chunk_samples, total)
        raw_chunk = audio[start:end]
        raw_num_samples = len(raw_chunk)
        if raw_num_samples < chunk_samples:
            raw_chunk = np.pad(raw_chunk, (0, chunk_samples - raw_num_samples), mode="constant", constant_values=0.0)
        yield chunk_idx, raw_chunk, raw_num_samples


def chunk_input_time_range(chunk_idx, num_samples, chunk_seconds):
    start_time = chunk_idx * chunk_seconds
    end_time = start_time + (num_samples / IN_SAMPLE_RATE)
    return start_time, end_time


def chunk_output_time_range(current_time, chunk_seconds, audio_num_samples=0):
    start_time = current_time * chunk_seconds
    audio_duration = audio_num_samples / OUT_SAMPLE_RATE if audio_num_samples > 0 else 0.0
    end_time = start_time + max(chunk_seconds, audio_duration)
    return start_time, end_time


def normalize_text(text):
    text = str(text or "")
    for marker in ("<|tts_eos|>", "</s>", "<|im_end|>", "<|endoftext|>"):
        text = text.replace(marker, "")
    return text


def text_to_pieces(text):
    text = normalize_text(text)
    if not text.strip():
        return []

    pieces = []
    for match in PIECE_PATTERN.finditer(text):
        piece = match.group(0)
        if piece.isspace():
            continue
        if re.match(r"^[A-Za-z0-9_]+$", piece, flags=re.UNICODE):
            if match.start() > 0 and text[match.start() - 1].isspace():
                piece = " " + piece
        pieces.append(piece)

    if pieces and text[:1].isspace() and not pieces[0].startswith(" "):
        pieces[0] = " " + pieces[0]
    return pieces


def pieces_to_raw_tokens(pieces):
    raw_tokens = []
    for piece in pieces:
        if piece.startswith(" "):
            raw_tokens.append("▁" + piece[1:])
        else:
            raw_tokens.append(piece)
    return raw_tokens


def decode_text_pieces(raw_tokens):
    return "".join(
        piece.replace("▁", " ")
        for piece in raw_tokens
        if piece not in {"PAD", "EPAD", "BOS", "EOS"}
    )


def append_alignment_pieces(alignments, tokens, raw_tokens, text, start_time, end_time):
    pieces = text_to_pieces(text)
    if not pieces:
        return

    duration = max(0.01, end_time - start_time)
    step = duration / len(pieces)
    for idx, piece in enumerate(pieces):
        piece_start = round(start_time + idx * step, 2)
        piece_end = round(start_time + (idx + 1) * step, 2)
        if piece_end <= piece_start:
            piece_end = round(piece_start + 0.01, 2)
        alignments.append([piece, [piece_start, piece_end], SPEAKER])
        tokens.append(piece)
    raw_tokens.extend(pieces_to_raw_tokens(pieces))


def merge_timed_audio(timed_audio):
    if not timed_audio:
        return np.zeros(0, dtype=np.float32)

    max_end = 0
    for start_time, audio in timed_audio:
        if audio is None or len(audio) == 0:
            continue
        start_sample = int(round(start_time * OUT_SAMPLE_RATE))
        max_end = max(max_end, start_sample + len(audio))

    merged = np.zeros(max_end, dtype=np.float32)
    for start_time, audio in timed_audio:
        if audio is None or len(audio) == 0:
            continue
        audio = np.asarray(audio, dtype=np.float32)
        start_sample = int(round(start_time * OUT_SAMPLE_RATE))
        end_sample = start_sample + len(audio)
        merged[start_sample:end_sample] += audio
    return np.clip(merged, -1.0, 1.0)


def pad_audio_to_duration(audio, target_duration_seconds):
    if target_duration_seconds <= 0:
        return audio
    target_samples = int(round(target_duration_seconds * OUT_SAMPLE_RATE))
    if len(audio) >= target_samples:
        return audio
    pad_width = target_samples - len(audio)
    return np.pad(audio, (0, pad_width), mode="constant", constant_values=0.0)


def write_output_wav(output_wav_path, timed_audio, target_duration_seconds, chunk_seconds):
    output_audio = merge_timed_audio(timed_audio)
    output_audio = pad_audio_to_duration(output_audio, target_duration_seconds)
    if len(output_audio) == 0:
        output_audio = np.zeros(int(round(chunk_seconds * OUT_SAMPLE_RATE)), dtype=np.float32)
    sf.write(output_wav_path, output_audio, OUT_SAMPLE_RATE)


def generation_prompt(system_prompt):
    system_prompt = str(system_prompt or "").strip()
    default_prompt = (
        "Streaming Omni Conversation. "
        "Listen to the ongoing meeting and respond naturally when appropriate."
    )
    if not system_prompt:
        return default_prompt
    return f"{default_prompt}\n{system_prompt}"


def run_one_audio(model, sample, system_prompt, output_path, args):
    audio_path = sample.audio_path
    output_wav_path = output_path.with_suffix(".wav")
    output_json_path = output_path
    debug_json_path = output_path.with_suffix(".debug.json")

    ref_audio = load_mono_audio(args.ref_audio, IN_SAMPLE_RATE)
    prompt_wav_path = ensure_prompt_wav(args.ref_audio, output_json_path.parent)
    user_audio, user_audio_metadata = load_user_audio(audio_path, IN_SAMPLE_RATE, args.user_channel)
    if len(user_audio) == 0:
        raise ValueError(f"Empty audio: {audio_path}")
    chunk_seconds = get_chunk_seconds(model)
    chunk_samples = int(round(chunk_seconds * IN_SAMPLE_RATE))
    target_output_duration_seconds = (len(user_audio) / IN_SAMPLE_RATE) + chunk_seconds

    model.prepare(
        prefix_system_prompt=generation_prompt(system_prompt),
        ref_audio=ref_audio,
        prompt_wav_path=str(prompt_wav_path),
    )

    alignments = []
    tokens = []
    raw_tokens = []
    timed_audio = []
    results_log = []

    for chunk_idx, audio_chunk, raw_num_samples in tqdm(
        iter_audio_chunks(user_audio, chunk_samples),
        total=(len(user_audio) + chunk_samples - 1) // chunk_samples,
        desc=Path(audio_path).name,
        leave=False,
    ):
        prefill_result = model.streaming_prefill(audio_waveform=audio_chunk)
        if not prefill_result.get("success", False):
            raise RuntimeError(
                f"streaming_prefill failed for {audio_path} at chunk {chunk_idx}: "
                f"{prefill_result.get('reason', '')}"
            )

        result = model.streaming_generate(
            prompt_wav_path=str(prompt_wav_path),
            max_new_speak_tokens_per_chunk=args.max_new_speak_tokens_per_chunk,
            decode_mode=args.decode_mode,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            listen_prob_scale=args.listen_prob_scale,
        )

        input_start, input_end = chunk_input_time_range(chunk_idx, raw_num_samples, chunk_seconds)
        model_current_time = float(result.get("current_time", chunk_idx + 1))
        text = normalize_text(result.get("text", ""))
        audio_waveform = result.get("audio_waveform")
        audio_length = int(len(audio_waveform)) if audio_waveform is not None else 0
        output_start, output_end = chunk_output_time_range(model_current_time, chunk_seconds, audio_length)

        if text.strip():
            append_alignment_pieces(alignments, tokens, raw_tokens, text, output_start, output_end)

        if audio_waveform is not None and audio_length > 0:
            timed_audio.append((output_start, audio_waveform))

        results_log.append(
            {
                "chunk_idx": chunk_idx,
                "prefill_success": True,
                "prefill_reason": prefill_result.get("reason", ""),
                "is_listen": bool(result.get("is_listen", False)),
                "text": text,
                "end_of_turn": bool(result.get("end_of_turn", False)),
                "current_time": output_start,
                "model_current_time": model_current_time,
                "input_time_range": [round(input_start, 2), round(input_end, 2)],
                "output_time_range": [round(output_start, 2), round(output_end, 2)],
                "audio_length": audio_length,
                "n_tokens": int(result.get("n_tokens", 0)),
                "n_tts_tokens": int(result.get("n_tts_tokens", 0)),
                "cost_all": float(result.get("cost_all", 0.0)),
            }
        )

    decoded_text = decode_text_pieces(raw_tokens)
    payload = {
        "decoded_text": decoded_text,
        "alignments": alignments,
        "tokens": tokens,
        "raw_tokens": raw_tokens,
    }
    debug_payload = {
        "decoded_text": decoded_text,
        "alignments": alignments,
        "tokens": tokens,
        "raw_tokens": raw_tokens,
        "metadata": {
            "model_path": str(args.model_path or args.ckpt_dir),
            "input_wav": str(audio_path),
            "ref_audio_path": str(args.ref_audio),
            "prompt_wav_path": str(prompt_wav_path),
            "text_prompt": system_prompt,
            "use_detailed_sys": bool(args.use_detailed_sys),
            "decode_mode": args.decode_mode,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "listen_prob_scale": args.listen_prob_scale,
            "user_channel": args.user_channel,
            "input_sample_rate": IN_SAMPLE_RATE,
            "output_sample_rate": OUT_SAMPLE_RATE,
            "chunk_seconds": chunk_seconds,
            "timing_source": "model_current_time_times_chunk_seconds",
            "num_chunks": len(results_log),
            "num_audio_segments": len(timed_audio),
            "output_duration_seconds": round(target_output_duration_seconds, 2),
            "decoded_text_num_chars": len(decoded_text),
            **user_audio_metadata,
        },
        "results_log": results_log,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with debug_json_path.open("w", encoding="utf-8") as f:
        json.dump(debug_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    write_output_wav(output_wav_path, timed_audio, target_output_duration_seconds, chunk_seconds)


def main():
    args = parse_args()
    model_path = str(args.model_path or args.ckpt_dir)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=True,
    )
    model.eval().cuda()
    model = model.as_duplex()

    def _run_one(sample, system_prompt, output_path, args_):
        run_one_audio(model, sample, system_prompt, output_path, args_)
        torch.cuda.empty_cache()

    return _adapter.run_loop(args, run_one=_run_one)


if __name__ == "__main__":
    raise SystemExit(main())
