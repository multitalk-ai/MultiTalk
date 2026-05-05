"""Inference adapter for Qwen3-Omni 30B-A3B-Instruct.

Non-streaming: feeds the entire audio in one shot and decodes thinker + talker
heads into a sentence-level alignment.

Example::

    python models/qwen3_omni.py \\
        --data-root  ./data \\
        --ckpt-dir   ~/models/Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --output-dir ./results/qwen3_omni/offline_output \\
        --use-detailed-sys

Speaker voice is auto-selected per language (Ethan for EN, Chelsie for ZH);
override via ``--speaker``.
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

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from . import adapter as _adapter


IN_SAMPLE_RATE = 16000
OUT_SAMPLE_RATE = 24000
SPEAKER = "SPEAKER_MAIN"
PIECE_PATTERN = re.compile(r"\s+|[㐀-䶿一-鿿]|[A-Za-z0-9_]+|[^\w\s]", flags=re.UNICODE)
USE_AUDIO_IN_VIDEO = False

LANGUAGE_TO_SPEAKER = {
    "en": "Ethan",
    "zh": "Chelsie",
}
DEFAULT_SPEAKER = "Ethan"


def parse_args():
    parser = _adapter.common_parser(__doc__)
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Override Qwen3-Omni model path. Default: --ckpt-dir.")
    parser.add_argument("--user-channel", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--speaker", type=str, default=None,
                        help="Override speaker voice. If unset, chosen per language.")
    parser.add_argument("--attn-implementation", default="flash_attention_2",
                        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--no-audio-output", action="store_true",
                        help="Disable audio generation; only text is produced.")
    parser.add_argument("--do-sample", action="store_true",
                        help="Enable sampling (otherwise greedy).")
    parser.add_argument("--thinker-temperature", type=float, default=0.6)
    parser.add_argument("--thinker-top-p", type=float, default=0.95)
    parser.add_argument("--thinker-top-k", type=int, default=20)
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-p", type=float, default=0.95)
    parser.add_argument("--talker-top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed torch for reproducible sampling.")
    return parser.parse_args()


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


def build_alignments(text, start_time, end_time):
    pieces = text_to_pieces(text)
    alignments = []
    tokens = []
    if not pieces:
        return alignments, tokens, []

    duration = max(0.01, end_time - start_time)
    step = duration / len(pieces)
    for idx, piece in enumerate(pieces):
        piece_start = round(start_time + idx * step, 2)
        piece_end = round(start_time + (idx + 1) * step, 2)
        if piece_end <= piece_start:
            piece_end = round(piece_start + 0.01, 2)
        alignments.append([piece, [piece_start, piece_end], SPEAKER])
        tokens.append(piece)
    raw_tokens = pieces_to_raw_tokens(pieces)
    return alignments, tokens, raw_tokens


def pad_audio_to_duration(audio, target_duration_seconds):
    if target_duration_seconds <= 0:
        return audio
    target_samples = int(round(target_duration_seconds * OUT_SAMPLE_RATE))
    if len(audio) >= target_samples:
        return audio
    pad_width = target_samples - len(audio)
    return np.pad(audio, (0, pad_width), mode="constant", constant_values=0.0)


def write_output_wav(output_wav_path, audio_np, target_duration_seconds):
    audio_np = np.clip(audio_np.astype(np.float32), -1.0, 1.0)
    audio_np = pad_audio_to_duration(audio_np, target_duration_seconds)
    if len(audio_np) == 0:
        audio_np = np.zeros(int(round(target_duration_seconds * OUT_SAMPLE_RATE)), dtype=np.float32)
    sf.write(output_wav_path, audio_np, OUT_SAMPLE_RATE)


def build_conversation(user_audio_path, system_prompt):
    conversation = []
    system_prompt = str(system_prompt or "").strip()
    if system_prompt:
        conversation.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })
    conversation.append({
        "role": "user",
        "content": [{"type": "audio", "audio": str(user_audio_path)}],
    })
    return conversation


def prepare_user_audio_wav(audio_path, output_dir, user_channel):
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / "input_user.wav"
    user_audio, user_audio_metadata = load_user_audio(audio_path, IN_SAMPLE_RATE, user_channel)
    if len(user_audio) == 0:
        raise ValueError(f"Empty audio: {audio_path}")
    sf.write(prepared_path, user_audio, IN_SAMPLE_RATE)
    return prepared_path, user_audio, user_audio_metadata


def pick_speaker(language, override):
    if override:
        return override
    return LANGUAGE_TO_SPEAKER.get(language, DEFAULT_SPEAKER)


def run_one_audio(
    model,
    processor,
    sample,
    system_prompt,
    output_path,
    args,
):
    audio_path = sample.audio_path
    output_wav_path = output_path.with_suffix(".wav")
    output_json_path = output_path
    debug_json_path = output_path.with_suffix(".debug.json")
    output_dir = output_path.parent

    prepared_path, user_audio, user_audio_metadata = prepare_user_audio_wav(
        audio_path, output_dir, args.user_channel
    )
    input_duration_seconds = float(len(user_audio)) / IN_SAMPLE_RATE
    target_output_duration_seconds = input_duration_seconds

    conversation = build_conversation(prepared_path, system_prompt)

    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text_prompt,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    speaker = pick_speaker(sample.language.value, args.speaker)

    gen_kwargs = {
        "speaker": speaker,
        "thinker_return_dict_in_generate": True,
        "use_audio_in_video": USE_AUDIO_IN_VIDEO,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.do_sample:
        gen_kwargs.update({
            "thinker_do_sample": True,
            "thinker_temperature": args.thinker_temperature,
            "thinker_top_p": args.thinker_top_p,
            "thinker_top_k": args.thinker_top_k,
            "talker_do_sample": True,
            "talker_temperature": args.talker_temperature,
            "talker_top_p": args.talker_top_p,
            "talker_top_k": args.talker_top_k,
        })
    else:
        gen_kwargs.update({"thinker_do_sample": False, "talker_do_sample": False})
    if args.no_audio_output:
        gen_kwargs["return_audio"] = False

    with torch.no_grad():
        gen_out = model.generate(**inputs, **gen_kwargs)

    if args.no_audio_output:
        text_ids = gen_out
        audio_tensor = None
    else:
        text_ids, audio_tensor = gen_out

    decoded_batch = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    decoded_text = normalize_text(decoded_batch[0] if decoded_batch else "")

    if audio_tensor is not None:
        audio_np = audio_tensor.reshape(-1).detach().cpu().numpy().astype(np.float32)
    else:
        audio_np = np.zeros(0, dtype=np.float32)
    audio_duration_seconds = float(len(audio_np)) / OUT_SAMPLE_RATE

    alignment_end = max(audio_duration_seconds, 0.01)
    alignments, tokens, raw_tokens = build_alignments(
        decoded_text, start_time=0.0, end_time=alignment_end
    )

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
            "prepared_user_wav": str(prepared_path),
            "text_prompt": system_prompt,
            "use_detailed_sys": bool(args.use_detailed_sys),
            "speaker": speaker,
            "max_new_tokens": args.max_new_tokens,
            "user_channel": args.user_channel,
            "input_sample_rate": IN_SAMPLE_RATE,
            "output_sample_rate": OUT_SAMPLE_RATE,
            "input_duration_seconds": round(input_duration_seconds, 2),
            "output_duration_seconds": round(audio_duration_seconds, 2),
            "decoded_text_num_chars": len(decoded_text),
            "sample_id": sample.sample_id,
            "language": sample.language.value,
            "meeting_id": sample.meeting_id,
            **user_audio_metadata,
        },
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with debug_json_path.open("w", encoding="utf-8") as f:
        json.dump(debug_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    write_output_wav(output_wav_path, audio_np, target_output_duration_seconds)


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))

    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + rank)

    model_path = str(args.model_path or args.ckpt_dir)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    def _run_one(sample, system_prompt, output_path, args_):
        try:
            run_one_audio(model, processor, sample, system_prompt, output_path, args_)
        finally:
            torch.cuda.empty_cache()

    return _adapter.run_loop(args, run_one=_run_one)


if __name__ == "__main__":
    raise SystemExit(main())
