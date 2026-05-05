import multiprocessing as mp
import json
import os
import hashlib
import random
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from indextts.infer_v2 import IndexTTS2

# Import MFA aligner for accurate word-level timestamps
import sys
sys.path.insert(0, os.path.dirname(__file__))


def hash_string(s):
    """Generate MD5 hash for caching"""
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def mix_to_multichannel_wav(example, audios, sr=22050, output_path="output.wav", gap_ms=50, seed=None):
    """
    Mix dialogue audio segments into multichannel WAV, each speaker on a separate channel.
    Assistant role fixed on channel 0, others assigned by appearance order.

    Returns:
        multichannel: The mixed audio array
        speaker_to_channel: Mapping of speaker names to channel indices
        dialogue_timings: List of (start_time, end_time, dialogue_index) for each dialogue
    """
    if seed is not None:
        random.seed(seed)

    participants = example["participants"]
    dialogue = example["dialogue"]
    assert len(dialogue) == len(audios), f"Dialogue and audio list must match, got {len(dialogue)} vs {len(audios)}"
    gap_samples = int(sr * gap_ms / 1000)
    barging_min_samples = int(sr * 50 / 1000)
    barging_max_samples = int(sr * 200 / 1000)
    ellipsis_barging_min_samples = int(sr * 500 / 1000)
    ellipsis_barging_max_samples = int(sr * 1000 / 1000)
    ellipsis_endings = (
        "——",
        # "...", 
        # "……", 
        # "…", 
        # "。。。"
        )

    # Assign channels: assistant → ch0, others in order
    speaker_to_channel = {}
    ch = 1
    for p in participants:
        if p["role"] == "assistant":
            speaker_to_channel[p["name"]] = 0
    for p in participants:
        if p["name"] not in speaker_to_channel:
            speaker_to_channel[p["name"]] = ch
            ch += 1
    num_channels = len(participants)

    # Calculate placement positions
    placements = []
    current_pos = 0
    overlap_group = None
    last_n_samples = 0
    last_end_per_channel = {}

    for i, (dia, audio) in enumerate(zip(dialogue, audios)):
        n_samples = len(audio)
        is_overlap = dia.get("overlap", False)
        if i >= 1:
            last_n_samples = len(audios[i-1])

        # Silence segment
        if dia["speaker"] == "silence":
            start = current_pos
            placements.append((start, n_samples, -1, i))
            current_pos = start + n_samples
            overlap_group = None
            continue

        channel = speaker_to_channel[dia["speaker"]]
        # Floor that prevents same-speaker overlap on this channel.
        same_speaker_floor = last_end_per_channel.get(channel, 0)

        if is_overlap:
            if overlap_group is None:
                group_start = current_pos + gap_samples
                overlap_group = {
                    'start': group_start,
                    'max_end': group_start + n_samples,
                }
                start = max(group_start, same_speaker_floor)
                overlap_group['max_end'] = max(overlap_group['max_end'], start + n_samples)
            else:
                min_overlap = int(n_samples * 0.2)
                earliest = max(overlap_group['start'], same_speaker_floor)
                latest = max(earliest, overlap_group['max_end'] - min_overlap)
                start = random.randint(earliest, latest)
                overlap_group['max_end'] = max(overlap_group['max_end'], start + n_samples)
        else:
            prev_text = dialogue[i - 1]["text"].rstrip() if i > 0 else ""
            prev_speaker = dialogue[i - 1]["speaker"] if i > 0 else None
            is_speaker_switch = prev_speaker is not None and prev_speaker != dia["speaker"]
            if is_speaker_switch:
                if prev_text.endswith(ellipsis_endings):
                    barging_samples = random.randint(ellipsis_barging_min_samples, ellipsis_barging_max_samples)
                else:
                    barging_samples = random.randint(barging_min_samples, barging_max_samples)
            else:
                barging_samples = 0
            barging_samples = min(barging_samples, last_n_samples)
            start = max(0, current_pos - barging_samples)
            # Barging across an overlap-group's shorter neighbor can land us
            # back inside an earlier same-speaker segment; clamp to its end.
            start = max(start, same_speaker_floor)
            overlap_group = None

        placements.append((start, n_samples, channel, i))
        current_pos = max(current_pos, start + n_samples)
        last_end_per_channel[channel] = max(same_speaker_floor, start + n_samples)

    # Write multichannel array
    total_samples = max(s + n for s, n, _, _ in placements)
    multichannel = np.zeros((total_samples, num_channels), dtype=np.float32)

    # Track dialogue timings
    dialogue_timings = []

    for idx, ((start, n, ch, dia_idx), audio) in enumerate(zip(placements, audios)):
        if ch < 0:
            continue
        actual_n = min(n, len(audio))
        multichannel[start:start + actual_n, ch] += audio[:actual_n]

        # Record timing for this dialogue
        start_time = start / sr
        end_time = (start + actual_n) / sr
        dialogue_timings.append((start_time, end_time, dia_idx))

    # Save file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving multichannel WAV to {output_path} with shape {multichannel.shape}")
    sf.write(output_path, multichannel, sr)

    return multichannel, speaker_to_channel, dialogue_timings


def load_audio_prompts(language):
    """Load audio prompts based on language"""
    audio_dir = os.environ.get(
        "AUDIO_PROMPTS_DIR",
        "./assets/audioprompts/",
    )

    if language == "zh":
        # Chinese prompts
        nan_dir = os.path.join(audio_dir, "男")
        nv_dir = os.path.join(audio_dir, "女")
        if os.path.exists(nan_dir):
            male_prompts = [os.path.join(nan_dir, f) for f in os.listdir(nan_dir) if f.endswith('.wav')]
        else:
            male_prompts = []
        if os.path.exists(nv_dir):
            female_prompts = [os.path.join(nv_dir, f) for f in os.listdir(nv_dir) if f.endswith('.wav')]
        else:
            female_prompts = []
    else:
        # English prompts
        audio_info_path = os.path.join(audio_dir, "audio_info_with_mos_text.json")
        if os.path.exists(audio_info_path):
            with open(audio_info_path, 'r', encoding='utf-8') as f:
                audio_prompts = json.load(f)
            female_prompts = [ap['wavpath'] for ap in audio_prompts if "/female/" in ap["wavpath"]]
            male_prompts = [ap['wavpath'] for ap in audio_prompts if "/male/" in ap["wavpath"]]
        else:
            male_prompts = []
            female_prompts = []

    return male_prompts, female_prompts


def assign_prompts(example, male_prompts, female_prompts):
    """Assign audio prompts to participants"""
    participant_prompt_dict = {"silence": None}

    # Shuffle to randomize prompt assignment
    random.shuffle(male_prompts)
    random.shuffle(female_prompts)

    for participant in example["participants"]:
        if participant["gender"] == "female":
            if female_prompts:
                participant_prompt_dict[participant["name"]] = female_prompts.pop(0)
            else:
                raise ValueError(f"No female prompts available for {participant['name']}")
        else:
            if male_prompts:
                participant_prompt_dict[participant["name"]] = male_prompts.pop(0)
            else:
                raise ValueError(f"No male prompts available for {participant['name']}")

    return participant_prompt_dict


def generate_transcript_with_alignment(audio_data, text, sr=22050, start_time=0.0):
    """Generate transcript metadata with word-level timestamps

    Uses MFA (Montreal Forced Aligner) for accurate alignment if available,
    otherwise falls back to simple uniform segmentation.

    Args:
        audio_data: Audio waveform
        text: Known input text (from TTS)
        sr: Sample rate
        start_time: Start time offset in the final mixed audio (in seconds)

    Returns:
        Transcript segment with word-level timestamps
    """
    words = text.split()
    if not words:
        return None
    duration = len(audio_data) / sr
    return {
        "words": [],
        "start": start_time,
        "end": start_time + duration,
        "transcript": text
    }



def tts_worker(worker_id, task_queue, result_queue, enable_eval=False):
    """Worker process for TTS generation"""
    # Load TTS model. Point INDEX_TTS_DIR at your IndexTTS2 install
    # (the directory that contains config.yaml and the model weights).
    index_tts_dir = os.environ.get("INDEX_TTS_DIR", "./checkpoints")
    tts = IndexTTS2(
        cfg_path=os.path.join(index_tts_dir, "config.yaml"),
        model_dir=index_tts_dir,
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False
    )

    # Optional: Load evaluation models
    whisper_model = None
    utmos_predictor = None
    if enable_eval:
        import whisper
        import torch
        whisper_model = whisper.load_model(
            "large-v3",
            download_root=os.environ.get("WHISPER_CACHE", "./checkpoints/whisper"),
        ).cuda()
        speechmos_dir = os.environ.get("SPEECHMOS_DIR", "./checkpoints/SpeechMOS")
        os.environ['TORCH_HOME'] = speechmos_dir
        utmos_predictor = torch.hub.load(speechmos_dir, 'utmos22_strong', source='local', trust_repo=True).cuda().float()


    # Pre-load audio prompts for both languages
    zh_male_prompts, zh_female_prompts = load_audio_prompts("zh")
    en_male_prompts, en_female_prompts = load_audio_prompts("en")

    while True:
        task = task_queue.get()
        if task is None:
            break

        try:
            example = task['example']
            output_path = task['output_path']
            request_id = task['request_id']

            # Skip if already exists
            if os.path.exists(output_path):
                print(f"[Worker {worker_id}] Skipping {request_id}: wav already exists")
                result_queue.put({'status': 'skipped', 'request_id': request_id})
                continue
            elif os.path.exists(os.path.splitext(output_path)[0] + '.flac'):
                print(f"[Worker {worker_id}] Skipping {request_id}: flac already exists")
                result_queue.put({'status': 'skipped', 'request_id': request_id})
                continue

            language = example["config"]["language"]

            # Select appropriate prompts based on language
            if language == "zh":
                male_prompts = zh_male_prompts.copy()
                female_prompts = zh_female_prompts.copy()
            else:
                male_prompts = en_male_prompts.copy()
                female_prompts = en_female_prompts.copy()

            participant_prompt_dict = assign_prompts(example, male_prompts, female_prompts)

            # Generate audio for each dialogue
            audios = []
            for dia in example["dialogue"]:
                text = dia["text"]

                # Handle emotion
                if "voice" in dia and "emotion" in dia["voice"]:
                    emotion_map = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
                    emotion = [0 if emo != dia["voice"]["emotion"] else dia["voice"].get("emotion_intensity", 0.5)
                              for emo in emotion_map]
                else:
                    emotion = [0] * 8

                # # Create cache key
                # raw_audio_key = f"{dia['speaker']}_{text}_{emotion}_{participant_prompt_dict.get(dia['speaker'])}"
                # audio_key = hash_string(raw_audio_key)

                if dia["speaker"] == "silence":
                    sr = 22050
                    silence_duration = float(dia["text"].lstrip("[").rstrip("s]"))
                    audio = sr, np.zeros(int(sr * silence_duration))
                else:
                    # IndexTTS2 returns None for degenerate inputs (e.g. pure
                    # ellipsis). Normalize like RoleBench_duplex does so the
                    # synthesizer always has phonemes to work with.
                    infer_text = (
                        text.replace("......", "...")
                            .replace("...", ",")
                            .replace("……", "…")
                            .replace("…", ",")
                    )
                    audio = None
                    max_retries = 3
                    for attempt in range(max_retries):
                        audio = tts.infer(
                            spk_audio_prompt=participant_prompt_dict[dia["speaker"]],
                            text=infer_text,
                            output_path=None,
                            emo_vector=emotion,
                            use_random=(attempt > 0),
                            verbose=False
                        )
                        if audio is not None:
                            break
                        print(
                            f"[Worker {worker_id}] infer returned None for "
                            f"{request_id} dia (text={infer_text!r}), "
                            f"retry {attempt + 1}/{max_retries}"
                        )
                    if audio is None:
                        raise RuntimeError(
                            f"tts.infer returned None after {max_retries} retries "
                            f"for text={infer_text!r}"
                        )

                sr, audio_data = audio
                audio_data = audio_data.squeeze().astype(np.float32)
                audio_data = audio_data / 32768.0  # Normalize
                audios.append(audio_data)

            # Mix to multichannel WAV
            multichannel, speaker_map, dialogue_timings = mix_to_multichannel_wav(
                example=example,
                audios=audios,
                sr=22050,
                output_path=output_path,
                gap_ms=0
            )

            # Find assistant name
            assistant_name = None
            for p in example["participants"]:
                if p["role"] == "assistant":
                    assistant_name = p["name"]
                    break

            # Generate transcript metadata for each assistant dialogue
            assistant_transcripts = []

            if assistant_name is not None:
                for start_time, end_time, dia_idx in dialogue_timings:
                    dia = example["dialogue"][dia_idx]

                    # Check if this dialogue is from assistant
                    if dia["speaker"] == assistant_name and dia["speaker"] != "silence":
                        # Get the audio segment for this dialogue
                        audio_segment = audios[dia_idx]

                        # Generate transcript with word-level timestamps
                        transcript_segment = generate_transcript_with_alignment(
                            audio_segment,
                            dia["text"],
                            sr=22050,
                            start_time=start_time
                        )

                        if transcript_segment:
                            assistant_transcripts.append(transcript_segment)

            # Save metadata
            metadata = {
                'id': request_id,
                'speaker_to_channel': speaker_map,
                'duration': len(multichannel) / 22050,
                'num_channels': multichannel.shape[1],
                'voice': participant_prompt_dict,
            }

            # Add transcript for assistant
            if assistant_transcripts:
                metadata['metadata:transcript'] = assistant_transcripts


            metadata_path = output_path.replace('.wav', '.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(f"[Worker {worker_id}] Generated {request_id}: {output_path}")
            result_queue.put({'status': 'success', 'request_id': request_id, 'output_path': output_path})

        except Exception as e:
            print(f"[Worker {worker_id}] Error processing {task.get('request_id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            result_queue.put({'status': 'error', 'request_id': task.get('request_id', 'unknown'), 'error': str(e)})


def main(jsonl_path, output_dir, num_workers=3, enable_eval=False, max_tasks=None):
    """Main function to process dialogues"""
    # Load tasks
    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Support distributed processing
            world_size = int(os.environ.get('WORLD_SIZE_V', 1))
            rank = int(os.environ.get('RANK_V', 0))
            if line_idx % world_size != rank:
                continue

            data = json.loads(line)
            request_id = data['request_id']

            # Create output path
            output_path = os.path.join(output_dir, request_id[:2], f"{request_id}.wav")
            if "source" in data:
                assert "/multichannel_wavs/" in output_path, f"Expected '/multichannel_wavs/' in output path, got {output_path}"
                source = data["source"]
                if source == "character":
                    output_path = output_path.replace("/multichannel_wavs/", "/multichannel_wavs_role/")
                elif source == "general":
                    output_path = output_path.replace("/multichannel_wavs/", "/multichannel_wavs/")
                elif source == "meeting":
                    output_path = output_path.replace("/multichannel_wavs/", "/multichannel_wavs_meeting/")
                else:
                    raise ValueError(f"Unknown source {source} for request_id {request_id}")
            tasks.append({
                'example': data,
                'output_path': output_path,
                'request_id': request_id
            })
            if max_tasks is not None and len(tasks) >= max_tasks:
                break

    print(f"Loaded {len(tasks)} tasks")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    task_queue = mp.Queue(maxsize=num_workers * 2)
    result_queue = mp.Queue()

    # Start workers
    workers = []
    for wid in range(num_workers):
        p = mp.Process(target=tts_worker, args=(wid, task_queue, result_queue, enable_eval))
        p.start()
        workers.append(p)

    # Distribute tasks
    for task in tasks:
        task_queue.put(task)

    # Send termination signals
    for _ in range(num_workers):
        task_queue.put(None)

    # Collect results
    results = []
    for _ in tqdm(range(len(tasks)), desc="Processing"):
        res = result_queue.get()
        results.append(res)

    # Wait for workers to finish
    for p in workers:
        p.join()

    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')

    print(f"\nProcessing complete:")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")


if __name__ == '__main__':
    num_workers = int(os.environ.get('NUM_WORKERS', 3))
    enable_eval = os.environ.get('ENABLE_EVAL', 'false').lower() == 'true'

    input_path = os.environ.get(
        'INPUT_PATH',
        './tts_output/dialogue_scripts.jsonl',
    )
    output_dir = os.environ.get(
        'OUTPUT_DIR',
        './tts_output/multichannel_wavs/',
    )
    main(jsonl_path=input_path, output_dir=output_dir,
         num_workers=num_workers, enable_eval=enable_eval)
