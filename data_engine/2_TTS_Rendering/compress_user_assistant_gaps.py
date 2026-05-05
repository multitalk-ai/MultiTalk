import os
import numpy as np
import orjson
import soundfile as sf
from tqdm import tqdm
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor

VERBOSE = True
OVERWRITE_EXISTING = True
SAVE_RECOVERED_DEBUG_AUDIO = False
REMOVE_ORIGINAL_IF_RECOVERY_PASS = True
ENABLE_CLUSTER_MODE = True  # 若 True，则原文件已含 A/U 重叠时用 cluster 原子化策略压缩；否则跳过
CLUSTER_COVERAGE_THRESHOLD = 0.8  # 检测出的簇占原音频比例超过该阈值则放弃压缩（保守）


def _compressed_output_paths(file_path):
    json_path = file_path.replace(".flac", ".json").replace(".wav", ".json")
    return (
        json_path,
        file_path.replace(".flac", "_compressed.flac").replace(".wav", "_compressed.flac"),
        json_path.replace(".json", "_compressed.json"),
    )


def _remove_file_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _cleanup_incomplete_compressed_outputs(
    file_path: str,
    compressed_audio_path: str,
    compressed_json_path: str,
) -> None:
    audio_exists = os.path.exists(compressed_audio_path)
    json_exists = os.path.exists(compressed_json_path)
    if audio_exists == json_exists:
        return
    print(f"Removing incomplete compressed outputs for {file_path}")
    _remove_file_if_exists(compressed_audio_path)
    _remove_file_if_exists(compressed_json_path)


def _atomic_write_audio(dest_path: str, audio: np.ndarray, sample_rate: int) -> None:
    directory = os.path.dirname(dest_path) or "."
    _, ext = os.path.splitext(dest_path)
    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(dest_path)}.",
        suffix=ext,
    )
    os.close(fd)
    try:
        sf.write(tmp_path, audio, sample_rate)
        with open(tmp_path, "r+b") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest_path)
    except Exception:
        _remove_file_if_exists(tmp_path)
        raise


def _atomic_write_bytes(dest_path: str, payload: bytes) -> None:
    directory = os.path.dirname(dest_path) or "."
    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(dest_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest_path)
    except Exception:
        _remove_file_if_exists(tmp_path)
        raise


def _should_skip_file(file_path):
    if (not file_path.endswith(".flac") and not file_path.endswith(".wav")) or file_path.endswith("_compressed.flac") or not os.path.exists(file_path):
        print(f"Skipping {file_path} due to invalid extension or missing file.")
        return True
    if OVERWRITE_EXISTING:
        return False
    _, new_stereo_path, new_stereo_json_path = _compressed_output_paths(file_path)
    return os.path.exists(new_stereo_path) and os.path.exists(new_stereo_json_path)

def check_have_overlap(assistant_audio, user_audio):
    assert len(assistant_audio) == len(user_audio), "Audio lengths do not match"
    a = np.asarray(assistant_audio)
    u = np.asarray(user_audio)
    return bool(np.any((abs(a) > 0) & (abs(u) > 0)))


def detect_overlap_clusters(
    assistant_audio,
    user_audio,
    transcript: list[dict],
    sample_rate: int,
    ratio: float = 1e-4,
    floor: float = 1e-10,
) -> list[dict]:
    """识别简单 A-seg / U-gap 交替模型无法忠实表示的区域，作为必须双通道原子搬动的簇。

    简单模型的覆盖规则：
      - ch0 只覆盖 ⋃ [seg.start, seg.end)
      - ch1 只覆盖各 head/gap/tail 切片内 [first_active, last_active+1)

    因此 cluster 的种子是"简单模型覆盖不到的活跃采样"：
        seed = (active_u & in_segment) | (active_a & ~in_segment)

    这自然涵盖所有"同采样双通道活跃"的经典 overlap 情形，同时还捕获
    assistant 在 seg 尾部先停、user 独自在 seg 时段内说话这类过去会被静默丢弃的案例。

    Pipeline:
      1. 种子 = 上述 seed
      2. 向两侧扩张到"双通道同时静音"边界（不截断任一通道的连续发声）
      3. 迭代扩张至完全包含任何与簇在时间上相交的 transcript segment，合并触碰的簇
      4. 绑定簇内完整包含的 transcript 索引；若不连续则报错

    返回每个簇：
      - src_start_samp / src_end_samp：半开区间，单位=采样
      - transcript_indices：簇内完全包含的 transcript segment 索引（按顺序且连续）；
        可能为空，表示种子全部位于 transcript 未覆盖的区域，调用方需自行决定如何处理
    """
    a = np.asarray(assistant_audio)
    u = np.asarray(user_audio)
    n = min(len(a), len(u))
    if n == 0:
        return []
    a = a[:n]
    u = u[:n]

    abs_a = np.abs(a)
    abs_u = np.abs(u)
    peak = float(max(abs_a.max(initial=0.0), abs_u.max(initial=0.0)))
    if peak < floor:
        return []
    eps = max(peak * ratio, floor)

    active_a = abs_a > eps
    active_u = abs_u > eps

    # 构建 segment 时间覆盖掩码（ch0 的简单模型覆盖域）
    in_segment = np.zeros(n, dtype=bool)
    for seg in transcript:
        ss = int(round(float(seg["start"]) * sample_rate))
        se = int(round(float(seg["end"]) * sample_rate))
        ss = max(0, ss)
        se = min(n, se)
        if ss < se:
            in_segment[ss:se] = True

    # 种子集：简单模型覆盖不到的活跃采样
    seed = (active_u & in_segment) | (active_a & ~in_segment)
    if not seed.any():
        return []

    not_both_silent = active_a | active_u

    # 1-2) 从 seed 向两侧扩张到双静音边界；向量化实现
    #      连通段由 not_both_silent 的 True 游程决定，保留包含至少一个种子的段
    diff = np.diff(not_both_silent.astype(np.int8), prepend=0, append=0)
    run_starts = np.flatnonzero(diff == 1)
    run_ends = np.flatnonzero(diff == -1)
    assert len(run_starts) == len(run_ends), "run-length detection mismatch"

    clusters_samp: list[list[int]] = []
    for rs, re in zip(run_starts, run_ends):
        if seed[rs:re].any():
            clusters_samp.append([int(rs), int(re)])

    if not clusters_samp:
        return []

    # 3) 迭代扩张至完全包含交叠的 transcript segment + 合并
    def seg_samp_range(seg):
        return (
            int(round(float(seg["start"]) * sample_rate)),
            int(round(float(seg["end"]) * sample_rate)),
        )

    seg_ranges = [seg_samp_range(s) for s in transcript]

    changed = True
    while changed:
        changed = False
        for idx in range(len(clusters_samp)):
            cs, ce = clusters_samp[idx]
            for ss, se in seg_ranges:
                if ss < ce and se > cs:  # 半开区间相交
                    new_s = min(cs, ss)
                    new_e = max(ce, se)
                    if (new_s, new_e) != (cs, ce):
                        clusters_samp[idx] = [new_s, new_e]
                        cs, ce = new_s, new_e
                        changed = True
        clusters_samp.sort()
        merged: list[list[int]] = []
        for c in clusters_samp:
            if merged and c[0] <= merged[-1][1]:
                if c[1] > merged[-1][1]:
                    merged[-1][1] = c[1]
                    changed = True
            else:
                merged.append(list(c))
        if len(merged) != len(clusters_samp):
            changed = True
        clusters_samp = merged

    # 4) 绑定 transcript 索引，校验连续性
    result: list[dict] = []
    for cs, ce in clusters_samp:
        indices = [j for j, (ss, se) in enumerate(seg_ranges) if ss < ce and se > cs]
        if indices and indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(
                f"cluster [{cs}, {ce}) encloses non-contiguous transcript indices {indices}"
            )
        # 簇必须完整覆盖其包含的 segment
        for j in indices:
            ss, se = seg_ranges[j]
            assert cs <= ss and ce >= se, (
                f"cluster [{cs}, {ce}) does not fully enclose segment {j} [{ss}, {se})"
            )
        result.append(
            {
                "src_start_samp": int(cs),
                "src_end_samp": int(ce),
                "transcript_indices": indices,
            }
        )
    return result


def add_alignments_to_metadata(metadata):
    # 假设 metadata 中有一个 "segments" 字段是一个列表，每个元素包含 "start_time" 和 "end_time"
    alignments = []
    for segment in metadata["metadata:transcript"]:
        for word_info in segment["words"]:
            alignments.append([
                word_info["word"],
                [word_info["start"], word_info["end"]], 
                "SPEAKER_MAIN"
                ])
    
    metadata["alignments"] = alignments

    return metadata

ellipsis_endings = (
     "——",
    # "...", "……", "…", "。。。"
    )

def _validate_transcript_for_compression(transcript: list[dict], sample_rate: int) -> str | None:
    if not transcript:
        return "empty transcript"

    prev_end = None
    tolerance = 0.01  # 10 ms tolerance for detecting overlaps or out-of-order segments
    for i, segment in enumerate(transcript):
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            return f"segment {i} has invalid start/end"

        if end <= start:
            return f"segment {i} has non-positive duration: start={start:.6f}, end={end:.6f}"
        if prev_end is not None and start < prev_end - tolerance:
            return (
                f"segment {i} overlaps or is out of order: "
                f"start={start:.6f}, previous_end={prev_end:.6f} "
                f"with {prev_end - start:.3f}s overlap"
            )
        prev_end = end

    return None


def _mapping_item(raw_start_samp: int, dst_start_samp: int, actual: int, sr: int) -> dict:
    raw_end_samp = raw_start_samp + actual
    dst_end_samp = dst_start_samp + actual
    return {
        "raw": [float(raw_start_samp / sr), float(raw_end_samp / sr)],
        "dst": [float(dst_start_samp / sr), float(dst_end_samp / sr)],
        "raw_samples": [int(raw_start_samp), int(raw_end_samp)],
        "dst_samples": [int(dst_start_samp), int(dst_end_samp)],
    }

def _sample_ranges_from_mapping(item: dict) -> tuple[int, int, int, int]:
    if "raw_samples" not in item or "dst_samples" not in item:
        raise ValueError(
            "legacy compressed metadata is not exactly recoverable; recompress source audio"
        )
    raw_start_samp, raw_end_samp = (int(v) for v in item["raw_samples"])
    dst_start_samp, dst_end_samp = (int(v) for v in item["dst_samples"])
    if raw_end_samp < raw_start_samp or dst_end_samp < dst_start_samp:
        raise ValueError(f"invalid mapping range: {item}")
    if (raw_end_samp - raw_start_samp) != (dst_end_samp - dst_start_samp):
        raise ValueError(f"mismatched mapping lengths: {item}")
    return raw_start_samp, raw_end_samp, dst_start_samp, dst_end_samp


def _clip_mappings_to_trim(mapping_items: list[dict], trim_samp: int, sr: int) -> list[dict]:
    clipped_items = []
    for item in mapping_items:
        raw_start_samp, raw_end_samp, dst_start_samp, dst_end_samp = _sample_ranges_from_mapping(item)
        clipped_end_samp = min(dst_end_samp, trim_samp)
        actual = clipped_end_samp - dst_start_samp
        if actual <= 0:
            continue
        clipped_items.append(_mapping_item(raw_start_samp, dst_start_samp, actual, sr))
    return clipped_items


def _build_recovery_masks(
    num_samples: int,
    mapping_items: list[dict],
    boundary_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    core_mask = np.zeros(int(num_samples), dtype=bool)
    boundary_mask = np.zeros(int(num_samples), dtype=bool)
    for item in mapping_items:
        raw_start_samp, raw_end_samp, _, _ = _sample_ranges_from_mapping(item)
        if raw_end_samp > num_samples:
            raise ValueError(f"mapping exceeds raw audio length: {item}")
        if boundary_samples <= 0:
            core_mask[raw_start_samp:raw_end_samp] = True
            continue
        core_start_samp = min(raw_end_samp, raw_start_samp + boundary_samples)
        core_end_samp = max(core_start_samp, raw_end_samp - boundary_samples)
        boundary_mask[raw_start_samp:core_start_samp] = True
        boundary_mask[core_end_samp:raw_end_samp] = True
        core_mask[core_start_samp:core_end_samp] = True
    return core_mask, boundary_mask


def _build_recoverable_mask(num_samples: int, mapping_items: list[dict]) -> np.ndarray:
    core_mask, boundary_mask = _build_recovery_masks(num_samples, mapping_items, boundary_samples=0)
    return core_mask | boundary_mask


def evaluate_recovery(
    raw_audio,
    recovered_raw_audio,
    fake_raw_assistant,
    fake_raw_user,
    atol: float = 1e-4,
    boundary_samples: int = 1,
) -> dict:
    diff = np.abs(raw_audio - recovered_raw_audio)
    assistant_core_mask, assistant_boundary_mask = _build_recovery_masks(
        len(raw_audio), fake_raw_assistant, boundary_samples=boundary_samples
    )
    user_core_mask, user_boundary_mask = _build_recovery_masks(
        len(raw_audio), fake_raw_user, boundary_samples=boundary_samples
    )
    recoverable_mask = np.stack([assistant_core_mask, user_core_mask], axis=1)
    boundary_mask = np.stack([assistant_boundary_mask, user_boundary_mask], axis=1)
    unmapped_mask = ~(recoverable_mask | boundary_mask)

    recoverable_count = int(np.count_nonzero(recoverable_mask))
    boundary_count = int(np.count_nonzero(boundary_mask))
    unrecoverable_count = int(np.count_nonzero(unmapped_mask))

    recoverable_max_abs_diff = float(diff[recoverable_mask].max()) if recoverable_count else 0.0
    boundary_max_abs_diff = float(diff[boundary_mask].max()) if boundary_count else 0.0
    unrecoverable_max_abs_diff = float(diff[unmapped_mask].max()) if unrecoverable_count else 0.0
    recoverable_num_over_atol = int(np.count_nonzero(diff[recoverable_mask] > atol)) if recoverable_count else 0
    boundary_num_over_atol = int(np.count_nonzero(diff[boundary_mask] > atol)) if boundary_count else 0
    unrecoverable_num_over_atol = int(np.count_nonzero(diff[unmapped_mask] > atol)) if unrecoverable_count else 0
    # recoverable_pass: mapped 区域的 raw ↔ dst 必须 bit-exact
    recoverable_pass = recoverable_num_over_atol == 0
    # unmapped_significant_pass: 任何未 mapped 采样若在 raw 里有 |value| > atol，就视为
    # 压缩管线丢失了可听内容（即便 mapping 内部完全正确）。这是 recoverable_pass 之外
    # 的独立安全网，用于在 detect_overlap_clusters 或 gap 检测遗漏时兜底。
    unmapped_significant_pass = unrecoverable_num_over_atol == 0
    all_pass = recoverable_pass and unmapped_significant_pass

    return {
        "recoverable_pass": recoverable_pass,
        "unmapped_significant_pass": unmapped_significant_pass,
        "all_pass": all_pass,
        "recoverable_count": recoverable_count,
        "boundary_count": boundary_count,
        "unrecoverable_count": unrecoverable_count,
        "recoverable_max_abs_diff": recoverable_max_abs_diff,
        "boundary_max_abs_diff": boundary_max_abs_diff,
        "unrecoverable_max_abs_diff": unrecoverable_max_abs_diff,
        "recoverable_num_over_atol": recoverable_num_over_atol,
        "boundary_num_over_atol": boundary_num_over_atol,
        "unrecoverable_num_over_atol": unrecoverable_num_over_atol,
    }


def recover_raw_audio(new_stereo, fake_raw_assistant, fake_raw_user, raw_num_samples):
    recovered = np.zeros((int(raw_num_samples), 2), dtype=new_stereo.dtype)

    for item in fake_raw_assistant:
        raw_start_samp, raw_end_samp, dst_start_samp, dst_end_samp = _sample_ranges_from_mapping(item)
        if raw_end_samp > raw_num_samples or dst_end_samp > len(new_stereo):
            raise ValueError(f"assistant mapping out of bounds: {item}")
        recovered[raw_start_samp:raw_end_samp, 0] = new_stereo[dst_start_samp:dst_end_samp, 0]

    for item in fake_raw_user:
        raw_start_samp, raw_end_samp, dst_start_samp, dst_end_samp = _sample_ranges_from_mapping(item)
        if raw_end_samp > raw_num_samples or dst_end_samp > len(new_stereo):
            raise ValueError(f"user mapping out of bounds: {item}")
        recovered[raw_start_samp:raw_end_samp, 1] = new_stereo[dst_start_samp:dst_end_samp, 1]

    return recovered

def compress_gaps(
    assistant_audio,
    user_audio,
    transcript: list[dict],
    target_overlap_sec: tuple[float, float]|float = 0.3,
    target_overlap_sec_large: tuple[float, float]|float = 0.6,
    sample_rate: int = 22050,
    clusters: list[dict] | None = None,
) -> tuple[np.ndarray, list[dict], list[dict], list[dict], dict]:
    """
    压缩多轮对话中各 turn 之间的静音，
    使每个 turn 边界恰好有 target_overlap_sec 的重叠。
    若 assistant segment 文本以省略号结尾，该 A→U 边界使用
    target_overlap_sec_large。

    特殊处理：
    - 当某段语音时长短于目标重叠时，自动缩小有效重叠量
    - 同一说话人的音频片段之间绝不重叠
    - 若传入 clusters（来自 detect_overlap_clusters），每个簇作为一个
      原子虚拟段：整体平移、双声道成对搬动、不改变簇内相对时序；簇的两侧
      边界禁用跨说话人 overlap，避免 user 通道在 dst 中累加失真（recovery
      会直接失败）。每个簇的 transcript_indices 必须非空且连续。

    返回
    ----
    stereo            : np.ndarray shape (N, 2), float32
    updated_transcript: 时间戳已对齐到输出音频
    fake_raw_assistant: 用于恢复原始音频的辅助信息
    fake_raw_user: 用于恢复原始音频的辅助信息
    overlap_stats     : 有效 overlap 统计，用于判断是否值得压缩
    """
    fake_raw_user, fake_raw_assistant = [], []
    a = np.asarray(assistant_audio, dtype=np.float32)
    u = np.asarray(user_audio, dtype=np.float32)
    n = len(transcript)
    effective_overlap_total = 0.0
    nonzero_overlap_count = 0

    total = max(len(a), len(u))
    if len(a) < total:
        a = np.pad(a, (0, total - len(a)))
    if len(u) < total:
        u = np.pad(u, (0, total - len(u)))

    sr = sample_rate

    def _sec2samp(sec: float) -> int:
        return int(round(sec * sr))

    def _pick_ov_from_text(text: str) -> float:
        assert len(text) > 0, "Empty transcript text"
        target = target_overlap_sec_large if text.endswith(ellipsis_endings) else target_overlap_sec
        return target if isinstance(target, float) else random.uniform(*target)

    def _first_last_nonzero(arr: np.ndarray,
                          ratio: float = 0.0001,
                          floor: float = 1e-10):
        """
        返回 (first_idx, last_idx)；若全静音返回 None。

        Parameters
        ----------
        ratio : float
            相对于峰值的比例门限，默认 0.0001（即峰值的 0.01%，约 -80 dB）。
        floor : float
            绝对下限，防止峰值本身极小时阈值退化到 0。
        """
        if arr.size == 0:
            return None
        abs_arr = np.abs(arr)
        peak = abs_arr.max()
        if peak < floor:
            return None
        eps = max(peak * ratio, floor)
        mask = abs_arr > eps
        if not mask.any():
            return None
        first = np.argmax(mask)
        last = len(mask) - 1 - np.argmax(mask[::-1])
        return first, last

    # ── 0. 将 transcript + clusters 合成为 virtual segments ─────────────
    # 不带 clusters 时 virtuals 与 transcript 一一对应，后续逻辑等价于旧实现
    if clusters is None:
        clusters = []
    in_cluster = [-1] * n
    for ci, c in enumerate(clusters):
        indices = list(c["transcript_indices"])
        if not indices:
            raise ValueError(
                f"cluster {ci} [{c['src_start_samp']}, {c['src_end_samp']}) has no enclosed "
                "transcript segments; caller must filter these out before calling compress_gaps"
            )
        for idx in indices:
            if not (0 <= idx < n):
                raise ValueError(f"cluster {ci} has invalid transcript index {idx}")
            if in_cluster[idx] != -1:
                raise ValueError(
                    f"transcript index {idx} belongs to multiple clusters "
                    f"({in_cluster[idx]} and {ci})"
                )
            in_cluster[idx] = ci

    virtuals: list[dict] = []
    i = 0
    while i < n:
        ci = in_cluster[i]
        if ci >= 0:
            c = clusters[ci]
            indices = list(c["transcript_indices"])
            if indices[0] != i:
                raise ValueError(
                    f"cluster {ci} not entered at its first transcript index "
                    f"(expected {indices[0]}, got {i})"
                )
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError(
                    f"cluster {ci} transcript indices are not contiguous: {indices}"
                )
            src_start_samp = int(c["src_start_samp"])
            src_end_samp = int(c["src_end_samp"])
            if src_start_samp > _sec2samp(transcript[indices[0]]["start"]):
                raise ValueError(f"cluster {ci} does not enclose first segment start")
            if src_end_samp < _sec2samp(transcript[indices[-1]]["end"]):
                raise ValueError(f"cluster {ci} does not enclose last segment end")
            virtuals.append({
                "is_cluster": True,
                "src_start_sec": src_start_samp / sr,
                "src_end_sec": src_end_samp / sr,
                "src_start_samp": src_start_samp,
                "src_end_samp": src_end_samp,
                "indices": indices,
                "boundary_text": transcript[indices[-1]]["transcript"],
            })
            i = indices[-1] + 1
        else:
            start_sec = float(transcript[i]["start"])
            end_sec = float(transcript[i]["end"])
            virtuals.append({
                "is_cluster": False,
                "src_start_sec": start_sec,
                "src_end_sec": end_sec,
                "src_start_samp": _sec2samp(start_sec),
                "src_end_samp": _sec2samp(end_sec),
                "indices": [i],
                "boundary_text": transcript[i]["transcript"],
            })
            i += 1

    nv = len(virtuals)

    # ── 1. 检测每个 gap 内 user 的首/末活跃采样 ─────────────────────────
    gaps_src: list[tuple[float, float]] = []
    for vi in range(nv - 1):
        gs = virtuals[vi]["src_end_samp"]
        ge = virtuals[vi + 1]["src_start_samp"]
        result = _first_last_nonzero(u[gs:ge]) if ge > gs else None
        if result is not None:
            u_s = (gs + result[0]) / sr
            u_e = (gs + result[1] + 1) / sr
        else:
            mid = (virtuals[vi]["src_end_sec"] + virtuals[vi + 1]["src_start_sec"]) / 2
            u_s = u_e = mid
        gaps_src.append((u_s, u_e))

    # ── 1b. 检测 head gap（录音开头 → 第一个 virtual）──────────────────
    head_ge = virtuals[0]["src_start_samp"]
    head_result = _first_last_nonzero(u[:head_ge]) if head_ge > 0 else None
    if head_result is not None:
        u_head_src_s = head_result[0] / sr
        u_head_src_e = (head_result[1] + 1) / sr
        u_head_dur = u_head_src_e - u_head_src_s
    else:
        u_head_src_s = u_head_src_e = u_head_dur = 0.0

    # ── 2. 递推新时间线（保证同一说话人不重叠）───────────────────────────
    a_dst = [0.0] * nv
    u_dst = [0.0] * (nv - 1)

    # ★ 追踪 user 通道最后音频结束位置，防止 user 自身重叠
    u_channel_end = 0.0

    # head gap → 第一个 virtual 留 overlap
    if u_head_dur > 0:
        v_dur_0 = virtuals[0]["src_end_sec"] - virtuals[0]["src_start_sec"]
        if virtuals[0]["is_cluster"]:
            # 簇内 user 声道已可能有内容，不能和 head user 在 dst 同一通道叠加
            ov_head = 0.0
        else:
            ov_head = min(
                target_overlap_sec if isinstance(target_overlap_sec, float) else target_overlap_sec[0],
                u_head_dur,
                v_dur_0,
            )
        effective_overlap_total += ov_head
        if ov_head > 0:
            nonzero_overlap_count += 1
        a_dst[0] = max(0.0, u_head_dur - ov_head)
        u_channel_end = u_head_dur  # head user 从 dst=0 开始

    for vi in range(nv):
        v_dur = virtuals[vi]["src_end_sec"] - virtuals[vi]["src_start_sec"]
        v_end_new = a_dst[vi] + v_dur
        if vi < nv - 1:
            # A→U 边界：当前 virtual 是簇 → 禁用 overlap（簇 user 声道有内容）
            if virtuals[vi]["is_cluster"]:
                ov_au = 0.0
                ov_au_eff = 0.0
            else:
                ov_au = _pick_ov_from_text(virtuals[vi]["boundary_text"].strip())
                # ★ assistant 短于目标重叠时，缩小有效重叠
                ov_au_eff = min(ov_au, v_dur)
            effective_overlap_total += ov_au_eff
            if ov_au_eff > 0:
                nonzero_overlap_count += 1

            u_dur = gaps_src[vi][1] - gaps_src[vi][0]

            # ★ user 起始：允许与 assistant 跨说话人重叠，
            #   但绝不与前一段 user 音频重叠
            u_dst[vi] = max(v_end_new - ov_au_eff, u_channel_end)

            u_end_new = u_dst[vi] + u_dur
            u_channel_end = u_end_new  # 更新 user 通道末尾

            # U→A 边界：下一 virtual 是簇 → 禁用 overlap（簇 user 声道有内容）
            if virtuals[vi + 1]["is_cluster"]:
                ov_ua = 0.0
                ov_ua_eff = 0.0
            else:
                ov_ua = target_overlap_sec if isinstance(target_overlap_sec, float) else random.uniform(*target_overlap_sec)
                # ★ user 短于目标重叠时，缩小有效重叠
                ov_ua_eff = min(ov_ua, u_dur)
            effective_overlap_total += ov_ua_eff
            if ov_ua_eff > 0:
                nonzero_overlap_count += 1

            # ★ 下一段 assistant 起始：允许与 user 跨说话人重叠，
            #   但绝不与当前 virtual 段重叠
            a_dst[vi + 1] = max(v_end_new, u_end_new - ov_ua_eff)

            if VERBOSE:
                print(
                    f"Segment {vi}: A→U ov={ov_au_eff:.3f}s (target {ov_au:.2f}), "
                    f"U→A ov={ov_ua_eff:.3f}s (target {ov_ua:.2f})"
                )

    # ── 3. 更新 transcript（手动浅拷贝）──────────────────────────────────
    updated: list[dict] = [None] * n
    for vi, v in enumerate(virtuals):
        shift = a_dst[vi] - v["src_start_sec"]
        for j in v["indices"]:
            seg = transcript[j]
            new_seg = {
                **seg,
                "start": float(round(seg["start"] + shift, 6)),
                "end": float(round(seg["end"] + shift, 6)),
            }
            if "words" in seg:
                new_seg["words"] = [
                    {
                        **w,
                        "start": float(round(w["start"] + shift, 6)),
                        "end": float(round(w["end"] + shift, 6)),
                    }
                    for w in seg["words"]
                ]
            updated[j] = new_seg

    # ── 4. 重建输出音频 ─────────────────────────────────────────────────
    last_v_dur = virtuals[-1]["src_end_sec"] - virtuals[-1]["src_start_sec"]
    last_a_end = a_dst[-1] + last_v_dur

    # 尾部 user 音频
    tail_gs = virtuals[-1]["src_end_samp"]
    tail_result = _first_last_nonzero(u[tail_gs:]) if tail_gs < total else None
    if tail_result is not None:
        u_tail_src_s = (tail_gs + tail_result[0]) / sr
        u_tail_src_e = (tail_gs + tail_result[1] + 1) / sr
        u_tail_dur = u_tail_src_e - u_tail_src_s
        if virtuals[-1]["is_cluster"]:
            ov_tail_eff = 0.0
        else:
            ov_tail = _pick_ov_from_text(virtuals[-1]["boundary_text"].strip())
            # ★ 末段 assistant 或尾部 user 短于目标重叠时，缩小
            ov_tail_eff = min(ov_tail, last_v_dur, u_tail_dur)
        effective_overlap_total += ov_tail_eff
        if ov_tail_eff > 0:
            nonzero_overlap_count += 1
        # ★ 不与最后一段 gap user 重叠
        u_tail_dst_s = max(last_a_end - ov_tail_eff, u_channel_end)
        tail_end = u_tail_dst_s + u_tail_dur
    else:
        u_tail_src_s = u_tail_src_e = u_tail_dst_s = tail_end = 0.0

    out_len = _sec2samp(max(last_a_end, tail_end)) + sr
    stereo = np.zeros((out_len, 2), dtype=np.float32)


    def place(
        src_start_sec: float,
        src_end_sec: float,
        dst_start_sec: float,
        *,
        record_assistant: bool = False,
        record_user: bool = False,
    ) -> None:
        ss = max(0, _sec2samp(src_start_sec))
        se = min(total, _sec2samp(src_end_sec))
        ds = _sec2samp(dst_start_sec)
        if ss >= se or ds < 0:
            return
        actual = min(se - ss, out_len - ds)
        if actual <= 0:
            return
        if record_assistant:
            stereo[ds:ds+actual, 0] += a[ss:ss+actual]
        if record_user:
            stereo[ds:ds+actual, 1] += u[ss:ss+actual]
        mapping_item = _mapping_item(ss, ds, actual, sr)
        if record_assistant:
            fake_raw_assistant.append(mapping_item)
        if record_user:
            fake_raw_user.append(mapping_item)


    # head gap
    if u_head_dur > 0:
        place(u_head_src_s, u_head_src_e, 0.0, record_user=True)

    # virtuals: cluster → 双声道一起搬；isolated → 仅 assistant
    for vi, v in enumerate(virtuals):
        if v["is_cluster"]:
            place(
                v["src_start_sec"], v["src_end_sec"], a_dst[vi],
                record_assistant=True, record_user=True,
            )
        else:
            place(
                v["src_start_sec"], v["src_end_sec"], a_dst[vi],
                record_assistant=True,
            )

    # inter-virtual user gaps
    for vi in range(nv - 1):
        place(gaps_src[vi][0], gaps_src[vi][1], u_dst[vi], record_user=True)

    if u_tail_src_e > u_tail_src_s:
        place(u_tail_src_s, u_tail_src_e, u_tail_dst_s, record_user=True)

    np.clip(stereo, -1.0, 1.0, out=stereo)

    # 裁尾部静音
    nz = np.flatnonzero(stereo.any(axis=1))
    trim = nz[-1] + 1 if nz.size else 0
    fake_raw_assistant = _clip_mappings_to_trim(fake_raw_assistant, trim, sr)
    fake_raw_user = _clip_mappings_to_trim(fake_raw_user, trim, sr)
    stereo = stereo[:trim]
    overlap_stats = {
        "effective_overlap_total": float(effective_overlap_total),
        "nonzero_overlap_count": int(nonzero_overlap_count),
    }
    return stereo, updated, fake_raw_assistant, fake_raw_user, overlap_stats

def process_file(file_path):
    cnt_compressed = 0
    if (not file_path.endswith(".flac") and not file_path.endswith(".wav")) or file_path.endswith("_compressed.flac"):
        return 0
    json_path, new_stereo_path, new_stereo_json_path = _compressed_output_paths(file_path)
    _cleanup_incomplete_compressed_outputs(
        file_path,
        new_stereo_path,
        new_stereo_json_path,
    )
    if os.path.exists(new_stereo_path) and os.path.exists(new_stereo_json_path) and not OVERWRITE_EXISTING:
        if VERBOSE:
            print(f"Compressed files already exist for {file_path}, skipping.")
        return 0
    with open(json_path, 'rb') as f:
        metadata = orjson.loads(f.read())
    if "metadata:transcript" not in metadata:
        print(f"Skipping {file_path} due to missing transcript")
        return 0
    raw_audio, sr = sf.read(file_path, dtype="float32", always_2d=True)
    transcript_error = _validate_transcript_for_compression(metadata["metadata:transcript"], sample_rate=sr)
    if transcript_error is not None:
        print(f"Skipping {file_path} due to invalid transcript: {transcript_error}")
        return 0
    assert raw_audio.shape[1] == 2, f"{file_path} is not stereo"
    assistant_audio = raw_audio[:, 0]
    user_audio = raw_audio[:, 1]

    clusters: list[dict] = []
    if check_have_overlap(assistant_audio, user_audio):
        if not ENABLE_CLUSTER_MODE:
            if VERBOSE:
                print(f"Overlap found in {file_path}, skipping (ENABLE_CLUSTER_MODE=False)")
            return 0
        clusters = detect_overlap_clusters(
            assistant_audio,
            user_audio,
            metadata["metadata:transcript"],
            sample_rate=sr,
        )
        # 安全过滤：任一簇不包含 transcript segment（重叠发生在未转写区域）
        # 都直接跳过，避免丢失 assistant 通道音频
        orphan_clusters = [c for c in clusters if not c["transcript_indices"]]
        if orphan_clusters:
            print(
                f"Skipping {file_path}: {len(orphan_clusters)} overlap cluster(s) outside any "
                f"transcript segment; safe recovery not guaranteed"
            )
            return 0
        if clusters:
            cluster_samp_total = sum(
                c["src_end_samp"] - c["src_start_samp"] for c in clusters
            )
            coverage = cluster_samp_total / max(1, len(raw_audio))
            if coverage > CLUSTER_COVERAGE_THRESHOLD:
                if VERBOSE:
                    print(
                        f"Skipping {file_path}: cluster coverage {coverage:.2%} "
                        f"> threshold {CLUSTER_COVERAGE_THRESHOLD:.2%}"
                    )
                return 0
            if VERBOSE:
                print(
                    f"Overlap found in {file_path}: {len(clusters)} cluster(s) covering "
                    f"{coverage:.2%} of audio — compressing with cluster mode"
                )
        else:
            # check_have_overlap 比 detect_overlap_clusters 更严格（>0 vs 相对门限）
            # 没有簇 → 重叠是极微弱噪声，按普通路径压缩
            if VERBOSE:
                print(f"Sub-threshold overlap in {file_path} — compressing without clusters")
    else:
        if VERBOSE:
            print(f"No overlap in {file_path}")

    (
        new_stereo,
        metadata["metadata:transcript"],
        fake_raw_assistant,
        fake_raw_user,
        overlap_stats,
    ) = compress_gaps(
        assistant_audio,
        user_audio,
        metadata["metadata:transcript"],
        target_overlap_sec=(0.2, 0.6),
        target_overlap_sec_large=(1, 2),
        sample_rate=sr,
        clusters=clusters,
    )
    if overlap_stats["nonzero_overlap_count"] == 0:
        if VERBOSE:
            print(
                f"Skipping {file_path}: effective overlap total is 0.000s "
                f"(cluster mode produced no compressible boundaries)"
            )
        return 0
    metadata["fake_raw_assistant"] = fake_raw_assistant
    metadata["fake_raw_user"] = fake_raw_user
    metadata["raw_num_samples"] = int(len(raw_audio))
    add_alignments_to_metadata(metadata)
    metadata["duration"] = float(len(new_stereo) / sr)
    #  check_recoverable_and_remove_file
    recovered_raw_audio = recover_raw_audio(new_stereo, fake_raw_assistant, fake_raw_user, len(raw_audio))
    if SAVE_RECOVERED_DEBUG_AUDIO:
        sf.write(new_stereo_path.replace("_compressed.flac", "_recovered_raw.flac"), recovered_raw_audio, sr)
    recovery_stats = evaluate_recovery(
        raw_audio,
        recovered_raw_audio,
        fake_raw_assistant,
        fake_raw_user,
        atol=1e-4,
    )

    if not recovery_stats["all_pass"] or not REMOVE_ORIGINAL_IF_RECOVERY_PASS:
        # 区分两类失败，便于定位
        if not recovery_stats["recoverable_pass"]:
            print(f"Warning: mapped-region recovery FAILED for {file_path} "
                  f"(mapping/placement bug)")
        if not recovery_stats["unmapped_significant_pass"]:
            print(f"Warning: audible content OUTSIDE mapped regions for {file_path} "
                  f"(simple model + cluster detection didn't cover all active samples)")
        # os.remove(new_stereo_path)
        # os.remove(new_stereo_json_path)
        print(
            f"Recovery stats for {file_path}: "
            f"recoverable_max_abs_diff={recovery_stats['recoverable_max_abs_diff']:.10f}, "
            f"recoverable_num_over_atol={recovery_stats['recoverable_num_over_atol']}, "
            f"boundary_max_abs_diff={recovery_stats['boundary_max_abs_diff']:.10f}, "
            f"boundary_num_over_atol={recovery_stats['boundary_num_over_atol']}, "
            f"unmapped_max_abs_diff={recovery_stats['unrecoverable_max_abs_diff']:.10f}, "
            f"unmapped_num_over_atol={recovery_stats['unrecoverable_num_over_atol']}"
        )
        # exit(1)
    metadata_bytes = orjson.dumps(metadata)
    _atomic_write_audio(new_stereo_path, new_stereo, sr)
    _atomic_write_bytes(new_stereo_json_path, metadata_bytes)
    if recovery_stats["all_pass"] and REMOVE_ORIGINAL_IF_RECOVERY_PASS:
        # Only remove the source after both compressed outputs are durable.
        os.remove(file_path)
    if VERBOSE:
        print(f"Compressed audio saved to {new_stereo_path}")
        print(f"Compressed metadata saved to {new_stereo_json_path}")
        print(f"Original duration: {len(assistant_audio) / sr:.2f} seconds, New duration: {len(new_stereo) / sr:.2f} seconds")
    cnt_compressed = 1
    return cnt_compressed


def _safe_process_file(file_path):
    try:
        return file_path, process_file(file_path), None
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return file_path, None, repr(exc)


def _collect_batch_from_jsonl(jsonl_file, base_dir, batch_size, max_entries=None):
    batch = []
    total_entries = 0
    skipped_prefilter = 0
    while len(batch) < batch_size:
        line = jsonl_file.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        total_entries += 1
        if max_entries is not None and total_entries > max_entries:
            break
        data = orjson.loads(line)
        file_path = os.path.join(base_dir, data["path"]) # .replace(".wav", ".flac")
        if _should_skip_file(file_path):
            skipped_prefilter += 1
            continue
        batch.append(file_path)

    return batch, total_entries, skipped_prefilter

def process_dir(data_dir="./tts_output/datasets"):
    count_compressed = 0
    count_skipped = 0
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            for filename in tqdm(os.listdir(subdir_path), desc=f"Processing {subdir}"):
                file_path = os.path.join(subdir_path, filename)
                cnt_compressed = process_file(file_path)
                if cnt_compressed:
                    count_compressed += 1
                else:
                    count_skipped += 1
            total_checked = count_compressed + count_skipped
            print(f"Total files compressed: {count_compressed} ({(count_compressed / total_checked) * 100:.2f}%)")
            print(f"Total files skipped: {count_skipped} ({(count_skipped / total_checked) * 100:.2f}%)")
            print(f"Total files checked: {total_checked}")
                    
def process_jsonl(jsonl_path, num_workers=None, batch_size=None, chunksize=None, max_entries=None):
    if num_workers is None:
        num_workers = min(32, os.cpu_count() or 1)
    if batch_size is None:
        batch_size = max(num_workers * 8, 256)
    if chunksize is None:
        chunksize = max(4, batch_size // max(num_workers * 8, 1))

    count_compressed = 0
    count_skipped = 0
    failures = []
    total_processed = 0
    batch_count = 0
    skipped_prefilter = 0
    total_entries = 0

    progress = tqdm(desc=f"Processing {os.path.basename(jsonl_path)}")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        base_dir = os.path.dirname(jsonl_path)
        with open(jsonl_path, 'r') as jsonl_file:
            while True:
                batch, batch_total_entries, batch_skipped_prefilter = _collect_batch_from_jsonl(
                    jsonl_file,
                    base_dir,
                    batch_size,
                    max_entries
                )
                total_entries += batch_total_entries
                skipped_prefilter += batch_skipped_prefilter

                if not batch:
                    if batch_total_entries == 0:
                        break
                    continue

                batch_count += 1
                for file_path, cnt_compressed, error in executor.map(
                    _safe_process_file,
                    batch,
                    chunksize=chunksize,
                ):
                    if error is not None:
                        failures.append((file_path, error))
                    elif cnt_compressed:
                        count_compressed += 1
                        if count_compressed % 1000 == 0:
                            print(f"Compressed: {file_path}")
                    else:
                        count_skipped += 1
                    total_processed += 1
                    progress.update(1)
                if max_entries is not None and total_processed >= max_entries:
                    break
    progress.close()

    if total_entries == 0:
        print(f"No valid entries found in {jsonl_path}")
        return

    print(
        f"Submitted in {batch_count} batches "
        f"(batch_size={batch_size}, chunksize={chunksize}, workers={num_workers})"
    )
    if total_processed:
        print(f"Total files compressed: {count_compressed} ({(count_compressed / total_processed) * 100:.2f}%)")
        print(f"Total files skipped: {count_skipped} ({(count_skipped / total_processed) * 100:.2f}%)")
    if skipped_prefilter:
        print(f"Prefilter skipped before dispatch: {skipped_prefilter}")
    print(f"Total files checked: {total_processed}")
    print(f"Total jsonl entries seen: {total_entries}")

    if failures:
        print(f"Failed files: {len(failures)}")
        for file_path, error in failures[:10]:
            print(f"  {file_path}: {error}")

if __name__ == "__main__":
    process_jsonl(
        jsonl_path=os.environ.get(
            "DURATIONS_JSONL",
            "./tts_output/durations_stereo.jsonl",
        ),
        num_workers=int(os.environ.get("NUM_WORKERS", 32)),
        batch_size=int(os.environ.get("BATCH_SIZE", 1024)),
        chunksize=int(os.environ.get("CHUNKSIZE", 4)),
        max_entries=None,
    )
