"""
mfa_aligner.py

使用 Montreal Forced Aligner (MFA) 3.3.9 进行字/词级别强制对齐。
支持中文和英文，自动检测语言，返回与 Whisper 版本兼容的输出格式。

依赖:
    conda install -c conda-forge montreal-forced-aligner=3.3.9
    pip install textgrid soundfile numpy

    # 预下载模型:
    mfa model download acoustic mandarin_mfa
    mfa model download dictionary mandarin_mfa
    mfa model download acoustic english_mfa
    mfa model download dictionary english_mfa

Usage:
    from mfa_aligner import MFAAligner

    aligner = MFAAligner(mfa_root="~/Documents/MFA")

    # 从文件对齐（只对齐 1.0s ~ 3.5s 之间的片段）
    result = aligner.align("audio.wav", "呸,我大哥是汉室宗亲",
                           language="zh", start_time=1.0, end_time=3.5)

    # 从 numpy 数组对齐
    result = aligner.align_audio(audio_np, sr, "Hello world", language="en")

    print(result["language"])
    for word, (start, end) in result["words"]:
        print(f"{word}  {start:.3f} - {end:.3f}")
"""

import os
import re
import shutil
import subprocess
import tempfile
import traceback  # Bug 1/3 fix: 移至模块级，避免方法内局部 import 遮蔽异常信息
import uuid
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import soundfile as sf

try:
    import textgrid
except ImportError:
    textgrid = None
    print("[WARN] 'textgrid' package not installed. Run: pip install textgrid")


# ============================================================
# Constants
# ============================================================

CJK_PUNCTUATION = '，。！？、；：\u201c\u201d\u2018\u2019（）【】《》—…·「」『』〈〉'
EN_PUNCTUATION = ',.!?;:\'"()[]{}/-\u2013\u2014\u2026'
ALL_PUNCTUATION = set(CJK_PUNCTUATION + EN_PUNCTUATION)

CJK_LANGUAGES = {'zh', 'ja', 'ko', 'yue'}

# MFA 模型名称映射
MFA_MODELS = {
    'zh': {
        'acoustic':   'mandarin_mfa',
        'dictionary': 'mandarin_mfa',
    },
    'en': {
        'acoustic':   'english_mfa',
        'dictionary': 'english_mfa',
    },
    'ja': {
        'acoustic':   'japanese_mfa',
        'dictionary': 'japanese_mfa',
    },
    'ko': {
        'acoustic':   'korean_mfa',
        'dictionary': 'korean_mfa',
    },
}

# MFA 预期采样率
MFA_SAMPLE_RATE = 16000

def sf_read(audio_path, **kwargs):
    if not os.path.exists(audio_path):
        ext_map = {'.wav': '.flac', '.flac': '.wav'}
        root, ext = os.path.splitext(audio_path)
        fallback = root + ext_map.get(ext.lower(), '')
        if fallback and os.path.exists(fallback):
            audio_path = fallback
    return sf.read(audio_path, **kwargs)


# ============================================================
# Language Detection
# ============================================================

def detect_language(text: str) -> str:
    """简易语言检测：统计 CJK 字符占比。"""
    if not text.strip():
        return 'en'

    cjk_count = 0
    alpha_count = 0
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF):
            cjk_count += 1
        elif ch.isalpha():
            alpha_count += 1

    total = cjk_count + alpha_count
    if total == 0:
        ret = 'en'
    else:
        ret = 'zh' if cjk_count / total > 0.3 else 'en'
    print(f"[LANG DETECT] CJK: {cjk_count}, Alpha: {alpha_count}, Detected: {ret}")
    return ret


# ============================================================
# TextGrid Parser
# ============================================================

def parse_textgrid(tg_path: str, language: str) -> Dict[str, Any]:
    """解析 MFA 输出的 TextGrid 文件。

    MFA 输出包含多层:
        - "words"   → 词级别边界
        - "phones"  → 音素级别边界

    返回:
        {
            "words":  [{"word": str, "start": float, "end": float}, ...],
            "phones": [{"phone": str, "start": float, "end": float}, ...],
        }
    """
    if textgrid is None:
        return _parse_textgrid_manual(tg_path)

    tg = textgrid.TextGrid.fromFile(tg_path)

    words = []
    phones = []

    for tier in tg:
        tier_name = tier.name.lower()
        for interval in tier:
            text = interval.mark.strip()
            if not text:
                continue

            entry = {
                "start": float(interval.minTime),
                "end":   float(interval.maxTime),
            }

            if 'word' in tier_name:
                entry["word"] = text
                words.append(entry)
            elif 'phone' in tier_name:
                entry["phone"] = text
                phones.append(entry)

    return {"words": words, "phones": phones}


def _parse_textgrid_manual(tg_path: str) -> Dict[str, Any]:
    """手动解析 TextGrid (不依赖 textgrid 库的 fallback)."""
    with open(tg_path, 'r', encoding='utf-8') as f:
        content = f.read()

    words = []
    phones = []

    # 按 item 分割
    items = re.split(r'item\s*\[\d+\]', content)

    for item in items:
        is_word  = bool(re.search(r'name\s*=\s*".*word.*"',  item, re.IGNORECASE))
        is_phone = bool(re.search(r'name\s*=\s*".*phone.*"', item, re.IGNORECASE))

        intervals = re.findall(
            r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"',
            item
        )

        for xmin, xmax, text in intervals:
            text = text.strip()
            if not text:
                continue

            if is_word:
                words.append({
                    "word":  text,
                    "start": float(xmin),
                    "end":   float(xmax),
                })
            elif is_phone:
                phones.append({
                    "phone": text,
                    "start": float(xmin),
                    "end":   float(xmax),
                })

    return {"words": words, "phones": phones}


# ============================================================
# Audio Preprocessing
# ============================================================

def prepare_audio(audio_path: str, output_path: str) -> str:
    """将音频转换为 MFA 要求的格式: 16kHz, mono, 16-bit WAV."""
    audio, sr = sf_read(audio_path, dtype='float32')

    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != MFA_SAMPLE_RATE:
        audio = _resample(audio, sr, MFA_SAMPLE_RATE)

    sf.write(output_path, audio, MFA_SAMPLE_RATE, subtype='PCM_16')
    return output_path


def prepare_audio_from_numpy(
    audio_np: np.ndarray, sr: int, output_path: str
) -> str:
    """从 numpy 数组保存为 MFA 兼容的 WAV."""
    audio = audio_np.astype(np.float32)

    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != MFA_SAMPLE_RATE:
        audio = _resample(audio, sr, MFA_SAMPLE_RATE)

    sf.write(output_path, audio, MFA_SAMPLE_RATE, subtype='PCM_16')
    return output_path


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """简易线性插值重采样。如有 librosa 可替换为更高质量的版本。"""
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        pass

    # Fallback: 线性插值
    duration   = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    indices    = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# ============================================================
# Chinese Text → MFA Lab Format
# ============================================================

def text_to_mfa_lab(text: str, language: str) -> str:
    """将文本转换为 MFA 的 .lab/.txt 格式。

    中文: MFA mandarin_mfa 字典接受汉字输入，每个字用空格分隔。
    英文: 原文直接写入（MFA 按空格分词）。
    """
    if language in CJK_LANGUAGES:
        chars = []
        for ch in text:
            if ch in ALL_PUNCTUATION or ch.isspace():
                continue
            chars.append(ch)
        return ' '.join(chars)
    else:
        cleaned = re.sub(r"[^\w\s']", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


# ============================================================
# MFA Aligner Class
# ============================================================

class MFAAligner:
    """封装 MFA 3.3.9 的强制对齐功能。

    Attributes:
        mfa_bin:           mfa 可执行文件路径 (默认 "mfa")
        mfa_root:          包含 pretrained_models/ 子目录的 MFA 根目录。
                           默认值 ~/Documents/MFA 与 MFA 3.x 的默认安装路径一致。
                           可通过环境变量 MFA_ROOT_DIR 或此参数覆盖。
        extra_args:        额外传给 mfa align 的参数
        keep_temp:         是否保留临时文件 (调试用)
        beam / retry_beam: MFA 解码参数
    """

    def __init__(
        self,
        mfa_bin:    str            = "mfa",
        mfa_root:   Optional[str]  = None,   # ← Bug 2 fix: 替换硬编码路径
        beam:       int            = 400,
        retry_beam: int            = 2000,
        extra_args: Optional[List[str]] = None,
        keep_temp:  bool           = False,
        use_fallback: bool         = False,
    ):
        self.mfa_bin   = mfa_bin
        self.beam      = beam
        self.retry_beam = retry_beam
        self.extra_args = extra_args or []
        self.keep_temp  = keep_temp
        self.use_fallback = use_fallback

        # Bug 2 fix: 解析 mfa_root，优先级：参数 > 环境变量 > 默认值
        _root = (
            mfa_root
            or os.environ.get("MFA_ROOT_DIR")
            or str(Path.home() / "Documents" / "MFA")
        )
        self.mfa_root: Path = Path(_root).expanduser().resolve()

        pretrained_src = self.mfa_root / "pretrained_models"
        if not pretrained_src.exists():
            print(
                f"[MFA][WARN] pretrained_models 目录不存在: '{pretrained_src}'。\n"
                "  请确认 mfa_root 指向包含 pretrained_models/ 的目录，\n"
                "  或运行: mfa model download acoustic <model_name>"
            )

    def _check_mfa(self):
        """验证 MFA 是否已安装且可调用。"""
        try:
            result = subprocess.run(
                [self.mfa_bin, "version"],
                capture_output=True, text=True, timeout=30
            )
            version = result.stdout.strip()
            print(f"[MFA] Detected version: {version}")
        except FileNotFoundError:
            raise RuntimeError(
                f"MFA not found at '{self.mfa_bin}'. "
                "Please install: conda install -c conda-forge montreal-forced-aligner"
            )

    # --------------------------------------------------------
    # 内部工具：创建隔离的 MFA_ROOT_DIR 并软链接 pretrained_models
    # --------------------------------------------------------

    def _make_per_process_root(self) -> Path:
        """
        为当前调用创建一个独立的临时 MFA_ROOT_DIR，
        并将真实的 pretrained_models 目录软链接进去。

        这样多个进程可以并发调用 MFA，而不会因共享全局
        MFA_ROOT_DIR 下的写入状态相互干扰。
        """
        per_process_root = Path(tempfile.mkdtemp(prefix="mfa_root_"))
        src = self.mfa_root / "pretrained_models"
        dst = per_process_root / "pretrained_models"
        os.symlink(src, dst)
        return per_process_root

    @staticmethod
    def _cleanup_per_process_root(per_process_root: Path) -> None:
        """
        Bug 5 fix: 先显式 unlink 软链接，再 rmtree，
        避免在某些平台上 rmtree 跟随软链接删除源目录内容。
        """
        symlink_path = per_process_root / "pretrained_models"
        if symlink_path.is_symlink():
            symlink_path.unlink()
        shutil.rmtree(per_process_root, ignore_errors=True)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def align(
        self,
        audio_path: str,
        text:       str,
        language:   Optional[str] = None,
        start_time: float = 0.0,
        end_time:   Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """从音频文件路径对齐。

        Args:
            audio_path: 音频文件路径
            text:       已知文本
            language:   'zh' / 'en' / None(自动检测)
            start_time: 片段在全局时间线中的起始位置(秒)
            end_time:   片段在全局时间线中的结束位置(秒)；
                        为 None 时对齐到文件末尾

        Returns:
            {
                "words":      [[word_or_char, [abs_start, abs_end]], ...],
                "phones":     [{"phone": str, "start": float, "end": float}, ...],
                "start":      float,
                "end":        float,
                "transcript": str,
                "language":   str
            }
        """
        if not text or not text.strip():
            return None

        lang = language or detect_language(text)
        return self._run_alignment(
            audio_source=("file", audio_path),
            text=text,
            language=lang,
            start_time=start_time,
            end_time=end_time,
        )

    def align_audio(
        self,
        audio_data: np.ndarray,
        sr:         int,
        text:       str,
        language:   Optional[str] = None,
        start_time: float = 0.0,
        end_time:   Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """从 numpy 数组对齐。"""
        if not text or not text.strip():
            return None

        lang = language or detect_language(text)
        return self._run_alignment(
            audio_source=("numpy", audio_data, sr),
            text=text,
            language=lang,
            start_time=start_time,
            end_time=end_time,
        )

    def align_batch(
        self,
        segment_requests: List[Dict[str, Any]],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        将任意多个片段（可来自不同音频文件）合并为一次 MFA 调用。

        Args:
            segment_requests: 片段列表，每个元素包含：
                audio_path (str):      音频文件路径
                text       (str):      文本内容
                start    (float):      在原始文件中的起始时间（秒）
                end      (float|None): 结束时间（秒），None 表示到文件末尾
                language   (str|None): 语言代码，None 时自动检测

        Returns:
            与输入等长的结果列表；失败位置为 None

        注意：
            同一 batch 内所有片段须为同一语言。
            调用方应在打 batch 前按语言分组（align_duplex.py 已实现此逻辑）。
        """
        if not segment_requests:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * len(segment_requests)

        # 语言检测：取首个有效片段（batch 内语言应已统一）
        language: Optional[str] = None
        for req in segment_requests:
            lang = req.get('language')
            if lang:
                language = lang
                break
            text = req.get('text', '').strip()
            if text:
                language = detect_language(text)
                break
        if language is None:
            language = 'en'
        if language not in MFA_MODELS:
            print(f"[WARN] No MFA model for '{language}', falling back to 'en'")
            language = 'en'

        model_cfg  = MFA_MODELS[language]
        acoustic   = model_cfg['acoustic']
        dictionary = model_cfg['dictionary']

        work_dir   = Path(tempfile.mkdtemp(prefix="mfa_work_"))
        corpus_dir = work_dir / "corpus"
        output_dir = work_dir / "output"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Bug 2 fix: 使用 self.mfa_root 而非硬编码路径
        per_process_root = self._make_per_process_root()

        # batch 内音频缓存：同一文件只读取和重采样一次
        audio_cache: Dict[str, np.ndarray] = {}
        utt_ids: List[Optional[str]] = [None] * len(segment_requests)

        try:
            # --------------------------------------------------
            # 1. 读取音频（带缓存）、裁剪、写 wav + txt
            # --------------------------------------------------
            for idx, req in enumerate(segment_requests):
                text       = req.get('text', '').strip()
                audio_path = req.get('audio_path', '')
                start_time = float(req.get('start', 0.0))
                end_time   = req.get('end')

                if not text or not audio_path:
                    continue

                if audio_path not in audio_cache:
                    audio_full, orig_sr = sf_read(audio_path, dtype='float32')
                    if audio_full.ndim > 1:
                        audio_full = audio_full[:, 0]
                    if orig_sr != MFA_SAMPLE_RATE:
                        audio_full = _resample(audio_full, orig_sr, MFA_SAMPLE_RATE)
                    audio_cache[audio_path] = audio_full

                audio_full    = audio_cache[audio_path]
                total_samples = len(audio_full)

                start_sample = max(0, int(round(start_time * MFA_SAMPLE_RATE)))
                end_sample   = (
                    int(round(float(end_time) * MFA_SAMPLE_RATE))
                    if end_time is not None
                    else total_samples
                )
                end_sample = min(end_sample, total_samples)
                clip       = audio_full[start_sample:end_sample]
                if len(clip) == 0:
                    continue

                utt_id       = f"seg{idx:06d}_{uuid.uuid4().hex[:6]}"
                utt_ids[idx] = utt_id

                sf.write(
                    str(corpus_dir / f"{utt_id}.wav"),
                    clip, MFA_SAMPLE_RATE, subtype='PCM_16',
                )
                with open(
                    corpus_dir / f"{utt_id}.txt", 'w', encoding='utf-8'
                ) as f:
                    f.write(text_to_mfa_lab(text, language))

            valid_count = sum(1 for uid in utt_ids if uid is not None)
            if valid_count == 0:
                return results

            # --------------------------------------------------
            # 2. 一次 MFA 调用处理全部片段
            # --------------------------------------------------
            cmd = [
                self.mfa_bin, "align",
                str(corpus_dir), dictionary, acoustic, str(output_dir),
                "--clean", "--single_speaker",
                "--temporary_directory", str(per_process_root),
                "--beam",          str(self.beam),
                "--retry_beam",    str(self.retry_beam),
                "--output_format", "long_textgrid",
            ] + self.extra_args

            env                 = os.environ.copy()
            env["MFA_ROOT_DIR"] = str(per_process_root)

            # timeout 随 batch 大小线性放大，每个片段预留 30 秒
            timeout = max(600, valid_count * 30)

            print(f"[MFA] Batch aligning {valid_count} segments ...")
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=timeout
            )

            if proc.returncode != 0:
                print(f"[MFA] Batch failed:\n{proc.stderr[-2000:]}")
                return results

            # --------------------------------------------------
            # 3. 逐一解析 TextGrid，写回 results
            # --------------------------------------------------
            for idx, req in enumerate(segment_requests):
                utt_id = utt_ids[idx]
                if utt_id is None:
                    continue

                tg_path = self._find_textgrid_by_id(str(output_dir), utt_id)
                if not tg_path:
                    continue

                text       = req.get('text', '')
                start_time = float(req.get('start', 0.0))
                end_time   = req.get('end')
                audio_path = req.get('audio_path', '')

                audio_full    = audio_cache.get(audio_path)
                total_samples = len(audio_full) if audio_full is not None else 0
                start_sample  = max(0, int(round(start_time * MFA_SAMPLE_RATE)))
                end_sample    = (
                    int(round(float(end_time) * MFA_SAMPLE_RATE))
                    if end_time is not None
                    else total_samples
                )
                end_sample = min(end_sample, total_samples)
                duration   = (end_sample - start_sample) / MFA_SAMPLE_RATE

                parsed     = parse_textgrid(tg_path, language)
                words_list = self._build_output(parsed, text, language, start_time)
                phones_out = [
                    {
                        "phone": p["phone"],
                        "start": round(p["start"] + start_time, 6),
                        "end":   round(p["end"]   + start_time, 6),
                    }
                    for p in parsed.get("phones", [])
                ]

                results[idx] = {
                    "words":      words_list,
                    "phones":     phones_out,
                    "start":      start_time,
                    "end":        round(start_time + duration, 6),
                    "transcript": text,
                    "language":   language,
                }

            return results

        except subprocess.TimeoutExpired:
            print(f"[MFA] Batch ({valid_count} segs) timed out.")
            return results
        except Exception as e:
            print(f"[MFA] Batch alignment error: {e}")
            traceback.print_exc()
            return results
        finally:
            if not self.keep_temp:
                shutil.rmtree(work_dir,  ignore_errors=True)
                # Bug 5 fix: 先 unlink 软链接，再 rmtree
                self._cleanup_per_process_root(per_process_root)

    def _find_textgrid_by_id(self, output_dir: str, utt_id: str) -> Optional[str]:
        """按 utt_id 精确查找对应的 TextGrid 文件。"""
        for root, _, files in os.walk(output_dir):
            for fname in files:
                if fname == f"{utt_id}.TextGrid":
                    return os.path.join(root, fname)
        return None

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------

    def _run_alignment(
        self,
        audio_source: tuple,
        text:         str,
        language:     str,
        start_time:   float,
        end_time:     Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """执行完整的 MFA 对齐流程（单片段）。"""

        if language not in MFA_MODELS:
            print(f"[WARN] No MFA model configured for '{language}', falling back to 'en'")
            language = 'en'

        model_cfg       = MFA_MODELS[language]
        acoustic_model  = model_cfg['acoustic']
        dictionary      = model_cfg['dictionary']

        # Bug 1 fix: 在 try 块之前给 duration 一个安全默认值。
        # 若 sf_read / 重采样在 try 内部抛出异常，except 分支中的
        # self._fallback(text, duration, ...) 不会因未绑定名字而触发 NameError。
        duration: float = (
            (end_time - start_time) if end_time is not None else 0.0
        )

        work_dir   = Path(tempfile.mkdtemp(prefix="mfa_work_"))
        corpus_dir = work_dir / "corpus"
        output_dir = work_dir / "output"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Bug 2 fix: 使用 self.mfa_root 而非硬编码路径
        per_process_root = self._make_per_process_root()

        utt_id = f"utt_{uuid.uuid4().hex[:8]}"

        try:
            # --------------------------------------------------
            # 1. 准备音频
            # --------------------------------------------------
            wav_path = str(corpus_dir / f"{utt_id}.wav")

            if audio_source[0] == "file":
                audio, orig_sr = sf_read(audio_source[1], dtype='float32')

                if audio.ndim > 1:
                    audio = audio[:, 0]
                if orig_sr != MFA_SAMPLE_RATE:
                    audio = _resample(audio, orig_sr, MFA_SAMPLE_RATE)

                start_sample = max(0, int(round(start_time * MFA_SAMPLE_RATE)))
                end_sample   = (
                    int(round(end_time * MFA_SAMPLE_RATE))
                    if end_time is not None
                    else len(audio)
                )
                end_sample = min(end_sample, len(audio))
                audio      = audio[start_sample:end_sample]

                if len(audio) == 0:
                    raise ValueError(
                        f"裁剪后音频为空: start_time={start_time}, "
                        f"end_time={end_time}"
                    )

                sf.write(wav_path, audio, MFA_SAMPLE_RATE, subtype='PCM_16')
                # Bug 1 fix: 成功读取后精化 duration
                duration = len(audio) / MFA_SAMPLE_RATE

            else:  # numpy
                audio_np, sr = audio_source[1], audio_source[2]
                prepare_audio_from_numpy(audio_np, sr, wav_path)
                # Bug 1 fix: 成功处理后精化 duration
                duration = (
                    end_time - start_time
                    if end_time is not None
                    else len(audio_np) / sr
                )

            # --------------------------------------------------
            # 2. 准备文本
            # --------------------------------------------------
            lab_text = text_to_mfa_lab(text, language)
            with open(
                corpus_dir / f"{utt_id}.txt", 'w', encoding='utf-8'
            ) as f:
                f.write(lab_text)

            # --------------------------------------------------
            # 3. 运行 MFA
            # --------------------------------------------------
            cmd = [
                self.mfa_bin, "align",
                str(corpus_dir), dictionary, acoustic_model, str(output_dir),
                "--clean",
                "--single_speaker",
                "--temporary_directory", str(per_process_root),
                "--beam",          str(self.beam),
                "--retry_beam",    str(self.retry_beam),
                "--output_format", "long_textgrid",
            ] + self.extra_args

            print(f"[MFA] Running: {' '.join(cmd)}")

            env                 = os.environ.copy()
            env["MFA_ROOT_DIR"] = str(per_process_root)

            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=600,
            )

            if proc.returncode != 0:
                print(f"[MFA] STDERR:\n{proc.stderr}")
                print(f"[MFA] STDOUT:\n{proc.stdout}")
                # Bug 3 fix: 在调用点检查 use_fallback，而非依赖 _fallback 内部 raise
                if self.use_fallback:
                    print("[MFA] Alignment failed, using fallback.")
                    return self._fallback(text, duration, start_time, language)
                print("[MFA] Alignment failed.")
                return None

            # --------------------------------------------------
            # 4. 查找并解析 TextGrid
            # --------------------------------------------------
            tg_path = self._find_textgrid(str(output_dir), utt_id)
            if not tg_path:
                # Bug 3 fix: 同上
                if self.use_fallback:
                    print("[MFA] TextGrid not found, using fallback.")
                    return self._fallback(text, duration, start_time, language)
                print("[MFA] TextGrid not found.")
                return None

            parsed = parse_textgrid(tg_path, language)

            # --------------------------------------------------
            # 5. 构造输出（MFA 时间戳 + start_time 偏移）
            # --------------------------------------------------
            words_list = self._build_output(parsed, text, language, start_time)

            phones_out = [
                {
                    "phone": p["phone"],
                    "start": round(p["start"] + start_time, 6),
                    "end":   round(p["end"]   + start_time, 6),
                }
                for p in parsed.get("phones", [])
            ]

            return {
                "words":      words_list,
                "phones":     phones_out,
                "start":      start_time,
                "end":        round(start_time + duration, 6),
                "transcript": text,
                "language":   language,
            }

        except subprocess.TimeoutExpired:
            print("[MFA] Alignment timed out.")
            # Bug 3 fix: 统一在调用点检查 use_fallback
            if self.use_fallback:
                return self._fallback(text, duration, start_time, language)
            return None

        except Exception as e:
            print(f"[MFA] Error: {e}")
            traceback.print_exc()
            # Bug 3 fix: 同上
            if self.use_fallback:
                return self._fallback(text, duration, start_time, language)
            return None

        finally:
            if not self.keep_temp:
                shutil.rmtree(work_dir, ignore_errors=True)
                # Bug 5 fix: 先 unlink 软链接，再 rmtree
                self._cleanup_per_process_root(per_process_root)
            else:
                print(f"[MFA] Temp dir preserved: {work_dir}")

    def _find_textgrid(self, output_dir: str, utt_id: str) -> Optional[str]:
        """在 MFA 输出目录中按 utt_id 精确查找 TextGrid 文件。

        Minor fix: 原实现忽略了 utt_id 参数，改为精确匹配文件名。
        （单片段调用的 output_dir 是隔离目录，目前只有一个文件；
          精确匹配使行为与 _find_textgrid_by_id 保持一致，也更安全。）
        """
        target = f"{utt_id}.TextGrid"
        for root, _, files in os.walk(output_dir):
            if target in files:
                return os.path.join(root, target)
        return None

    def _build_output(
        self,
        parsed:        Dict,
        original_text: str,
        language:      str,
        start_time:    float,
    ) -> List:
        """将 MFA 输出的词级别对齐映射回原始文本格式。"""
        mfa_words = parsed.get("words", [])
        if not mfa_words:
            return []

        if language in CJK_LANGUAGES:
            return self._build_output_zh(mfa_words, original_text, start_time)
        else:
            return self._build_output_en(mfa_words, original_text, start_time)

    def _build_output_zh(
        self,
        mfa_words:     List[Dict],
        original_text: str,
        start_time:    float,
    ) -> List:
        """中文: MFA 逐字对齐 → 映射回原文(含标点)。"""
        mfa_chars = []
        for w in mfa_words:
            text = w["word"].strip()
            if text and text not in ALL_PUNCTUATION:
                mfa_chars.append({
                    "char":  text,
                    "start": w["start"],
                    "end":   w["end"],
                })

        groups = self._parse_text_groups_zh(original_text)

        words_list = []
        mfa_idx    = 0

        for group_text, content_len in groups:
            if content_len == 0:
                continue
            if mfa_idx >= len(mfa_chars):
                if words_list:
                    last_end = words_list[-1][1][1]
                    words_list.append([group_text, [last_end, last_end + 0.01]])
                continue

            g_start = mfa_chars[mfa_idx]["start"] + start_time
            end_idx = min(mfa_idx + content_len - 1, len(mfa_chars) - 1)
            g_end   = mfa_chars[end_idx]["end"] + start_time
            mfa_idx += content_len

            words_list.append([group_text, [round(g_start, 6), round(g_end, 6)]])

        return words_list

    def _build_output_en(
        self,
        mfa_words:     List[Dict],
        original_text: str,
        start_time:    float,
    ) -> List:
        """英文: 将 MFA 词对齐映射回原文词序。"""
        original_words = original_text.split()
        if not original_words:
            return []

        def norm(w):
            return re.sub(r"[^\w']", "", w.lower())

        orig_norm = [norm(w) for w in original_words]
        mfa_norm  = [norm(w["word"]) for w in mfa_words]

        if len(orig_norm) == len(mfa_norm):
            matches = sum(1 for a, b in zip(orig_norm, mfa_norm) if a == b)
            if matches / len(orig_norm) > 0.8:
                return [
                    [w, [round(mfa_words[i]["start"] + start_time, 6),
                         round(mfa_words[i]["end"]   + start_time, 6)]]
                    for i, w in enumerate(original_words)
                ]

        alignment = self._word_align(orig_norm, mfa_norm, mfa_words)
        max_end   = max(w["end"] for w in mfa_words) if mfa_words else 1.0

        words_list = []
        for i, word in enumerate(original_words):
            if i in alignment:
                s, e = alignment[i]
                words_list.append([word, [round(s + start_time, 6),
                                          round(e + start_time, 6)]])
            else:
                s, e = self._interpolate(i, len(original_words), alignment, max_end)
                words_list.append([word, [round(s + start_time, 6),
                                          round(e + start_time, 6)]])

        return self._enforce_monotonic(words_list)

    def _word_align(
        self,
        known:      List[str],
        recog:      List[str],
        recog_data: List[Dict],
    ) -> Dict[int, Tuple[float, float]]:
        """简易编辑距离词对齐。"""
        from difflib import SequenceMatcher

        n, m = len(known), len(recog)
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0
        for i in range(1, n + 1):
            dp[i][0] = i
        for j in range(1, m + 1):
            dp[0][j] = j * 0.5

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if known[i-1] == recog[j-1] else (
                    1 - SequenceMatcher(None, known[i-1], recog[j-1]).ratio()
                )
                dp[i][j] = min(
                    dp[i-1][j]   + 1,
                    dp[i][j-1]   + 0.5,
                    dp[i-1][j-1] + cost,
                )

        alignment = {}
        i, j = n, m
        while i > 0 and j > 0:
            cost = 0 if known[i-1] == recog[j-1] else (
                1 - SequenceMatcher(None, known[i-1], recog[j-1]).ratio()
            )
            diag = dp[i-1][j-1] + cost

            if abs(dp[i][j] - diag) < 1e-9:
                if cost < 0.7:
                    alignment[i-1] = (recog_data[j-1]["start"],
                                      recog_data[j-1]["end"])
                i -= 1
                j -= 1
            elif abs(dp[i][j] - (dp[i-1][j] + 1)) < 1e-9:
                i -= 1
            else:
                j -= 1

        return alignment

    def _interpolate(
        self, idx: int, total: int, alignment: Dict, duration: float
    ) -> Tuple[float, float]:
        """线性插值未对齐词的时间戳。"""
        prev_end = 0.0
        prev_idx = -1
        for k in range(idx - 1, -1, -1):
            if k in alignment:
                prev_end = alignment[k][1]
                prev_idx = k
                break

        next_start = duration
        next_idx   = total
        for k in range(idx + 1, total):
            if k in alignment:
                next_start = alignment[k][0]
                next_idx   = k
                break

        gap  = max(next_idx - prev_idx - 1, 1)
        pos  = idx - prev_idx
        slot = (next_start - prev_end) / gap
        return (prev_end + (pos - 1) * slot, prev_end + pos * slot)

    @staticmethod
    def _enforce_monotonic(words_list: List) -> List:
        """确保时间戳单调递增。"""
        for i in range(1, len(words_list)):
            prev_end = words_list[i - 1][1][1]
            if words_list[i][1][0] < prev_end:
                words_list[i][1][0] = prev_end
            if words_list[i][1][1] <= words_list[i][1][0]:
                words_list[i][1][1] = words_list[i][1][0] + 0.01
        return words_list

    @staticmethod
    def _parse_text_groups_zh(text: str) -> List[Tuple[str, int]]:
        """中文分组: 标点附着到前一个字。"""
        groups = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            if ch in ALL_PUNCTUATION:
                if groups:
                    prev_text, prev_cnt = groups[-1]
                    groups[-1] = (prev_text + ch, prev_cnt)
                i += 1
                continue

            group_text = ch
            i += 1
            while i < n and text[i] in ALL_PUNCTUATION:
                group_text += text[i]
                i += 1
            groups.append((group_text, 1))

        return groups

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    def _fallback(
        self,
        text:       str,
        duration:   float,
        start_time: float,
        language:   str,
    ) -> Dict[str, Any]:
        """MFA 失败时的均匀分配 fallback。

        Bug 3 fix: 此方法不再自行检查 use_fallback 并 raise；
        调用方（_run_alignment）在调用前已确保 use_fallback=True。
        保留断言作为编程错误的最后防线。
        """
        if not self.use_fallback:
            # 此分支不应在正常流程中触达
            raise RuntimeError(
                "BUG: _fallback() called with use_fallback=False. "
                "This is a programming error in MFAAligner."
            )

        if language in CJK_LANGUAGES:
            groups = self._parse_text_groups_zh(text)
            total  = len(groups) or 1
            words_list = [
                [g, [round(start_time + (i / total) * duration, 6),
                     round(start_time + ((i + 1) / total) * duration, 6)]]
                for i, (g, _) in enumerate(groups)
            ]
        else:
            words = text.split()
            total = len(words) or 1
            words_list = [
                [w, [round(start_time + (i / total) * duration, 6),
                     round(start_time + ((i + 1) / total) * duration, 6)]]
                for i, w in enumerate(words)
            ]

        return {
            "words":      words_list,
            "phones":     [],
            "start":      start_time,
            "end":        round(start_time + duration, 6),
            "transcript": text,
            "language":   language,
        }


# ============================================================
# 独立函数接口 (兼容 whisper 版本的调用方式)
# ============================================================

_default_aligner: Optional[MFAAligner] = None


def get_aligner(**kwargs) -> MFAAligner:
    """获取或创建全局 MFAAligner 实例。"""
    global _default_aligner
    if _default_aligner is None:
        _default_aligner = MFAAligner(**kwargs)
    return _default_aligner


def generate_transcript_with_alignment(
    audio_data: np.ndarray,
    text:       str,
    sr:         int = 22050,
    start_time: float = 0.0,
    end_time:   Optional[float] = None,
    language:   Optional[str] = None,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """与 whisper 版本兼容的函数接口。"""
    aligner = get_aligner(**kwargs)
    return aligner.align_audio(
        audio_data=audio_data,
        sr=sr,
        text=text,
        language=language,
        start_time=start_time,
        end_time=end_time,
    )


def align_from_file(
    audio_path: str,
    text:       str,
    start_time: float = 0.0,
    end_time:   Optional[float] = None,
    language:   Optional[str] = None,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """与 whisper 版本兼容的文件接口。"""
    aligner = get_aligner(**kwargs)
    return aligner.align(
        audio_path=audio_path,
        text=text,
        language=language,
        start_time=start_time,
        end_time=end_time,
    )


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python transcript_mfa_aligner.py <audio.wav> <text> "
              "[language] [start_time] [end_time] [mfa_root]")
        sys.exit(1)

    audio_file = sys.argv[1]
    input_text = sys.argv[2]
    lang       = sys.argv[3] if len(sys.argv) > 3 else None
    t_start    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    t_end      = float(sys.argv[5]) if len(sys.argv) > 5 else None
    mfa_root   = sys.argv[6] if len(sys.argv) > 6 else None

    aligner = MFAAligner(keep_temp=False, mfa_root=mfa_root)
    result  = aligner.align(
        audio_file, input_text,
        language=lang, start_time=t_start, end_time=t_end,
    )

    if result:
        print(f"\nLanguage: {result['language']}")
        print(f"Duration: {result['start']:.3f} – {result['end']:.3f}")
        print(f"\n{'Word':<8s}  {'Start':>8s}  {'End':>8s}")
        print("-" * 30)
        for word, (s, e) in result["words"]:
            print(f"{word:<8s}  {s:>8.3f}  {e:>8.3f}")

        if result.get("phones"):
            print(f"\n{'Phone':<8s}  {'Start':>8s}  {'End':>8s}")
            print("-" * 30)
            for p in result["phones"][:20]:
                print(f"{p['phone']:<8s}  {p['start']:>8.3f}  {p['end']:>8.3f}")
            if len(result["phones"]) > 20:
                print(f"  ... ({len(result['phones'])} phones total)")
    else:
        print("Alignment failed.")