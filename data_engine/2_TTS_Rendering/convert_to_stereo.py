#!/usr/bin/env python3
"""
递归将目录下所有 WAV / FLAC 文件转换为双声道，输出为 _stereo.flac。
例如：music.wav → music_stereo.flac

用法：
    python convert_to_stereo.py [目录]
"""

import sys
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path


def to_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return np.column_stack([data, data])
    if data.shape[1] == 2:
        return data
    left  = data[:, 0]
    right = np.mean(data[:, 1:], axis=1)
    return np.column_stack([left, right])


def process_file(src: Path) -> str:
    dst = src.with_name(src.stem + '_stereo.flac')
    src_json = src.with_suffix('.json')
    dst_json = dst.with_suffix('.json')
    if not dst_json.exists():
        # 快速复制 JSON 文件，保持元信息一致
        assert src_json.exists()
        dst_json.write_bytes(src_json.read_bytes())
    if dst.exists():
        return f"跳过：{dst.name} 已存在"
    compressed_dst = dst.with_name(dst.stem.replace('_stereo', '_stereo_compressed') + dst.suffix)
    if compressed_dst.exists():
        return f"跳过：{compressed_dst.name} 已存在"
    info = sf.info(str(src))
    data, sr = sf.read(str(src), always_2d=False)
    stereo = to_stereo(data)

    sf.write(str(dst), stereo, sr, format='FLAC')

    if data.ndim == 1:
        return f"单声道 → {dst.name}（声道 0 复制）"
    else:
        n = info.channels
        return f"{n} 声道 → {dst.name}（声道 1~{n-1} 已合并）"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', default='./tts_output', help='音频文件所在目录')
    args = parser.parse_args()

    src_dir = Path(args.directory).resolve()
    if not src_dir.is_dir():
        print(f'错误："{src_dir}" 不是有效目录。')
        sys.exit(1)

    suffixes = {'.wav', '.flac'}
    files = sorted(
        p for p in src_dir.rglob('*')
        if p.is_file()
        and p.suffix.lower() in suffixes
        and not p.stem.endswith('_stereo')
        and not p.stem.endswith('_compressed')
    )

    if not files:
        print('未找到 WAV 或 FLAC 文件。')
        return

    print(f'共找到 {len(files)} 个文件\n')
    ok = err = 0

    for src in files:
        rel = src.relative_to(src_dir)
        print(f'  处理：{rel}')
        try:
            desc = process_file(src)
            print(f'    ✓ {desc}')
            ok += 1
        except Exception as exc:
            print(f'    ✗ 错误：{exc}')
            err += 1

    print(f'\n完成！  成功：{ok}  失败：{err}')


if __name__ == '__main__':
    main()