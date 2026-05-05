"""Model inference adapters for MultiTalkBench.

Every adapter is a standalone script. The contract::

    python models/<adapter>.py \\
        --data-root  <hf-dataset-clone> \\
        --ckpt-dir   <model-checkpoint> \\
        --output-dir <where-to-write-offline-output> \\
        [--use-detailed-sys] \\
        [--rank N --world-size W] \\
        [--limit K] [--force]

Each adapter writes one JSON per sample to::

    <output-dir>/<lang>/<sample_id>.json

following the schema consumed by ``benchmark.load_offline_utterances``::

    {"alignments": [[token, [start, end], "SPEAKER_MAIN"], ...],
     "decoded_text": "...",
     "metadata": {...}}

See ``adapter.py`` for the shared CLI / iterator helper.
"""
