"""Launch run_glue.py on all local TPU chips with PyTorch/XLA multiprocessing.

Usage: python src/xla_spawn.py <run_glue.py args...>
Each spawned process calls run_glue._mp_fn, which parses sys.argv as usual.
"""
import os
import sys

import torch_xla.distributed.xla_multiprocessing as xmp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_glue  # noqa: E402

if __name__ == "__main__":
    # nprocs=None uses every local TPU chip (4 on a v5e-4 host)
    xmp.spawn(run_glue._mp_fn, args=(), nprocs=None)
