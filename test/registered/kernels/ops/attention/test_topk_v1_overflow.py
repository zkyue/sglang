"""Correctness guard for the DSA indexer JIT top-k v1 candidate buffer.

The radix select stages every element whose coarse (fp16 high byte) bin equals
the threshold bin into a fixed ``SMEM_INPUT_SIZE`` buffer. Concentrated scores
collapse that histogram, so the bin can hold far more elements than the buffer
and the surplus used to be dropped silently, yielding a wrong top-k.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

K = 512
PAGE_SIZE = 1  # identity page table, so emitted slots equal raw token positions

# ~33k of the 65504 elements land in the threshold bin, against SMEM_INPUT_SIZE=8192.
_OVERFLOW_CHILD = """
import torch
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512

batch, seq, k = 4, 65504, 512
torch.manual_seed(42)
scores = 10.0 + 0.1 * torch.randn(batch, seq, dtype=torch.float32, device="cuda")
seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
page_table = torch.arange(seq, dtype=torch.int32, device="cuda").repeat(batch, 1)
out = torch.empty((batch, k), dtype=torch.int32, device="cuda")
topk_transform_512(scores, seq_lens, page_table, out, 1)
torch.cuda.synchronize()
"""


@pytest.mark.skipif(
    torch.version.hip is not None, reason="HIP dispatches to the AOT kernel"
)
@torch.inference_mode()
def test_topk_v1_within_capacity() -> None:
    batch, seq = 4, 30938
    torch.manual_seed(42)
    scores = torch.randn(batch, seq, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    page_table = torch.arange(seq, dtype=torch.int32, device="cuda").repeat(batch, 1)
    out = torch.empty((batch, K), dtype=torch.int32, device="cuda")
    raw = torch.empty((batch, K), dtype=torch.int32, device="cuda")

    topk_transform_512(scores, seq_lens, page_table, out, PAGE_SIZE, raw)
    torch.cuda.synchronize()

    ref = torch.topk(scores, K, dim=-1, sorted=True).values
    got = scores.gather(1, raw.long()).sort(dim=-1, descending=True).values
    torch.testing.assert_close(got, ref)


@pytest.mark.skipif(
    torch.version.hip is not None, reason="HIP dispatches to the AOT kernel"
)
def test_topk_v1_candidate_buffer_overflow_is_loud() -> None:
    # The device-side assert poisons the CUDA context, so this runs out of process.
    proc = subprocess.run(
        [sys.executable, "-c", _OVERFLOW_CHILD], capture_output=True, text=True
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "threshold bin overflowed the candidate buffer" in proc.stderr, proc.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
