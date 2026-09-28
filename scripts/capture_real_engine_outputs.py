"""Capture real TensorRT engine outputs + the real GPU-kernel risk score for each, on
E3 -- second half of Codex's mandatory audit requirement (2026-09-29 review): "uu tien
them mot so output that lay tu TensorRT engine, khong chi synthetic logits". Runs the
SAME production code path measure_router_overhead.py uses in deployment
(`infer_no_copy` + `risk_score_gpu`, no intermediate host copy for the kernel's own
read) so this also validates end-to-end integration (real engine -> real GPU buffer ->
kernel), not just the kernel's arithmetic in isolation (already covered by
`audit_gpu_risk_kernel.py`'s synthetic-logits sweep).

Also saves the FULL host-side copy of each sample's raw output (via `infer_sync`, a
second, separate inference call -- deliberately not read from the same buffer the
no-copy path just consumed, so the two capture methods can't influence each other) so
the real PyTorch reference (`router.risk_probe.compute_risk_score`, which needs torch
-- not installed on any edge device in this project) can be run on the exact same real
network output later, on a machine that has torch.

Usage (on E3, inside imavis_overhead/):
    PATH=$PATH:/usr/local/cuda-11.4/bin ./venv/bin/python3 capture_real_engine_outputs.py \
        --engine-dir . --n-samples 10 --output-dir real_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "bool"):
    setattr(np, "bool", bool)  # noqa: B010

import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from pycuda.compiler import SourceModule

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
LEVELS = ["tiny", "small", "medium", "large"]

_ENTROPY_KERNEL_SRC = """
extern "C" __global__ void entropy_kernel(const float *logits, float *entropy_out, int num_classes, int num_pixels) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= num_pixels) return;
    float max_val = -1e30f;
    for (int c = 0; c < num_classes; c++) {
        float v = logits[c * num_pixels + pixel];
        if (v > max_val) max_val = v;
    }
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        sum_exp += expf(logits[c * num_pixels + pixel] - max_val);
    }
    float entropy = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        float p = expf(logits[c * num_pixels + pixel] - max_val) / sum_exp;
        entropy -= p * logf(fmaxf(p, 1e-12f));
    }
    entropy_out[pixel] = entropy;
}
"""
_entropy_module = SourceModule(_ENTROPY_KERNEL_SRC, no_extern_c=True)
_entropy_kernel = _entropy_module.get_function("entropy_kernel")


class TrtEngine:
    def __init__(self, engine_path: Path) -> None:
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings: list[int] = []
        self.host_in: Any = None
        self.device_in: Any = None
        self.host_out: Any = None
        self.device_out: Any = None
        self.output_shape: tuple[int, ...] = ()
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                self.host_in, self.device_in = host_mem, device_mem
                self.input_shape = tuple(shape)
            else:
                self.host_out, self.device_out = host_mem, device_mem
                self.output_shape = tuple(shape)
        _, num_classes, height, width = self.output_shape
        self.num_classes = num_classes
        self.num_pixels = height * width
        self.entropy_device = cuda.mem_alloc(self.num_pixels * 4)
        self.entropy_host = cuda.pagelocked_empty(self.num_pixels, np.float32)

    def infer_sync(self, input_array: np.ndarray) -> np.ndarray:
        np.copyto(self.host_in, input_array.ravel())
        cuda.memcpy_htod_async(self.device_in, self.host_in, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_out, self.device_out, self.stream)
        self.stream.synchronize()
        return np.array(self.host_out, copy=True)

    def infer_no_copy(self, input_array: np.ndarray) -> None:
        np.copyto(self.host_in, input_array.ravel())
        cuda.memcpy_htod_async(self.device_in, self.host_in, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()

    def risk_score_gpu(self) -> float:
        """Production path: reads self.device_out directly, no intermediate host
        copy for the kernel's own input -- exactly what measure_router_overhead.py's
        deployed 'gpu' entropy backend does."""
        block = 256
        grid = (self.num_pixels + block - 1) // block
        _entropy_kernel(
            self.device_out, self.entropy_device, np.int32(self.num_classes), np.int32(self.num_pixels),
            block=(block, 1, 1), grid=(grid, 1), stream=self.stream,
        )
        cuda.memcpy_dtoh_async(self.entropy_host, self.entropy_device, self.stream)
        self.stream.synchronize()
        return float(self.entropy_host.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", default=".")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--output-dir", default="real_outputs")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for level in LEVELS:
        engine = TrtEngine(Path(f"{args.engine_dir}/pace_seg_{level}.engine"))
        print(f"{level}: input_shape={engine.input_shape} output_shape={engine.output_shape}")
        logits_samples = []
        gpu_scores = []
        for i in range(args.n_samples):
            input_array = rng.standard_normal(engine.input_shape, dtype=np.float32)
            # Two SEPARATE inference calls on purpose: infer_no_copy+risk_score_gpu
            # (the real deployed path, reading straight from device_out) computes the
            # GPU score; infer_sync (a second, independent call) captures the full
            # host-side array for the offline PyTorch reference -- they cannot leak
            # into each other's result this way.
            engine.infer_no_copy(input_array)
            gpu_score = engine.risk_score_gpu()
            host_out = engine.infer_sync(input_array)
            logits_samples.append(host_out.copy())
            gpu_scores.append(gpu_score)
            print(f"  sample {i}: gpu_score={gpu_score:.6f}")
        np.savez_compressed(
            out_dir / f"{level}.npz",
            logits=np.stack(logits_samples),
            gpu_risk_score=np.array(gpu_scores, dtype=np.float64),
            output_shape=np.array(engine.output_shape),
        )
        print(f"wrote {out_dir / f'{level}.npz'}")


if __name__ == "__main__":
    main()
