"""
Phase 1 checkpoint script.

Run this after setting up the venv and installing dependencies.
It does NOT touch the speaker-extraction model yet (that's Phase 4) --
it only proves the environment is sane: CUDA visible, torch working,
and a real (tiny) tensor op actually running on the GPU.

Usage:
    python verify_setup.py
"""

import sys
import time


def check_python_version():
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 10):
        print("  WARNING: Python 3.10+ recommended for current torch wheels.")
    else:
        print("  OK")


def check_torch():
    try:
        import torch
    except ImportError:
        print("torch is NOT installed. See README.md for the correct install command.")
        return None
    print(f"torch version: {torch.__version__}")
    return torch


def check_cuda(torch):
    if torch is None:
        return
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if not available:
        print("  torch cannot see your GPU. Common causes:")
        print("  - You installed the CPU-only torch wheel (pip install torch with no index-url)")
        print("  - NVIDIA driver is outdated for the CUDA build you installed")
        print("  - You're inside a venv that doesn't have the CUDA torch build")
        return

    device_count = torch.cuda.device_count()
    print(f"  CUDA device count: {device_count}")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"  [{i}] {name}  (compute capability {cap[0]}.{cap[1]}, {mem_gb:.1f} GB)")

    print(f"  cuDNN version: {torch.backends.cudnn.version()}")


def check_gpu_inference(torch):
    """Do a small but real op on the GPU and time it, to prove the pipeline
    actually executes on-device rather than silently falling back to CPU."""
    if torch is None or not torch.cuda.is_available():
        print("Skipping GPU inference check (no CUDA torch available).")
        return

    device = torch.device("cuda:0")

    # Warm-up (first CUDA call has fixed init overhead, not representative)
    _ = torch.randn(1024, 1024, device=device) @ torch.randn(1024, 1024, device=device)
    torch.cuda.synchronize()

    # A matmul size roughly comparable to a conv layer's inner loop --
    # not a real model, just proof the GPU is doing work quickly.
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    start = time.perf_counter()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / 20

    print(f"GPU matmul (4096x4096) avg time: {elapsed_ms:.2f} ms/iter")
    if elapsed_ms > 15:
        print("  This seems slow for an RTX 4070 Super -- double check no other")
        print("  heavy GPU process (game, another model) is running.")
    else:
        print("  OK, GPU is responsive.")


def check_audio_libs():
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is NOT installed.")
        return
    print("sounddevice OK. Available devices:")
    for i, dev in enumerate(sd.query_devices()):
        marker = ""
        if dev["max_input_channels"] > 0:
            marker += " [IN]"
        if dev["max_output_channels"] > 0:
            marker += " [OUT]"
        print(f"  [{i}] {dev['name']}{marker}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 environment check")
    print("=" * 60)
    check_python_version()
    print()
    torch = check_torch()
    print()
    check_cuda(torch)
    print()
    check_gpu_inference(torch)
    print()
    check_audio_libs()
    print()
    print("If everything above says OK and your RTX 4070 Super shows up,")
    print("Phase 1 goal is met. Next: Phase 2, audio device manager.")
