"""What this PC can give the turn search: where the nets run (its NVIDIA GPU, or the processor) and how many worker
processes.  Decided once, when the service starts, from what is really there - nothing is assumed about the PC.

  the GPU is used only when ALL of this holds: the Python's torch was built with CUDA, it finds a GPU, a small
  computation on it gives the right number, and it has free memory for at least MIN_GPU_WORKERS workers and the
  service.  Anything else - a torch without CUDA, no NVIDIA card (an AMD or Intel GPU cannot be used by this torch), a
  driver the build does not talk to, a card torch no longer supports, too little free memory - is the processor,
  with the reason said in one line.
  workers: one per physical core, at most MAX_WORKERS, and no more than the free memory holds (each worker loads every
  net and an engine of its own); on the GPU also no more than its free memory holds.

A GPU that fails LATER (out of memory because another program took it, a driver reset) ends the service with the exit
code GPU_FAILED; whoever started it starts it again with --device cpu (wotr_bridge/core/turn_search.py does).
Too little memory for even one worker ends it with CANNOT_RUN: the turn search is then off, the game plays on.

Measured on the development PC (10 workers, RTX 5090, 2026-11-14): 1.6 - 1.9 GB of memory per worker, 1.0 GB for the
service; 8.7 GB of GPU memory for the 11 processes.  The numbers below are those, rounded up.
This module imports nothing of the kit, and torch only inside device().
"""
import os
import re

GPU_FAILED = 4                  # exit code: the GPU failed while the service ran
CANNOT_RUN = 5                  # exit code: this PC cannot run the turn search now (memory)
MAX_WORKERS = 10
WORKER_MB = 2000                # memory of one worker
SERVICE_MB = 1200               # memory of the service process itself
SPARE_MB = 2000                 # memory left alone (the game's server and client are running too)
GPU_PROCESS_MB = 850            # GPU memory of one process (a worker, or the service)
GPU_SPARE_MB = 500
MIN_GPU_WORKERS = 3             # with fewer workers than this on the GPU, the processor with all its workers is used


def cores():
    """Physical cores (the engine's work does not gain from hyper-threads: measured, 8 = 10 = 16 workers on 8 cores)."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def free_mb():
    """Free physical memory in MB, or None when the system does not say."""
    try:
        import psutil
        return int(psutil.virtual_memory().available // 2 ** 20)
    except Exception:
        pass
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in ("total", "avail", "total_page", "avail_page", "total_virtual", "avail_virtual", "extended")]
        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail // 2 ** 20)
    except Exception:
        pass
    return None


def gpu_free_mb(torch, index):
    """Free memory of the GPU in MB.  torch's own number is this PROCESS's share under Windows (measured: 30 GB "free"
    while another program held 28 of the card's 32), so the driver's tool is asked for the card's real use too, and
    the smaller number counts; without the tool, torch's."""
    free = int(torch.cuda.mem_get_info(0)[0] // 2 ** 20)
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits", "-i", str(index)],
                           capture_output=True, text=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        total, used = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
        free = min(free, total - used)
    except Exception:
        pass
    return free


def workers(wanted, memory_mb=None):
    """-> (number of workers, why).  wanted: a number, or "auto".  0 workers = the turn search cannot run now."""
    memory_mb = free_mb() if memory_mb is None else memory_mb
    fit = MAX_WORKERS if memory_mb is None else int((memory_mb - SERVICE_MB - SPARE_MB) // WORKER_MB)
    n = min(cores(), MAX_WORKERS) if str(wanted) == "auto" else max(1, int(wanted))
    why = "%s: %d physical cores" % (wanted, cores()) if str(wanted) == "auto" else "asked for %d" % n
    if fit < n:
        why += "; only %d MB of memory are free, enough for %d" % (memory_mb, max(fit, 0))
        n = max(fit, 0)
    return n, why


def device(wanted, n_workers, gpu="0"):
    """-> (device, number of workers, why).  wanted: "auto" (the GPU if it can be used), "cuda" (the same: a GPU that
    cannot be used is never an error) or "cpu".  gpu: which GPU, as CUDA_VISIBLE_DEVICES names it."""
    if wanted == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu", n_workers, "the processor was asked for"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)        # the game's launcher hides the GPU from its own server
    try:
        import torch
        if not torch.backends.cuda.is_built():
            why = "this Python's torch (%s) was built without CUDA" % torch.__version__
        elif not torch.cuda.is_available():
            why = "torch %s finds no NVIDIA GPU it can use (no such card, or a driver too old for this build)" % torch.__version__
        else:
            x = torch.arange(64, dtype=torch.float32, device="cuda:0")
            got = float((x @ x).item())                  # a real computation: a card the build cannot run fails here
            if got != 85344.0:
                why = "the GPU's first computation gave %r" % got
            else:
                free = gpu_free_mb(torch, gpu)
                fit = int((free - GPU_SPARE_MB) // GPU_PROCESS_MB) - 1                   # one share is the service's
                name = torch.cuda.get_device_name(0)
                if fit >= min(n_workers, MIN_GPU_WORKERS):
                    n = min(n_workers, fit)
                    return "cuda", n, "%s, %d MB free%s" % (name, free, "" if n == n_workers else
                                                            " - room for %d of the %d workers" % (n, n_workers))
                why = "%s has only %d MB free, not enough for %d workers" % (name, free, MIN_GPU_WORKERS)
    except Exception as error:
        why = "the GPU could not be used: %s" % (str(error).replace("\n", " ")[:200],)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return "cpu", n_workers, why


def is_gpu_error(text):
    """Is this error (its text or traceback) the GPU's - out of its memory, a CUDA or cuDNN failure, a lost device?"""
    return bool(re.search(r"CUDA|cuDNN|CUBLAS|cublas|cudnn", str(text)))
