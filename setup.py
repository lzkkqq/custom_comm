# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build script for custom_comm.

Two output artifacts, split by C++ ABI to bridge CANN and torch:

  * libcustom_comm_impl.so  (-D_GLIBCXX_USE_CXX11_ABI=0)
    All .cc under ops/ that touch HCCL/CCU C++ headers. Built directly
    with g++ so we can override the ABI flag regardless of torch defaults.
    Exposes only extern "C" entry points declared in ops/<op>/inc/*.h.

  * custom_comm._C.*.so    (ABI=1, NpuExtension)
    Pytorch-side binding. Links dynamically against libcustom_comm_impl.so
    via RPATH=$ORIGIN. Must only include the public `extern "C"` headers.

Source layout (glob-discovered; no setup.py edits when adding a new op):

    ops/<op>/inc/          -- extern-C public header(s)
    ops/<op>/src/**/*.cc   -- joins the shim
    torch_ext/csrc/*.cpp   -- joins the torch extension

Skipped: any path containing a `vendor/`, `__pycache__/`, or `build/`
segment. See issue #10 for the ABI-boundary background.
"""

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from setuptools import setup


_TTY = sys.stdout.isatty()
_BOLD  = "\033[1m"  if _TTY else ""
_DIM   = "\033[2m"  if _TTY else ""
_CYAN  = "\033[36m" if _TTY else ""
_RED   = "\033[31m" if _TTY else ""
_GREEN = "\033[32m" if _TTY else ""
_RESET = "\033[0m"  if _TTY else ""

_TAG = f"{_CYAN}{_BOLD}[custom_comm]{_RESET}"

def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE / "python" / "custom_comm"
SHIM_BASENAME = "libcustom_comm_impl.so"
SHIM_OUT = PKG_DIR / SHIM_BASENAME

# ---------------------------------------------------------------------------
# CANN SDK discovery
# ---------------------------------------------------------------------------

SDK = Path(os.environ.get(
    "ASCEND_HOME_PATH",
    os.environ.get(
        "ASCEND_CANN_PACKAGE_PATH",
        "/usr/local/Ascend/ascend-toolkit/latest",
    ),
))

_hcomm_dev = SDK / "hcomm" / "hcomm"
if (SDK / "include" / "hccl" / "hccl_types.h").exists():
    SDK_ISYSTEM = []
    SDK_INC = [str(SDK / "include"), str(SDK / "pkg_inc")]
    SDK_LIB = [str(SDK / "lib64"), str(SDK / "x86_64-linux" / "lib64")]
elif (_hcomm_dev / "include" / "hccl" / "hccl_types.h").exists():
    SDK_ISYSTEM = [str(_hcomm_dev / "include")]
    SDK_INC = [str(_hcomm_dev / "pkg_inc")]
    SDK_LIB = [str(_hcomm_dev / "lib64"), str(SDK / "lib64")]
else:
    SDK_ISYSTEM = []
    SDK_INC = []
    SDK_LIB = []


# ---------------------------------------------------------------------------
# Source discovery (glob under ops/ and torch_ext/, skip WIP dirs)
# ---------------------------------------------------------------------------

SKIP_DIR_NAMES = {"vendor", "__pycache__", "build"}


def _discover(root: Path, suffix: str):
    results = []
    for path in root.rglob(f"*{suffix}"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        results.append(str(path.relative_to(HERE)))
    return sorted(results)


SHIM_SOURCES = _discover(HERE / "ops", ".cc")
BINDING_SOURCES = _discover(HERE / "torch_ext" / "csrc", ".cpp")

OP_INC_DIRS = sorted(
    str(p) for p in (HERE / "ops").glob("*/inc")
    if p.is_dir() and "vendor" not in p.parts
)


# ---------------------------------------------------------------------------
# Shim build: compile in parallel, then link to libcustom_comm_impl.so
# ---------------------------------------------------------------------------

_SHIM_INCLUDES = (
    OP_INC_DIRS
    + [
        os.path.join(str(SDK), "pkg_inc", "hcomm"),
        os.path.join(str(SDK), "pkg_inc", "hcomm", "ccu"),
        os.path.join(str(SDK), "include", "hccl"),
        str(HERE / "ops" / "allgather_batch" / "src"),
    ]
    + SDK_INC
)


# Release build (default). Set CUSTOM_COMM_RELEASE=0 to fall back to -O2 with
# assert() still firing — useful when debugging shim/runtime issues.
_RELEASE = os.environ.get("CUSTOM_COMM_RELEASE", "1") != "0"
_OPT_CFLAGS = ["-O3", "-DNDEBUG", "-flto"] if _RELEASE else ["-O2"]
_OPT_LDFLAGS = ["-flto"] if _RELEASE else []


_progress_lock = threading.Lock()
_progress = {"done": 0, "total": 0}


def _compile_one(rel_src: str, obj_dir: str) -> str:
    """Compile a single source file into object file. Returns the .o path."""
    abs_src = HERE / rel_src
    obj = os.path.join(obj_dir, rel_src.replace(os.sep, "__") + ".o")
    os.makedirs(os.path.dirname(obj), exist_ok=True)
    cxx = os.environ.get("CXX", "g++")
    cmd = [
        cxx, "-c",
        "-fPIC", "-std=c++17", *_OPT_CFLAGS, "-Wall",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        # torch_npu 2.9 bundles an older hccl_types.h that uses
        # HCCL_COMM_HCCL_QOS_CONFIG_NOT_SET; CANN 9.0 uses the shorter
        # HCCL_COMM_QOS_CONFIG_NOT_SET. Remove this once upstream is aligned.
        "-DHCCL_COMM_HCCL_QOS_CONFIG_NOT_SET=HCCL_COMM_QOS_CONFIG_NOT_SET",
        "-fvisibility=default",
    ]
    for inc in _SHIM_INCLUDES:
        cmd += ["-I", inc]
    for isys in SDK_ISYSTEM:
        cmd += ["-isystem", isys]
    cmd += [str(abs_src), "-o", obj]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(f"\n[custom_comm] compile FAILED: {rel_src}\n")
        sys.stderr.write(f"  $ {' '.join(cmd)}\n")
        if result.stdout:
            sys.stderr.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)

    if result.stderr.strip():
        # forward warnings without failing
        sys.stderr.write(result.stderr)

    with _progress_lock:
        _progress["done"] += 1
        done = _progress["done"]
        total = _progress["total"]
    width = len(str(total))
    print(f"  [{done:>{width}}/{total}] cc {rel_src}", flush=True)
    return obj


def build_shim():
    os.makedirs(str(PKG_DIR), exist_ok=True)
    obj_dir = str(HERE / "build" / "shim_obj")
    os.makedirs(obj_dir, exist_ok=True)

    jobs = int(os.environ.get("MAX_JOBS") or os.cpu_count() or 1)
    _progress["done"] = 0
    _progress["total"] = len(SHIM_SOURCES)
    _log(f"compiling {_progress['total']} shim sources "
         f"(MAX_JOBS={jobs}, mode={'release' if _RELEASE else 'default'})")

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_compile_one, s, obj_dir) for s in SHIM_SOURCES]
        objs = [f.result() for f in futures]

    cxx = os.environ.get("CXX", "g++")
    link_cmd = [cxx, "-shared", "-fPIC", *_OPT_LDFLAGS,
                f"-Wl,-soname,{SHIM_BASENAME}"] + objs
    for libdir in SDK_LIB:
        link_cmd += ["-L", libdir]
    link_cmd += ["-lhccl", "-lhcomm", "-lascendcl",
                 "-o", str(SHIM_OUT)]

    _log(f"linking -> {SHIM_BASENAME}")
    lr = subprocess.run(link_cmd, capture_output=True, text=True)
    if lr.returncode != 0:
        sys.stderr.write(f"\n{_RED}{_BOLD}[custom_comm] link failed{_RESET}\n")
        sys.stderr.write(f"  $ {' '.join(link_cmd)}\n")
        if lr.stdout:
            sys.stderr.write(lr.stdout)
        if lr.stderr:
            sys.stderr.write(lr.stderr)
        raise subprocess.CalledProcessError(lr.returncode, link_cmd)
    if lr.stderr.strip():
        sys.stderr.write(lr.stderr)

    _verify_shim_abi()
    _log(f"{_GREEN}ok{_RESET}  {SHIM_OUT.relative_to(HERE)}")


def _verify_shim_abi() -> None:
    import shutil
    nm = shutil.which("nm")
    if not nm:
        return
    raw = subprocess.check_output([nm, "-D", str(SHIM_OUT)], text=True)
    cxx11_lines = [l for l in raw.splitlines() if "__cxx11" in l]
    t_lines = [l for l in raw.splitlines() if " T " in l]
    if cxx11_lines:
        raise RuntimeError(
            f"ABI check failed: shim carries {len(cxx11_lines)} CXX11-ABI symbols "
            "(expected 0). Verify -D_GLIBCXX_USE_CXX11_ABI=0 is applied."
        )
    if not t_lines:
        raise RuntimeError(
            "ABI check failed: shim exports no T (defined) symbols. "
            "Check -fvisibility=default and that extern \"C\" functions exist."
        )


# ---------------------------------------------------------------------------
# Torch extension
# ---------------------------------------------------------------------------

ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension
    from torch_npu.utils.cpp_extension import NpuExtension

    class BuildExtWithShim(BuildExtension):
        def run(self):
            build_shim()
            super().run()

    # torch_npu 2.7+ wheel ships a bundled hccl_types.h that still uses the
    # stuttered name HCCL_COMM_HCCL_QOS_CONFIG_NOT_SET. CANN 9.0's SDK
    # renamed it to HCCL_COMM_QOS_CONFIG_NOT_SET. Alias old to new so torch
    # bindings still compile when they transitively include torch_npu's copy.
    # Drop this when torch_npu ships headers matching CANN 9.0.
    _binding_macros = [
        ("HCCL_COMM_HCCL_QOS_CONFIG_NOT_SET", "HCCL_COMM_QOS_CONFIG_NOT_SET"),
    ]

    ext_modules = [NpuExtension(
        name="custom_comm._C",
        sources=BINDING_SOURCES,
        include_dirs=OP_INC_DIRS + SDK_INC,
        library_dirs=[str(PKG_DIR)] + SDK_LIB,
        libraries=["custom_comm_impl"],
        extra_compile_args=[
            "-std=c++17",
            *_OPT_CFLAGS,
        ] + [f"-isystem{d}" for d in SDK_ISYSTEM],
        extra_link_args=["-Wl,-rpath,$ORIGIN", *_OPT_LDFLAGS],
        define_macros=_binding_macros,
    )]
    cmdclass = {"build_ext": BuildExtWithShim}
except ImportError:
    pass


setup(
    name="custom_comm",
    version="0.1.0",
    packages=["custom_comm", "custom_comm.converters"],
    package_dir={"": "python"},
    package_data={"custom_comm": ["libcustom_comm_impl.so"]},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
