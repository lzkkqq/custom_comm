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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from setuptools import setup


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


def _compile_one(rel_src: str, obj_dir: str) -> str:
    """Compile a single source file into object file. Returns the .o path."""
    abs_src = HERE / rel_src
    obj = os.path.join(obj_dir, rel_src.replace(os.sep, "__") + ".o")
    os.makedirs(os.path.dirname(obj), exist_ok=True)
    cxx = os.environ.get("CXX", "g++")
    cmd = [
        cxx, "-c",
        "-fPIC", "-std=c++17", "-O2", "-Wall",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-DHCCL_COMM_HCCL_QOS_CONFIG_NOT_SET=HCCL_COMM_QOS_CONFIG_NOT_SET",
        "-fvisibility=default",
    ]
    for inc in _SHIM_INCLUDES:
        cmd += ["-I", inc]
    for isys in SDK_ISYSTEM:
        cmd += ["-isystem", isys]
    cmd += [str(abs_src), "-o", obj]
    subprocess.check_call(cmd)
    return obj


def build_shim():
    os.makedirs(str(PKG_DIR), exist_ok=True)
    obj_dir = str(HERE / "build" / "shim_obj")
    os.makedirs(obj_dir, exist_ok=True)

    jobs = int(os.environ.get("MAX_JOBS") or os.cpu_count() or 1)
    print(f"[custom_comm] compiling {len(SHIM_SOURCES)} shim sources (MAX_JOBS={jobs})")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_compile_one, s, obj_dir) for s in SHIM_SOURCES]
        objs = [f.result() for f in futures]  # propagates exceptions

    # Link
    cxx = os.environ.get("CXX", "g++")
    link_cmd = [cxx, "-shared", "-fPIC",
                f"-Wl,-soname,{SHIM_BASENAME}"] + objs
    for libdir in SDK_LIB:
        link_cmd += ["-L", libdir]
    link_cmd += ["-lhccl", "-lhcomm", "-lascendcl",
                 "-o", str(SHIM_OUT)]
    subprocess.check_call(link_cmd)

    _verify_shim_abi()


def _verify_shim_abi():
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
        ] + [f"-isystem{d}" for d in SDK_ISYSTEM],
        extra_link_args=["-Wl,-rpath,$ORIGIN"],
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
