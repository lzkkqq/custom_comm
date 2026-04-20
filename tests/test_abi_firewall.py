# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ABI boundary sanity: verify the shim / binding split is intact.

libcustom_comm_impl.so must be ABI=0 (matches libhcomm.so) and expose the
C API; custom_comm._C.so must be ABI=1 (matches torch_npu) and link to the
shim. If this invariant breaks, CCU paths segfault at std::string destruction
(see issue #10).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _pkg_dir() -> Path:
    import custom_comm
    return Path(custom_comm.__file__).parent


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        pytest.skip(f"{name} not in PATH")
    return path


def _nm(so: Path) -> str:
    return subprocess.check_output([_tool("nm"), "-D", str(so)], text=True)


def _readelf(so: Path) -> str:
    return subprocess.check_output([_tool("readelf"), "-d", str(so)], text=True)


@pytest.mark.ext
def test_shim_is_abi_zero():
    """libcustom_comm_impl.so must have 0 __cxx11 symbols."""
    shim = _pkg_dir() / "libcustom_comm_impl.so"
    if not shim.exists():
        pytest.skip("shim not built")
    symbols = _nm(shim)
    count = sum(1 for line in symbols.splitlines() if "__cxx11" in line)
    assert count == 0, (
        f"libcustom_comm_impl.so has {count} __cxx11 symbols — "
        "it must compile with -D_GLIBCXX_USE_CXX11_ABI=0"
    )


def test_shim_exports_at_least_one_c_api():
    """The shim must export at least one extern-C symbol.

    We don't name a specific function so new ops don't have to update the test.
    What we require is: at least one T (global defined text) symbol that's not
    C++ mangled — i.e., an `extern "C"` API actually made it out.
    """
    shim = _pkg_dir() / "libcustom_comm_impl.so"
    if not shim.exists():
        pytest.skip("shim not built")
    t_symbols = [
        line.rsplit(None, 1)[-1]
        for line in _nm(shim).splitlines()
        if " T " in line
    ]
    # Filter out C++ mangled names (they start with _Z).
    c_symbols = [s for s in t_symbols if not s.startswith("_Z")]
    assert c_symbols, (
        "shim exports no extern-C entry point; "
        "check ops/<op>/inc/*.h declares `extern \"C\"`."
    )


def test_binding_imports_from_shim():
    """binding must have at least one U symbol that the shim defines as T.

    That's the actual, op-agnostic guarantee we need: the C ABI boundary is
    exercised. If we ever add a second op, this test keeps working without
    edits; if someone accidentally detaches _C.so from the shim, it fails.
    """
    shim = _pkg_dir() / "libcustom_comm_impl.so"
    bindings = sorted(_pkg_dir().glob("_C*.so"))
    if not shim.exists() or not bindings:
        pytest.skip("artifacts not built")

    def _syms(path, kind):
        return {
            line.rsplit(" ", 1)[-1]
            for line in _nm(path).splitlines()
            if f" {kind} " in line
        }

    shim_defined = _syms(shim, "T")
    binding_undefined = _syms(bindings[0], "U")
    bridge = shim_defined & binding_undefined
    assert bridge, (
        f"No C-API symbols bridge _C.so to libcustom_comm_impl.so. "
        f"Shim T symbols ({len(shim_defined)} total), "
        f"binding U symbols ({len(binding_undefined)} total) — "
        "but nothing in common. The binding is not actually linking to the shim."
    )


@pytest.mark.ext
def test_binding_is_cxx11_abi():
    """custom_comm._C.*.so must contain __cxx11 symbols (ABI=1)."""
    bindings = sorted(_pkg_dir().glob("_C*.so"))
    if not bindings:
        pytest.skip("_C*.so not built")
    symbols = _nm(bindings[0])
    assert "__cxx11" in symbols, (
        "custom_comm._C.so has no __cxx11 symbols — it must match "
        "torch's -D_GLIBCXX_USE_CXX11_ABI=1"
    )


@pytest.mark.ext
def test_binding_links_to_shim():
    """custom_comm._C.*.so must list libcustom_comm_impl.so in NEEDED."""
    bindings = sorted(_pkg_dir().glob("_C*.so"))
    if not bindings:
        pytest.skip("_C.so not built")
    readelf = _tool("readelf")
    out = subprocess.check_output([readelf, "-d", str(bindings[0])], text=True)
    assert "libcustom_comm_impl.so" in out, (
        "_C.so doesn't have libcustom_comm_impl.so in its NEEDED list; "
        "linker did not pick up -lcustom_comm_impl"
    )
    assert "$ORIGIN" in out, (
        "_C.so has no $ORIGIN in RUNPATH; shim won't be resolved at runtime"
    )


