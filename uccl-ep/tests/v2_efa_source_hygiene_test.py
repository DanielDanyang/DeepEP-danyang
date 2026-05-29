from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REMOVED_LEGACY_PATHS = [
    "bench/proxy_rdma_fifo.py",
    "bench/v2_proxy_smoke.py",
    "deep_ep_v2_wrapper/deep_ep/proxy_transport.py",
    "deep_ep_v2_wrapper/deep_ep/utils_uccl.py",
    "include/bench_kernel.cuh",
    "include/bench_utils.hpp",
    "include/ep_runtime.cuh",
    "include/uccl_bench.hpp",
    "src/bench_kernel.cu",
    "src/ep_runtime.cu",
    "src/uccl_bench.cpp",
]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_removed_legacy_paths_do_not_exist() -> None:
    for path in REMOVED_LEGACY_PATHS:
        assert not (ROOT / path).exists(), path


def test_python_wrapper_does_not_export_legacy_transport() -> None:
    init_py = read("deep_ep_v2_wrapper/deep_ep/__init__.py")
    assert "ProxyTransport" not in init_py
    assert "utils_uccl" not in init_py


def test_extension_build_is_v2_only() -> None:
    setup_py = read("setup.py")
    makefile = read("Makefile")

    assert '"./src/uccl_ep.cc"' in setup_py
    assert '"./src/v2_efa_runtime.cc"' in setup_py
    assert '"./src/v2_efa_deep_ep_jit.cc"' in setup_py
    assert "glob(\"./src/*.cu\")" not in setup_py
    assert "SRC_CU  :=" in makefile
    assert "SRC_CC_LIB := src/v2_efa_runtime.cc src/v2_efa_deep_ep_jit.cc" in makefile

    forbidden_build_sources = [
        "internode.cu",
        "intranode.cu",
        "layout.cu",
        "ep_runtime.cu",
        "bench_kernel.cu",
        "uccl_bench.cpp",
    ]
    for source in forbidden_build_sources:
        assert source not in setup_py
        assert source not in makefile


if __name__ == "__main__":
    test_removed_legacy_paths_do_not_exist()
    test_python_wrapper_does_not_export_legacy_transport()
    test_extension_build_is_v2_only()
