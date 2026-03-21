from setuptools import Extension, setup

try:
    import pybind11
except Exception as exc:
    raise RuntimeError("pybind11 is required. Install with: pip install pybind11") from exc

ext_modules = [
    Extension(
        "shogi3d_core",
        ["cpp_core/shogi3d_core.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/O2", "/std:c++17"],
    )
]

setup(
    name="shogi3d_core",
    version="0.1.0",
    description="C++ core helpers for 3D shogi",
    ext_modules=ext_modules,
)
