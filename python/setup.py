from glob import glob
from setuptools import setup, Extension

__version__ = "0.0.1"

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        "cumpsgemm_hijack_control",
        sorted(glob("src/*.cpp")),
        include_dirs = [
            "../include/",
            get_pybind_include(),
            get_pybind_include(user=True)
            ],
        language='c++'
    ),
]

setup(
    name="cumpsgemm_hijack_control",
    version=__version__,
    author="enp1s0",
    author_email="mutsuki@momo86.net",
    url="https://github.com/enp1s0/cumpsgemm",
    description="cuMpSGEMM hijacking control API",
    long_description="",
    ext_modules=ext_modules,
    zip_safe=False,
)
