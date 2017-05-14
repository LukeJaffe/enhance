#-------------------------------------------
# NVIDIA file collecting utilities
#
#       This module contains utilities for finding
#       and copying NVIDIA driver-related files.
#
# Copyright 2016 LLNL
# Written by:
#     Luke Jaffe <jaffe5@llnl.gov>
#     Mike Goldman <goldman21@llnl.gov>
#-------------------------------------------

import string
import os
from subprocess import Popen, PIPE

_headers = [
    '/usr/include/GL',
]

# taken from nvidia-docker/volumes.go
_binaries = [
    # binaries
    #"nvidia-modprobe",       # Kernel module loader
    #"nvidia-settings",       # X server settings
    #"nvidia-xconfig",        # X xorg.conf editor
    "nvidia-cuda-mps-control", # Multi process service CLI
    "nvidia-cuda-mps-server",  # Multi process service server
    "nvidia-debugdump",        # GPU coredump utility
    "nvidia-persistenced",     # Persistence mode utility
    "nvidia-smi",              # System management interface
]

_libraries = [
    # ------- X11 -------

    #"libnvidia-cfg.so",  # GPU configuration (used by nvidia-xconfig)
    #"libnvidia-gtk2.so", # GTK2 (used by nvidia-settings)
    #"libnvidia-gtk3.so", # GTK3 (used by nvidia-settings)
    #"libnvidia-wfb.so",  # Wrapped software rendering module for X server
    #"libglx.so",         # GLX extension module for X server

    # ----- Compute -----

    "libnvidia-ml.so",              # Management library
    "libcuda.so",                   # CUDA driver library
    "libnvidia-ptxjitcompiler.so",  # PTX-SASS JIT compiler (used by libcuda)
    "libnvidia-fatbinaryloader.so", # fatbin loader (used by libcuda)
    "libnvidia-opencl.so",          # NVIDIA OpenCL ICD
    "libnvidia-compiler.so",        # NVVM-PTX compiler for OpenCL (used by libnvidia-opencl)
    #"libOpenCL.so",               # OpenCL ICD loader

    # ------ Video ------

    "libvdpau_nvidia.so",  # NVIDIA VDPAU ICD
    "libnvidia-encode.so", # Video encoder
    "libnvcuvid.so",       # Video decoder
    "libnvidia-fbc.so",    # Framebuffer capture
    "libnvidia-ifr.so",    # OpenGL framebuffer capture

    # ----- Graphic -----

    # XXX In an ideal world we would only mount nvidia_* vendor specific libraries and
    # install ICD loaders inside the container. However, for backward compatibility reason
    # we need to mount everything. This will hopefully change once GLVND is well established.

    "libGL.so",         # OpenGL/GLX legacy _or_ compatibility wrapper (GLVND)
    "libGLX.so",        # GLX ICD loader (GLVND)
    "libOpenGL.so",     # OpenGL ICD loader (GLVND)
    "libGLESv1_CM.so",  # OpenGL ES v1 common profile legacy _or_ ICD loader (GLVND)
    "libGLESv2.so",     # OpenGL ES v2 legacy _or_ ICD loader (GLVND)
    "libEGL.so",        # EGL ICD loader
    "libGLdispatch.so", # OpenGL dispatch (GLVND) (used by libOpenGL, libEGL and libGLES*)

    "libGLX_nvidia.so",         # OpenGL/GLX ICD (GLVND)
    "libEGL_nvidia.so",         # EGL ICD (GLVND)
    "libGLESv2_nvidia.so",      # OpenGL ES v2 ICD (GLVND)
    "libGLESv1_CM_nvidia.so",   # OpenGL ES v1 common profile ICD (GLVND)
    "libnvidia-eglcore.so",     # EGL core (used by libGLES* or libGLES*_nvidia and libEGL_nvidia)
    "libnvidia-egl-wayland.so", # EGL wayland extensions (used by libEGL_nvidia)
    "libnvidia-glcore.so",      # OpenGL core (used by libGL or libGLX_nvidia)
    "libnvidia-tls.so",         # Thread local storage (used by libGL or libGLX_nvidia)
    "libnvidia-glsi.so",        # OpenGL system interaction (used by libEGL_nvidia)
]

def _which(filename):
    for path in os.getenv("PATH").split(os.path.pathsep):
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path


# taken from stack overflow
# http://stackoverflow.com/questions/17195924/python-equivalent-of-unix-strings-utility
def _strings(filename):
    result = ""
    with open(filename, "rb") as f:
        for c in f.read():
            if c in string.printable:
                result += c
                continue
            if '.so' in result:
                yield result
            result = ""

# Get architecutre of real file path
def _get_arch(file_path):
    real_path = os.path.realpath(file_path)
    cmd = "file {}".format(real_path)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    if "32-bit" in stdout:
        return real_path, 32
    elif "64-bit" in stdout:
        return real_path, 64
    else:
        raise Exception("Unknown architecture for file: {}".format(real_path))

# map binaries
def get_bins():
    binlist = []
    for b in _binaries:
        binlist.append(_which(b))

    return binlist

# map headers
def get_headers():
    headerlist = []
    for h in _headers:
        headerlist.append(h)

    return headerlist

# map libs
def get_libs(cachefile='/etc/ld.so.cache'):
    # Accounting set for this function
    lib_set = {32:set(), 64:set()}
    # Lists returned to user
    lib_list = {32:[], 64:[]}
    link_list = {32:[], 64:[]}
    # Check all libraries in the machine's library cachefile
    for lib in _libraries:
        for libc in _strings(cachefile):
            if lib in libc and os.path.exists(libc):
                real_path, arch = _get_arch(libc)
                file_name = os.path.basename(libc)
                real_name = os.path.basename(real_path)
                if file_name not in lib_set[arch]:
                    lib_set[arch].add(file_name)
                    if os.path.islink(libc):
                        link_list[arch].append((real_name, file_name))
                    else:
                        lib_list[arch].append(real_path)
                if real_name not in lib_set[arch]:
                    lib_set[arch].add(real_name)
                    lib_list[arch].append(real_path)

    return lib_list, link_list
