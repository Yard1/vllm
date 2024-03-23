# Triton AOT

This module contains code necessary for Triton AOT compilation. Compared to JIT, AOT allows us to remove CPU overhead. It is especially useful for small kernels that are called many times. Any Triton JIT kernel can be compiled AOT.

This module contains several files vendored from upstream Triton (with modifications). It is not expected that those files will change much upstream or need to be updated here. Even if this module doesn't get upstreamed to vllm-project/vllm, the actual Triton kernel code will stay the same (only compiled JIT instead of AOT).

## Overview

Triton AOT compilation works by generating C files with inlined binary blobs. This C code can then be compiled through setuptools/torch extensions during wheel build.

Kernels will be automatically generated for different compute capabilities. You can also use templating to generate multiple kernels for different data types, constant expressions, block sizes etc. You can specify conditions as to which kernel will be used for given arguments.

Note that the generated kernels do not take in Torch Tensors as inputs. You need to separately create C shims that take in Torch Tensors and turn them into pointers (see `vllm/csrc/triton_aot`).

## Module structure

* `generate.py` is an executable script intended to be ran before wheel build. It will generate specified kernels. In order to add new kernels to be generated, that file has to be modified.
* `kernels` folder contains python Triton code for each kernel. These files are used by `generate.py` to generate C code.
* `aot_utils` folder contains:
    - `aot.py` which is an entrypoint for AOT logic,
    - `compile.py` (vendored from upstream Triton and modified) which reads Python files, extracts Triton code, compiles it, and writes it to C source files,
    - `link.py` (vendored from upstream Triton and modified) which links multiple kernels into single entrypoints (eg. different templates for a given kernel for different data types, constant expressions etc.),
    - `.c` and `.h` files which are templates used by `compile.py` and `link.py`.

## Adding new kernels

In order to add new kernels, one has to:
1. Add Triton code in a python file in `kernels` folder.
2. Modify `generate.py` to add the new kernel with templates (see docstrings of `CompileConfiguration` and `Template` classes).
3. Add C shim that takes in Torch Tensors and turns them into pointers (see `vllm/csrc/triton_aot`).
