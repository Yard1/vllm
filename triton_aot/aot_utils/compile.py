# ruff: noqa

# Based on https://github.com/openai/triton/commit/a767ca41e189988740d35cbb9aecd873c4874a62

# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import binascii
import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import triton
from triton.compiler.code_generator import kernel_suffix
from triton.compiler.compiler import get_architecture_descriptor
from triton.compiler.make_launcher import ty_to_cpp

desc = """
Triton ahead-of-time compiler:

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C source-code that embeds the `cubin`
data along with utilities to load, unload and launch the kernel.

signature is provided as a list of (optionally divisibility-hinted) types
or constexpr values, e.g.

`compile.py --kernel-name kernel --signature "*fp32:16, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that argument 0, 1 are assumed to be multiple of 16,
and argument 2 is assumed to be a compile-time constant of value 1024, i.e. it won't be part of the generated prototype.

The resulting entry point will have signature

CUresult kernel_{specialization_suffix}(CUstream stream, unsigned gX, unsigned gY, unsigned gZ, float* arg0, int32_t arg1, int32_t arg2)

Different such specialized entry points can be combined using the `linker.py` script.

NOTE: when resolving the scope of /path/to/kernel.py, the file will be executed from within its parent directory with the python interpreter
used to run this `compile.py` script
"""

CUBIN_TEMPLATE = """#define CUBIN_NAME_{cc} {kernel_name}_{cc}_cubin
CUmodule {kernel_name}_{cc}_mod = NULL;
CUfunction {kernel_name}_{cc}_func = NULL;
unsigned char CUBIN_NAME_{cc}[{bin_size}] = {{ {bin_data} }};
"""

UNLOAD_CASE = """      case {cc}:
        if ({kernel_name}_{cc}_func != NULL) {{
            CUDA_CHECK(cuModuleUnload({kernel_name}_{cc}_mod));
            {kernel_name}_{cc}_func = NULL;
        }}
        break;"""

LOAD_CASE = """      case {cc}:
        if ({kernel_name}_{cc}_func != NULL)
            break;
        bin = (void *)&CUBIN_NAME_{cc};
        CUDA_CHECK(cuModuleLoadData(&{kernel_name}_{cc}_mod, bin));
        CUDA_CHECK(cuModuleGetFunction(&{kernel_name}_{cc}_func, {kernel_name}_{cc}_mod, "{triton_kernel_name}"));
        // set dynamic shared memory if necessary
        CUDA_CHECK(cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
        if ({shared} > 49152 && shared_optin > 49152) {{
          CUDA_CHECK(cuFuncSetCacheConfig({kernel_name}_{cc}_func, CU_FUNC_CACHE_PREFER_SHARED));
          CUDA_CHECK(cuFuncSetAttribute({kernel_name}_{cc}_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin))
        }}
        break;"""

LAUNCH_CASE = """        case {cc}:
          load_{kernel_name}();
          result = cuLaunchKernel({kernel_name}_{cc}_func, gX, gY, gZ, {num_warps} * 32, 1, 1, {shared}, stream, args, NULL);
          break;"""


def compile(path,
            kernel_name: str,
            out_name: str,
            out_path: Path,
            signature: tuple,
            grid: tuple,
            num_stages: int = 3,
            num_warps: int = 1,
            compute_capabilities: Optional[List[int]] = None,
            dry_run: bool = False) -> Tuple[Path, Path]:
    out_name = out_name if out_name else kernel_name
    out_path = out_path if out_path else Path(out_name)

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(path)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, kernel_name)
    assert len(grid) == 3

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), signature))

    def hash_signature(signature: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(signature).encode())
        return m.hexdigest()[:8]

    meta_sig = f"warps{num_warps}xstages{num_stages}"
    sig_hash = hash_signature(signature + [meta_sig])

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        return None

    hints = {
        i: constexpr(s.split(":")[1])
        for i, s in enumerate(signature) if ":" in s
    }
    hints = {k: v for k, v in hints.items() if v is not None}
    constexprs = {i: constexpr(s) for i, s in enumerate(signature)}
    constexprs = {k: v for k, v in constexprs.items() if v is not None}
    signature = {
        i: s.split(":")[0]
        for i, s in enumerate(signature) if i not in constexprs
    }
    const_sig = "x".join([str(v) for v in constexprs.values()])
    doc_string = [
        f"{kernel.arg_names[i]}={constexprs[i]}" for i in constexprs.keys()
    ]
    doc_string += [f"num_warps={num_warps}", f"num_stages={num_stages}"]

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    divisible_by_16 = [i for i, h in hints.items() if h == 16]
    equal_to_1 = [i for i, h in hints.items() if h == 1]
    config = triton.compiler.instance_descriptor(
        divisible_by_16=divisible_by_16, equal_to_1=equal_to_1)
    for i in equal_to_1:
        constexprs.update({i: 1})
    compute_capabilities = compute_capabilities or [
        get_architecture_descriptor(None)
    ]
    ccinfo_by_compute_capability = []
    hex_by_compute_capability = []
    for compute_capability in compute_capabilities:
        ccinfo = triton.compile(kernel,
                                signature=signature,
                                constants=constexprs,
                                configs=[config],
                                num_warps=num_warps,
                                num_stages=num_stages,
                                cc=compute_capability)
        ccinfo_by_compute_capability.append(ccinfo)
        hex_by_compute_capability.append(
            str(binascii.hexlify(ccinfo.asm["cubin"]))[2:-1])
    arg_names = []
    arg_types = []
    for i in signature.keys():
        if i not in equal_to_1:
            arg_names += [kernel.arg_names[i]]
            arg_types += [signature[i]]

    # dump C stub code
    suffix = kernel_suffix(signature.values(), config)
    func_name = "_".join([out_name, sig_hash, suffix])
    triton_kernel_name = "_".join([kernel_name, suffix])

    cubins = "\n".join(
        CUBIN_TEMPLATE.format(
            cc=cc,
            kernel_name=func_name,
            bin_size=len(hex_),
            bin_data=", ".join(
                [f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        ) for cc, hex_ in zip(compute_capabilities, hex_by_compute_capability))

    unload_switch = "\n".join(
        UNLOAD_CASE.format(cc=cc, kernel_name=func_name)
        for cc in compute_capabilities)

    load_switch = "\n".join(
        LOAD_CASE.format(cc=cc,
                         kernel_name=func_name,
                         shared=ccinfo.shared,
                         triton_kernel_name=triton_kernel_name) for cc, ccinfo
        in zip(compute_capabilities, ccinfo_by_compute_capability))

    launch_switch = "\n".join(
        LAUNCH_CASE.format(cc=cc,
                           kernel_name=func_name,
                           shared=ccinfo.shared,
                           num_warps=num_warps) for cc, ccinfo in
        zip(compute_capabilities, ccinfo_by_compute_capability))

    params = {
        "kernel_name":
        func_name,
        "triton_kernel_name":
        triton_kernel_name,
        "cubins":
        cubins,
        "unload_switch":
        unload_switch,
        "load_switch":
        load_switch,
        "launch_switch":
        launch_switch,
        "signature":
        ", ".join([
            f"{ty_to_cpp(ty)} {name}"
            for name, ty in zip(arg_names, arg_types)
        ]),
        "full_signature":
        ", ".join([
            f"{ty_to_cpp(signature[i])} {kernel.arg_names[i]}"
            for i in signature.keys()
        ]),
        "arg_pointers":
        ", ".join([f"&{arg}" for arg in arg_names]),
        "num_args":
        len(arg_names),
        "kernel_docstring":
        doc_string,
        "algo_info":
        "_".join([const_sig, meta_sig]),
        "gridX":
        grid[0],
        "gridY":
        grid[1],
        "gridZ":
        grid[2],
        "_placeholder":
        "",
    }
    paths = []
    for ext in ["h", "c"]:
        template_path = Path(__file__).parent / f"compile.{ext}"
        out_path_with_suffix = out_path.with_suffix(
            f".{sig_hash}_{suffix}.{ext}")
        if dry_run:
            print(out_path_with_suffix)
        else:
            with out_path_with_suffix.open("w") as fp:
                fp.write(Path(template_path).read_text().format(**params))
        paths.append(out_path_with_suffix)
        utils_path = Path(__file__).parent / f"utils.{ext}"
        if dry_run:
            print(out_path.parent / utils_path.name)
        else:
            try:
                shutil.copyfile(utils_path, out_path.parent / utils_path.name)
            except shutil.SameFileError:
                pass
    return tuple(paths)
