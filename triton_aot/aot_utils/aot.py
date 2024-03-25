import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .compile import compile as _compile
from .link import link as _link


@dataclass
class Template:
    """
    Template for a kernel.

    Args:
        grid: Tuple of grid dimensions for kernel launch.
        condition: C code for the condition to use this template.
            Can use all arguments in the kernel + condition_args.
            Example: `n_cols <= 1024".
        constexpr_values: List of values for the constexpr arguments.
        num_warps: Number of warps to use.
        num_stages: Number of stages to use.
    """
    grid: Tuple[str, str, str]
    condition: str
    constexpr_values: List[str]
    num_warps: int = 1
    num_stages: int = 3


@dataclass
class CompileConfiguration:
    """
    Triton AOT compile configuration.

    Args:
        arg_ctypes: List of C types for the kernel arguments.
            Can be fp64, fp32, fp16, bf16, i32, i64, i1, i8, i16, u32, u64
            or value for constexpr.
            Prefix with * to indicate pointer.
            Suffix with `:16` to indicate divisibility by 16.
            Example: *fp32:16, i32:16, 1024, i32
        templates: List of templates to compile.
        condition_args: List of arguments to the condition.
            Condition arguments will not be used in the kernel
            itself but will be available in the entrypoint function
            to determine which specific kernel to launch,
        condition_ctypes: List of C types for the condition arguments
            (same as arg_ctypes).
    """
    arg_ctypes: List[str]
    templates: List[Template]
    condition_args: List[str] = field(default_factory=list)
    condition_ctypes: List[str] = field(default_factory=list)

    def __post_init__(self):
        assert all(
            len(self.templates[0].constexpr_values) == len(t.constexpr_values)
            for t in self.templates)


def compile_and_link_triton_kernel(
    path: Path,
    kernel_name: str,
    configuration: CompileConfiguration,
    out_name: str,
    out_path: Path,
    compute_capabilities: Optional[List[int]] = None,
    dry_run: bool = False,
) -> List[Path]:
    """
    AOT compile a templated Triton kernel. The kernel entrypoint will be
    `out_name` C function, which will pick the appropriate kernel
    based on the conditions set and compute capability.

    Args:
        path: Path to the python source file containing the kernel code.
        kernel_name: Name of the kernel function in path.
        configuration: Configuration of the kernel.
        out_name: Name of the entrypoint to compiled kernel.
        out_path: Path to the output directory.
        compute_capabilities: List of compute capabilities to compile for.
            Defaults to TRITON_CUDA_COMPUTE_CAPABILITIES env var.
        dry_run: If True, do not compile but print paths instead.
    """
    if compute_capabilities is None:
        compute_capabilities = os.getenv("TRITON_CUDA_COMPUTE_CAPABILITIES",
                                         "")
        if compute_capabilities:

            def parse_cc(cap: str) -> int:
                cap = cap.replace(".", "")
                return int(cap[0]) * 10 + int(cap[1])

            compute_capabilities = [
                parse_cc(cap) for cap in compute_capabilities.split(",")
            ]
    created_headers = []
    Path(out_path).absolute().mkdir(parents=True, exist_ok=True)
    for template in configuration.templates:
        header, _ = _compile(
            path=path,
            kernel_name=kernel_name,
            signature=configuration.arg_ctypes + template.constexpr_values,
            grid=template.grid,
            out_path=Path(out_path).absolute() / kernel_name.lstrip("_"),
            out_name=out_name,
            num_warps=template.num_warps,
            num_stages=template.num_stages,
            compute_capabilities=compute_capabilities,
            dry_run=dry_run,
        )
        created_headers.append(header)
    return _link(
        created_headers,
        Path(out_path).absolute() / out_name,
        configuration.condition_args,
        configuration.condition_ctypes,
        [template.condition for template in configuration.templates],
        dry_run=dry_run,
    )
