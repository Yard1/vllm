"""
This is a special generate file that will be ran during setup to generate
AOT Triton kernels to be compiled into the wheel. The metadata of each kernel
is specified here as Python code.
"""

import argparse
import shutil
import sys
from pathlib import Path

from aot_utils.aot import (CompileConfiguration, Template,
                           compile_and_link_triton_kernel)


def _bool_to_bang(b: bool) -> str:
    return "!" if not b else ""


def main(kernel_path: Path, out_path: Path, dry_run: bool):
    if not dry_run:
        print(f"Deleting '{out_path}'...", file=sys.stderr)
        shutil.rmtree(str(Path(out_path).absolute()), ignore_errors=True)

    sample_kernel_templates = []
    seeded_uniform_kernel_templates = []
    num_warps = 32
    # block_size = model vocabulary size
    for block_size in [str(2**i) for i in range(13, 18)]:
        for modify_greedy_probs in (True, False):
            for save_logprobs in (True, False):
                for save_modified_probs in (True, False):
                    sample_kernel_templates += [
                        Template(
                            constexpr_values=[
                                block_size,
                                str(int(modify_greedy_probs)),
                                str(int(save_logprobs)),
                                str(int(save_modified_probs))
                            ],
                            num_warps=num_warps,
                            grid=("n_samples", "n_best", 1),
                            condition=(f"n_cols <= {block_size} && "
                                       f"{_bool_to_bang(modify_greedy_probs)}"
                                       "modify_greedy_probs && "
                                       f"{_bool_to_bang(save_logprobs)}"
                                       "save_logprobs && "
                                       f"{_bool_to_bang(save_modified_probs)}"
                                       "save_modified_probs"),
                        )
                    ]

    for block_size in [2**i for i in range(2, 18)]:
        philox_block_size = max(block_size // 4, 1)
        n_slices = block_size // philox_block_size
        seeded_uniform_kernel_templates += [
            Template(constexpr_values=[str(n_slices),
                                       str(philox_block_size)],
                     num_warps=num_warps,
                     grid=("n_rows", "n_3d", 1),
                     condition=f"n_cols <= {block_size}"),
        ]
    sample_kernel_configuration = CompileConfiguration(
        arg_ctypes=[
            "*i64", "*i64", "*fp32", "*fp32", "*fp32", "*fp32", "*i64",
            "*fp32", "i64", "i64", "i64", "i64", "i32", "i32", "i32"
        ],
        templates=sample_kernel_templates,
        condition_args=[
            "modify_greedy_probs", "save_logprobs", "save_modified_probs"
        ],
        condition_ctypes=["bool", "bool", "bool"],
    )

    compile_and_link_triton_kernel(path=kernel_path / "sample_triton.py",
                                   kernel_name="_sample_triton",
                                   out_name="sample_triton",
                                   configuration=sample_kernel_configuration,
                                   out_path=out_path,
                                   dry_run=dry_run)

    for dtype in ("fp32", "fp16", "bf16"):
        seeded_uniform_kernel_configurations = CompileConfiguration(
            arg_ctypes=[
                f"*{dtype}", "*i64", "i64", "i64", "i64", "i64", "i64", "i32"
            ],
            templates=seeded_uniform_kernel_templates,
        )
        compile_and_link_triton_kernel(
            path=kernel_path / "rand_triton.py",
            kernel_name="_seeded_uniform_triton",
            out_name=f"seeded_uniform_{dtype}",
            configuration=seeded_uniform_kernel_configurations,
            out_path=out_path,
            dry_run=dry_run)

    for dtype in ("fp16", "bf16"):
        penalty_kernel_configuration = CompileConfiguration(
            arg_ctypes=[
                f"*{dtype}", f"*{dtype}", f"*{dtype}", f"*{dtype}", "*i32",
                "*i32", "*i32", "i64", "i64"
            ],
            templates=[
                Template(
                    constexpr_values=[block_size],
                    num_warps=8,
                    grid=("n_rows", 1, 1),
                    condition=f"p_max_len_in_batch <= {block_size}",
                )
                # same block sizes as in the original Triton kernel
                # from LightLLM
                for block_size in [str(2**i) for i in range(9, 17)]
            ],
            condition_args=["p_max_len_in_batch"],
            condition_ctypes=["int64_t"],
        )
        compile_and_link_triton_kernel(
            path=kernel_path / "penalty_triton.py",
            kernel_name="_apply_penalty",
            out_name=f"apply_penalty_{dtype}",
            configuration=penalty_kernel_configuration,
            out_path=out_path,
            dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, )
    parser.add_argument(
        "--kernel-path",
        type=Path,
        required=True,
        help="Directory containing the Triton kernels (in Python files).")
    parser.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help=("Directory to write the C code to. "
              "WARNING: will delete all contents of the directory."))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the paths of output without compiling the code.")
    args = parser.parse_args()
    main(args.kernel_path, args.out_path, args.dry_run)
