#!/usr/bin/env python3
"""
OPS Challenge - Solution Verifier

Verify src/solution.py correctness, execution time, and max error.

Usage:
    python verify.py
    python verify.py --input_path testcase/data.parquet --ref_ans_path testcase/answer.npy
"""

import os
import sys
import time
import argparse
import importlib.util
import numpy as np
from loguru import logger


if __name__ == "__main__":
    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

    parser = argparse.ArgumentParser(
        description="Verify OPS Challenge solution",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--solution_file",
        default="src/solution.py",
        help="Path to solution file (default: src/solution.py)"
    )
    parser.add_argument(
        "--entry_point",
        default="ops_rolling_rank",
        help="Entry point function name (default: ops_rolling_rank)"
    )
    parser.add_argument(
        "--input_path",
        default="testcase/data_for_rolling_rank.parquet",
        help="Path to input parquet file (default: testcase/data_for_rolling_rank.parquet)"
    )
    parser.add_argument(
        "--ref_ans_path",
        default="testcase/rolling_rank_dense_v1.npy",
        help="Path to reference answer npy file (default: testcase/rolling_rank_dense_v1.npy)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Window size parameter (default: 20)"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance (default: 1e-5)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance (default: 1e-8)"
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=20,
        help="Maximum number of threads (default: 20)"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("OPS Challenge - Solution Verifier")
    logger.info("=" * 70)

    # Check files
    logger.info("")
    logger.info("[1/5] Checking files...")
    for path, name in [(args.solution_file, "Solution"),
                       (args.input_path, "Input"),
                       (args.ref_ans_path, "Reference")]:
        if not os.path.exists(path):
            logger.error(f"ERROR: {name} file not found: {path}")
            sys.exit(1)
        logger.info(f"  OK: {path}")

    # Load solution
    logger.info("")
    logger.info(f"[2/5] Loading solution function: {args.entry_point}()...")
    try:
        spec = importlib.util.spec_from_file_location("solution", args.solution_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solution_func = getattr(module, args.entry_point)
        logger.info(f"  OK: Function loaded")
    except Exception as e:
        logger.error(f"ERROR: Failed to load solution: {e}")
        sys.exit(1)

    # Load reference answer
    logger.info("")
    logger.info(f"[3/5] Loading reference answer...")
    try:
        expected = np.load(args.ref_ans_path, allow_pickle=False)
        logger.info(f"  OK: shape={expected.shape}, dtype={expected.dtype}")
    except Exception as e:
        logger.error(f"ERROR: Failed to load reference: {e}")
        sys.exit(1)

    # Execute solution
    logger.info("")
    logger.info(f"[4/5] Executing solution (window={args.window})...")
    try:
        t0 = time.perf_counter()
        output = solution_func(input_path=args.input_path, window=args.window)
        elapsed = time.perf_counter() - t0
        logger.info(f"  OK: Execution completed")
        logger.info(f"  Time: {elapsed:.4f} seconds")
    except Exception as e:
        logger.error(f"ERROR: Runtime error: {e}")
        sys.exit(1)

    # Verify output
    logger.info("")
    logger.info(f"[5/5] Verifying output...")

    if not isinstance(output, np.ndarray):
        logger.error(f"ERROR: Output type mismatch")
        logger.error(f"  Expected: numpy.ndarray")
        logger.error(f"  Got: {type(output)}")
        sys.exit(1)

    if output.shape != expected.shape:
        logger.error(f"ERROR: Shape mismatch")
        logger.error(f"  Expected: {expected.shape}")
        logger.error(f"  Got: {output.shape}")
        sys.exit(1)

    if output.dtype != expected.dtype:
        logger.error(f"ERROR: Dtype mismatch")
        logger.error(f"  Expected: {expected.dtype}")
        logger.error(f"  Got: {output.dtype}")
        sys.exit(1)

    logger.info(f"  OK: dtype={output.dtype}, shape={output.shape}")

    # Calculate max diff
    diff = np.abs(output - expected)
    mask = np.isnan(output) & np.isnan(expected)
    diff[mask] = 0
    max_diff = float(np.nanmax(diff))

    logger.info(f"  Max diff: {max_diff}")

    # Check correctness
    is_correct = np.allclose(output, expected, rtol=args.rtol, atol=args.atol, equal_nan=True)

    logger.info("")
    logger.info("=" * 70)
    if is_correct:
        logger.success("RESULT: PASS")
        logger.info(f"  Time: {elapsed:.4f} seconds")
        logger.info(f"  Max diff: {max_diff}")
        logger.info(f"  Tolerance: rtol={args.rtol}, atol={args.atol}")
        logger.info("=" * 70)
        sys.exit(0)
    else:
        logger.error("RESULT: FAIL")
        logger.error(f"  Max diff: {max_diff} (exceeds tolerance)")
        logger.error(f"  Tolerance: rtol={args.rtol}, atol={args.atol}")
        logger.info("=" * 70)
        sys.exit(1)


