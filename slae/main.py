from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from time import perf_counter

from methods import AVAILABLE_METHODS
from solver import LinearSystemSolver


def _format_solution(values: List[float], reference: np.ndarray | None = None, elapsed: float | None = None) -> None:
    if elapsed is not None:
        print(f"    время: {elapsed * 1e3:.3f} мс")
    for idx, value in enumerate(values):
        line = f"    x[{idx}] = {value:.10f}"
        if reference is not None:
            delta = abs(value - reference[idx])
            line += f"    Δ={delta:.3e}"
        print(line)


def _run_case(label: str, system: Tuple[List[List[float]], List[float]], plan: List[Tuple[str, Dict]]) -> None:
    A, b = system
    solver = LinearSystemSolver(A, b)
    start_numpy = perf_counter()
    baseline = np.linalg.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))
    numpy_time = perf_counter() - start_numpy
    print(f"\n=== {label.upper()} ===")
    print("NumPy solve:")
    _format_solution(list(map(float, baseline)), elapsed=numpy_time)
    for method, params in plan:
        title = AVAILABLE_METHODS.get(method, method.upper())
        try:
            start = perf_counter()
            values = solver.solve(method=method, **params)
            elapsed = perf_counter() - start
            print(f"{title}:")
            _format_solution(values, baseline, elapsed)
        except Exception as error:
            print(f"{title}: ошибка: {error}")


def main() -> None:
    cases: Dict[str, Dict] = {
        "sym_spd_3x3": {
            "system": (
                [[4.0, 2.0, -1.0], [2.0, 5.0, 1.0], [-1.0, 1.0, 3.0]],
                [5.0, 12.0, -1.0],
            ),
            "methods": [
                ("lu", {}),
                ("simple", {"tolerance": 1e-9, "max_iterations": 20_000}),
                ("jacobi", {"tolerance": 1e-9, "max_iterations": 20_000}),
                ("seidel", {"tolerance": 1e-9, "max_iterations": 20_000}),
                ("steepest", {"tolerance": 1e-9, "max_iterations": 5_000}),
                ("cg", {"tolerance": 1e-9, "max_iterations": 5_000}),
            ],
        },
        "diag_dom_3x3": {
            "system": (
                [[10.0, 2.0, 1.0], [2.0, 8.0, 1.0], [1.0, 1.0, 5.0]],
                [7.0, 6.0, 5.0],
            ),
            "methods": [
                ("lu", {}),
                ("simple", {"tolerance": 1e-9, "max_iterations": 15_000}),
                ("jacobi", {"tolerance": 1e-9, "max_iterations": 15_000}),
                ("seidel", {"tolerance": 1e-9, "max_iterations": 15_000}),
                ("steepest", {"tolerance": 1e-9, "max_iterations": 5_000}),
                ("cg", {"tolerance": 1e-9, "max_iterations": 5_000}),
            ],
        },
        "spd_4x4": {
            "system": (
                [[6.0, 2.0, 1.0, 0.5], [2.0, 7.0, 2.0, 1.0], [1.0, 2.0, 5.0, 1.5], [0.5, 1.0, 1.5, 4.0]],
                [9.0, 8.0, 6.0, 5.0],
            ),
            "methods": [
                ("lu", {}),
                ("simple", {"tolerance": 1e-9, "max_iterations": 25_000}),
                ("jacobi", {"tolerance": 1e-9, "max_iterations": 25_000}),
                ("seidel", {"tolerance": 1e-9, "max_iterations": 25_000}),
                ("steepest", {"tolerance": 1e-9, "max_iterations": 5_000}),
                ("cg", {"tolerance": 1e-9, "max_iterations": 5_000}),
            ],
        },
        "spd_5x5": {
            "system": (
                [
                    [9.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 8.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 7.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 6.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 5.0],
                ],
                [5.0, 4.0, 3.0, 2.0, 1.0],
            ),
            "methods": [
                ("lu", {}),
                ("simple", {"tolerance": 1e-9, "max_iterations": 30_000}),
                ("jacobi", {"tolerance": 1e-9, "max_iterations": 30_000}),
                ("seidel", {"tolerance": 1e-9, "max_iterations": 30_000}),
                ("steepest", {"tolerance": 1e-9, "max_iterations": 10_000}),
                ("cg", {"tolerance": 1e-9, "max_iterations": 10_000}),
            ],
        },
    }

    for label, data in cases.items():
        _run_case(label, data["system"], data["methods"])


if __name__ == "__main__":
    main()
