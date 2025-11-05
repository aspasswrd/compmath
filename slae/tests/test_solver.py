import math
import unittest

import numpy as np

from solver import LinearSystemSolver


def make_example_system():
    """Возвращает SPD систему с заранее известным решением для тестов."""
    A = [
        [4.0, 2.0, -1.0],
        [2.0, 5.0, 1.0],
        [-1.0, 1.0, 3.0],
    ]
    b = [5.0, 12.0, -1.0]
    return A, b


def assert_vectors_close(testcase: unittest.TestCase, expected, actual, tol=1e-7) -> None:
    for idx, (exp, act) in enumerate(zip(expected, actual)):
        testcase.assertTrue(
            math.isclose(exp, act, rel_tol=tol, abs_tol=tol),
            msg=f"Координата x[{idx}] отличается: ожидалось {exp}, получено {act}",
        )


class LinearSystemSolverTestCase(unittest.TestCase):
    def test_lu_solution_matches_expected(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        expected = (-0.6, 2.942857142857143, -1.5142857142857142)
        actual = solver.solve(method="lu")
        assert_vectors_close(self, expected, actual, tol=1e-9)

    def test_simple_iterations_matches_expected(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        expected = (-0.6, 2.942857142857143, -1.5142857142857142)
        actual = solver.solve(
            method="simple",
            tolerance=1e-9,
            max_iterations=20_000,
        )
        assert_vectors_close(self, expected, actual, tol=1e-7)

    def test_jacobi_matches_expected(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        expected = (-0.6, 2.942857142857143, -1.5142857142857142)
        actual = solver.solve(
            method="jacobi",
            tolerance=1e-9,
            max_iterations=20_000,
        )
        assert_vectors_close(self, expected, actual, tol=1e-7)

    def test_gauss_seidel_matches_expected(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        expected = (-0.6, 2.942857142857143, -1.5142857142857142)
        actual = solver.solve(
            method="seidel",
            tolerance=1e-9,
            max_iterations=10_000,
        )
        assert_vectors_close(self, expected, actual, tol=1e-9)

    def test_steepest_descent_matches_expected(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        expected = (-0.6, 2.942857142857143, -1.5142857142857142)
        actual = solver.solve(
            method="steepest",
            tolerance=1e-9,
            max_iterations=10_000,
        )
        assert_vectors_close(self, expected, actual, tol=1e-7)

    def test_conjugate_gradients_matches_expected(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        expected = (-0.6, 2.942857142857143, -1.5142857142857142)
        actual = solver.solve(
            method="cg",
            tolerance=1e-9,
            max_iterations=10_000,
        )
        assert_vectors_close(self, expected, actual, tol=1e-9)

    def test_constructor_matrix_not_2d(self) -> None:
        with self.assertRaises(ValueError):
            LinearSystemSolver([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    def test_constructor_rhs_not_1d(self) -> None:
        with self.assertRaises(ValueError):
            LinearSystemSolver([[1.0, 0.0], [0.0, 1.0]], [[1.0], [2.0]])

    def test_constructor_empty_matrix(self) -> None:
        with self.assertRaises(ValueError):
            LinearSystemSolver([[]], [0.0])

    def test_constructor_non_square_matrix(self) -> None:
        with self.assertRaises(ValueError):
            LinearSystemSolver([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0, 2.0])

    def test_constructor_rejects_incorrect_rhs_size(self) -> None:
        with self.assertRaises(ValueError):
            LinearSystemSolver([[1.0, 2.0], [3.0, 4.0]], [1.0])

    def test_lu_raises_when_pivot_is_zero(self) -> None:
        solver = LinearSystemSolver([[0.0, 1.0], [0.0, 1.0]], [1.0, 1.0])
        with self.assertRaises(ZeroDivisionError):
            solver.solve(method="lu")

    def test_spd_methods_raise_on_non_symmetric_matrix(self) -> None:
        solver = LinearSystemSolver([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0])
        with self.assertRaises(ValueError):
            solver.solve(method="steepest")
        with self.assertRaises(ValueError):
            solver.solve(method="cg")

    def test_spd_methods_raise_on_non_positive_definite_matrix(self) -> None:
        solver = LinearSystemSolver([[0.0, 0.0], [0.0, 1.0]], [0.0, 1.0])
        with self.assertRaises(ValueError):
            solver.solve(method="steepest")
        with self.assertRaises(ValueError):
            solver.solve(method="cg")

    def test_simple_iterations_negative_tau_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="simple", tau=-0.5)

    def test_simple_iterations_non_positive_tolerance_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="simple", tolerance=0.0)

    def test_simple_iterations_non_positive_max_iterations_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="simple", max_iterations=0)

    def test_simple_iterations_runtime_error(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(RuntimeError):
            solver.solve(method="simple", tolerance=1e-20, max_iterations=1)

    def test_default_tau_zero_matrix_raises(self) -> None:
        solver = LinearSystemSolver([[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0])
        with self.assertRaises(ValueError):
            solver.solve(method="simple")

    def test_jacobi_invalid_tolerance_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="jacobi", tolerance=0.0)

    def test_jacobi_non_positive_max_iterations_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="jacobi", max_iterations=0)

    def test_jacobi_zero_diagonal_raises(self) -> None:
        solver = LinearSystemSolver([[0.0, 1.0], [1.0, 2.0]], [1.0, 1.0])
        with self.assertRaises(ZeroDivisionError):
            solver.solve(method="jacobi")

    def test_jacobi_runtime_error(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(RuntimeError):
            solver.solve(method="jacobi", tolerance=1e-20, max_iterations=1)

    def test_gauss_seidel_non_positive_tolerance_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="seidel", tolerance=0.0)

    def test_gauss_seidel_non_positive_max_iterations_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="seidel", max_iterations=0)

    def test_gauss_seidel_zero_diagonal_raises(self) -> None:
        solver = LinearSystemSolver([[0.0, 1.0], [1.0, 2.0]], [1.0, 1.0])
        with self.assertRaises(ZeroDivisionError):
            solver.solve(method="seidel")

    def test_gauss_seidel_runtime_error(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(RuntimeError):
            solver.solve(method="seidel", tolerance=1e-20, max_iterations=1)

    def test_steepest_descent_non_positive_tolerance_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="steepest", tolerance=0.0)

    def test_steepest_descent_non_positive_max_iterations_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="steepest", max_iterations=0)

    def test_steepest_descent_zero_denominator_raises(self) -> None:
        solver = LinearSystemSolver([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
        solver._ensure_spd = lambda: None  # type: ignore[method-assign]
        solver._A[:] = 0.0
        with self.assertRaises(ZeroDivisionError):
            solver.solve(method="steepest", tolerance=1e-9, max_iterations=5)

    def test_steepest_descent_runtime_error(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(RuntimeError):
            solver.solve(method="steepest", tolerance=1e-20, max_iterations=1)

    def test_conjugate_gradients_non_positive_tolerance_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="cg", tolerance=0.0)

    def test_conjugate_gradients_non_positive_max_iterations_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(ValueError):
            solver.solve(method="cg", max_iterations=0)

    def test_conjugate_gradients_zero_residual_returns_zero(self) -> None:
        solver = LinearSystemSolver([[4.0, 1.0], [1.0, 3.0]], [0.0, 0.0])
        result = solver.solve(method="cg")
        assert_vectors_close(self, (0.0, 0.0), result, tol=1e-12)

    def test_conjugate_gradients_zero_denominator_raises(self) -> None:
        solver = LinearSystemSolver([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
        solver._ensure_spd = lambda: None  # type: ignore[method-assign]
        solver._A[:] = 0.0
        with self.assertRaises(ZeroDivisionError):
            solver.solve(method="cg", tolerance=1e-9, max_iterations=5)

    def test_conjugate_gradients_runtime_error(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(RuntimeError):
            solver.solve(method="cg", tolerance=1e-20, max_iterations=1)

    def test_unknown_method_raises(self) -> None:
        A, b = make_example_system()
        solver = LinearSystemSolver(A, b)
        with self.assertRaises(NotImplementedError):
            solver.solve(method="unsupported")

    def test_back_substitution_zero_pivot_raises(self) -> None:
        solver = LinearSystemSolver([[1.0]], [1.0])
        with self.assertRaises(ZeroDivisionError):
            solver._back_substitution(np.array([[0.0]]), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
