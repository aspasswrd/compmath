from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np


class LinearSystemSolver:
    """Численно решает системы линейных уравнений Ax = b."""

    def __init__(self, coefficients: Sequence[Sequence[float]], rhs: Sequence[float]) -> None:
        matrix = np.asarray(coefficients, dtype=float)
        vector = np.asarray(rhs, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Матрица коэффициентов должна быть двумерной.")
        if vector.ndim != 1:
            raise ValueError("Вектор правых частей должен быть одномерным.")

        n_rows, n_cols = matrix.shape
        if n_rows == 0 or n_cols == 0:
            raise ValueError("Матрица коэффициентов должна быть непустой.")
        if n_rows != n_cols:
            raise ValueError("Матрица коэффициентов должна быть квадратной.")
        if vector.size != n_rows:
            raise ValueError("Размер вектора правых частей должен совпадать с размером матрицы.")

        self._n = n_rows
        self._A = matrix.copy()
        self._b = vector.reshape(n_rows).copy()

    def solve(self, method: str = "lu", **kwargs: Any) -> List[float]:
        """Решает Ax = b выбранным методом."""
        method = method.lower()
        dispatch = {
            "lu": self.solve_lu,
            "simple": self.solve_simple_iterations,
            "simple_iteration": self.solve_simple_iterations,
            "simple_iterations": self.solve_simple_iterations,
            "jacobi": self.solve_jacobi,
            "seidel": self.solve_gauss_seidel,
            "gauss-seidel": self.solve_gauss_seidel,
            "steepest": self.solve_steepest_descent,
            "steepest_descent": self.solve_steepest_descent,
            "cg": self.solve_conjugate_gradients,
            "conjugate_gradients": self.solve_conjugate_gradients,
        }

        if method not in dispatch:
            raise NotImplementedError(f"Метод '{method}' пока не реализован.")

        result = dispatch[method](**kwargs)
        return list(map(float, result))

    def solve_lu(self) -> np.ndarray:
        """Решает систему через LU-разложение."""
        L, U = self._lu_decomposition()
        y = self._forward_substitution(L, self._b)
        x = self._back_substitution(U, y)
        return x

    def solve_simple_iterations(
        self,
        tau: float | None = None,
        tolerance: float = 1e-9,
        max_iterations: int = 10_000,
    ) -> np.ndarray:
        """Решает систему методом простых итераций."""
        if tolerance <= 0:
            raise ValueError("Точность должна быть положительной.")
        if max_iterations <= 0:
            raise ValueError("Число итераций должно быть положительным.")

        if tau is None:
            tau = self._default_tau()
        elif tau <= 0:
            raise ValueError("Параметр tau должен быть положительным.")

        x = np.zeros(self._n)
        for _ in range(max_iterations):
            residual = self._residual(x)
            x_new = x + tau * residual
            if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
                return x_new
            x = x_new

        raise RuntimeError("Метод простых итераций не сошелся за заданное число шагов.")

    def solve_jacobi(
        self,
        tolerance: float = 1e-9,
        max_iterations: int = 10_000,
    ) -> np.ndarray:
        """Метод Якоби (иттерационное диагональное расщепление)."""
        if tolerance <= 0:
            raise ValueError("Точность должна быть положительной.")
        if max_iterations <= 0:
            raise ValueError("Число итераций должно быть положительным.")

        D = np.diag(self._A)
        if np.any(np.isclose(D, 0.0)):
            raise ZeroDivisionError("На диагонали найден нулевой элемент, метод Якоби неприменим.")

        R = self._A - np.diagflat(D)
        x = np.zeros(self._n)

        for _ in range(max_iterations):
            x_new = (self._b - R @ x) / D
            if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
                return x_new
            x = x_new

        raise RuntimeError("Метод Якоби не сошелся за заданное число шагов.")

    def solve_gauss_seidel(
        self,
        tolerance: float = 1e-9,
        max_iterations: int = 10_000,
    ) -> np.ndarray:
        """Метод Зейделя (Гаусса-Зейделя)."""
        if tolerance <= 0:
            raise ValueError("Точность должна быть положительной.")
        if max_iterations <= 0:
            raise ValueError("Число итераций должно быть положительным.")

        if np.any(np.isclose(np.diag(self._A), 0.0)):
            raise ZeroDivisionError("На диагонали найден нулевой элемент, метод Зейделя неприменим.")

        x = np.zeros(self._n)
        for _ in range(max_iterations):
            x_old = x.copy()
            for i in range(self._n):
                sigma1 = np.dot(self._A[i, :i], x[:i])
                sigma2 = np.dot(self._A[i, i + 1 :], x_old[i + 1 :])
                x[i] = (self._b[i] - sigma1 - sigma2) / self._A[i, i]

            if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
                return x

        raise RuntimeError("Метод Зейделя не сошелся за заданное число шагов.")

    def solve_steepest_descent(
        self,
        tolerance: float = 1e-9,
        max_iterations: int = 10_000,
    ) -> np.ndarray:
        """Метод наискорейшего спуска для SPD-матриц."""
        self._ensure_spd()
        if tolerance <= 0:
            raise ValueError("Точность должна быть положительной.")
        if max_iterations <= 0:
            raise ValueError("Число итераций должно быть положительным.")

        x = np.zeros(self._n)
        r = self._residual(x)

        for _ in range(max_iterations):
            if np.linalg.norm(r, ord=np.inf) < tolerance:
                return x
            Ar = self._A @ r
            denom = float(np.dot(r, Ar))
            if abs(denom) < 1e-18:
                raise ZeroDivisionError("Деление на ноль при вычислении шага метода спуска.")
            alpha = float(np.dot(r, r) / denom)
            x = x + alpha * r
            r = self._residual(x)

        raise RuntimeError("Метод наискорейшего спуска не сошелся за заданное число шагов.")

    def solve_conjugate_gradients(
        self,
        tolerance: float = 1e-9,
        max_iterations: int = 10_000,
    ) -> np.ndarray:
        """Метод сопряженных градиентов для SPD-матриц."""
        self._ensure_spd()
        if tolerance <= 0:
            raise ValueError("Точность должна быть положительной.")
        if max_iterations <= 0:
            raise ValueError("Число итераций должно быть положительным.")

        x = np.zeros(self._n)
        r = self._residual(x)
        if np.linalg.norm(r, ord=np.inf) < tolerance:
            return x
        p = r.copy()
        rs_old = float(np.dot(r, r))

        for _ in range(max_iterations):
            Ap = self._A @ p
            denom = float(np.dot(p, Ap))
            if abs(denom) < 1e-18:
                raise ZeroDivisionError("Деление на ноль при вычислении шага метода CG.")
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = float(np.dot(r, r))
            if np.sqrt(rs_new) < tolerance:
                return x
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        raise RuntimeError("Метод сопряженных градиентов не сошелся за заданное число шагов.")

    def _lu_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self._n
        A = self._A
        L = np.zeros((n, n))
        U = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                upper_sum = sum(L[i, k] * U[k, j] for k in range(i))
                U[i, j] = A[i, j] - upper_sum

            if abs(U[i, i]) < 1e-12:
                raise ZeroDivisionError(
                    "В LU-разложении встретился нулевой главный элемент. "
                    "Попробуйте использовать другой метод или добавить выбор главного элемента."
                )

            for j in range(i, n):
                if i == j:
                    L[i, i] = 1.0
                else:
                    lower_sum = sum(L[j, k] * U[k, i] for k in range(i))
                    L[j, i] = (A[j, i] - lower_sum) / U[i, i]

        return L, U

    def _forward_substitution(self, lower: np.ndarray, vector: np.ndarray) -> np.ndarray:
        n = self._n
        y = np.zeros(n)
        for i in range(n):
            partial = float(np.dot(lower[i, :i], y[:i]))
            y[i] = vector[i] - partial
        return y

    def _back_substitution(self, upper: np.ndarray, vector: np.ndarray) -> np.ndarray:
        n = self._n
        x = np.zeros(n)
        for i in reversed(range(n)):
            partial = float(np.dot(upper[i, i + 1 :], x[i + 1 :]))
            pivot = upper[i, i]
            if abs(pivot) < 1e-12:
                raise ZeroDivisionError("В обратном ходе встретился нулевой главный элемент.")
            x[i] = (vector[i] - partial) / pivot
        return x

    def _residual(self, x: np.ndarray) -> np.ndarray:
        return self._b - self._A @ x

    def _default_tau(self) -> float:
        norm = np.linalg.norm(self._A, ord=np.inf)
        if norm == 0:
            raise ValueError("Невозможно подобрать параметр tau: матрица нулевая.")
        return 1.0 / norm

    def _ensure_spd(self) -> None:
        if not np.allclose(self._A, self._A.T, atol=1e-12):
            raise ValueError("Метод требует симметричную матрицу.")
        try:
            np.linalg.cholesky(self._A)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Метод требует положительно определенную матрицу.") from exc
