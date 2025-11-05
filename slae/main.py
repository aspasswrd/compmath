from __future__ import annotations

from typing import Any, Dict

from solver import LinearSystemSolver

AVAILABLE_METHODS: Dict[str, str] = {
    "lu": "LU-разложение",
    "simple": "Метод простых итераций",
    "jacobi": "Метод Якоби",
    "seidel": "Метод Зейделя",
    "steepest": "Метод наискорейшего спуска (требует SPD матрицу)",
    "cg": "Метод сопряженных градиентов (требует SPD матрицу)",
}


def _read_system_from_input():
    """Считывает систему Ax = b из стандартного ввода."""
    try:
        n = int(input("Введите размер системы (n): ").strip())
    except ValueError as exc:
        raise ValueError("Размер системы должен быть целым числом.") from exc

    if n <= 0:
        raise ValueError("Размер системы должен быть положительным.")

    matrix = []
    print("Введите матрицу A построчно, элементы через пробел:")
    for row_idx in range(n):
        while True:
            raw = input(f"A[{row_idx}] = ").strip()
            parts = raw.split()
            if len(parts) != n:
                print(f"Нужно ввести ровно {n} чисел. Повторите попытку.")
                continue
            try:
                row = [float(value.replace(",", ".")) for value in parts]
            except ValueError:
                print("Не удалось преобразовать ввод в числа. Повторите попытку.")
                continue
            matrix.append(row)
            break

    vector = []
    print("Введите вектор правых частей b (по одному числу):")
    for idx in range(n):
        while True:
            raw = input(f"b[{idx}] = ").strip()
            try:
                value = float(raw.replace(",", "."))
            except ValueError:
                print("Не удалось преобразовать ввод в число. Повторите попытку.")
                continue
            vector.append(value)
            break

    return matrix, vector


def _ask_method() -> str:
    """Возвращает выбранный пользователем метод решения."""
    print("Доступные методы:")
    for key, desc in AVAILABLE_METHODS.items():
        print(f" - {key}: {desc}")

    while True:
        choice = input("Выберите метод [lu]: ").strip().lower()
        if not choice:
            return "lu"
        if choice in AVAILABLE_METHODS:
            return choice
        print("Метод не распознан. Повторите попытку.")


def _ask_optional_float(prompt: str, default: float | None) -> float | None:
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("Не удалось преобразовать ввод в число. Повторите попытку.")


def _ask_optional_int(prompt: str, default: int) -> int:
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Не удалось преобразовать ввод в целое число. Повторите попытку.")
            continue
        if value <= 0:
            print("Значение должно быть положительным. Повторите попытку.")
            continue
        return value


def _collect_method_params(method: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    if method == "simple":
        tau = _ask_optional_float("Введите параметр tau (Enter = по умолчанию): ", None)
        tolerance = _ask_optional_float("Введите точность (tol) [1e-9]: ", 1e-9)
        max_iterations = _ask_optional_int("Введите максимум итераций [10000]: ", 10_000)
        if tau is not None:
            params["tau"] = tau
        if tolerance is not None:
            params["tolerance"] = tolerance
        params["max_iterations"] = max_iterations
    elif method in {"jacobi", "seidel", "steepest", "cg"}:
        tolerance = _ask_optional_float("Введите точность (tol) [1e-9]: ", 1e-9)
        max_iterations = _ask_optional_int("Введите максимум итераций [10000]: ", 10_000)
        if tolerance is not None:
            params["tolerance"] = tolerance
        params["max_iterations"] = max_iterations

    if method in {"steepest", "cg"}:
        print("Напоминание: требуется симметричная положительно определенная матрица.")

    return params


def main() -> None:
    try:
        A, b = _read_system_from_input()
    except ValueError as error:
        print(f"Ошибка ввода: {error}")
        return

    method = _ask_method()
    params = _collect_method_params(method)
    solver = LinearSystemSolver(A, b)

    try:
        solution = solver.solve(method=method, **params)
    except (ZeroDivisionError, ValueError, RuntimeError) as error:
        print(f"Не удалось найти решение методом {method}: {error}")
        return

    print(f"Решение ({method.upper()}):")
    for idx, value in enumerate(solution):
        print(f"x[{idx}] = {value}")


if __name__ == "__main__":
    main()
