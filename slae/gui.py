from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt5 import QtCore, QtGui, QtWidgets

from methods import AVAILABLE_METHODS
from solver import LinearSystemSolver


class SLAEWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Решатель СЛАУ (Qt)")
        self.setMinimumSize(1280, 820)
        self._apply_palette()

        self._method_keys = list(AVAILABLE_METHODS.keys())
        self._build_ui()
        self._resize_tables()
        self._update_parameter_state()

    def _apply_palette(self) -> None:
        base_color = QtGui.QColor(24, 28, 38)
        panel_color = QtGui.QColor(34, 39, 52)
        accent_color = QtGui.QColor(88, 140, 255)
        text_color = QtGui.QColor(236, 238, 244)

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, base_color)
        palette.setColor(QtGui.QPalette.Base, panel_color.darker(110))
        palette.setColor(QtGui.QPalette.AlternateBase, panel_color)
        palette.setColor(QtGui.QPalette.Button, panel_color)
        palette.setColor(QtGui.QPalette.ButtonText, text_color)
        palette.setColor(QtGui.QPalette.Text, text_color)
        palette.setColor(QtGui.QPalette.WindowText, text_color)
        palette.setColor(QtGui.QPalette.Highlight, accent_color)
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
        self.setPalette(palette)

        self.setStyleSheet(
            """
            QWidget { background-color: #1c1f28; color: #eceef4; font-size: 16px; }
            QLabel { color: #e0e4f0; font-size: 16px; }
            QGroupBox {
                border: 1px solid #2c3142;
                border-radius: 10px;
                margin-top: 18px;
                padding: 12px;
                background-color: #232735;
                font-size: 16px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 4px; color: #8fa2ff; font-weight: bold; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 6px 8px;
                border-radius: 6px;
                border: 1px solid #3a4156;
                background-color: #1e222f;
                selection-background-color: #5a87ff;
            }
            QPushButton {
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
                background-color: #587dff;
                border: none;
            }
            QPushButton:hover { background-color: #6a8dff; }
            QHeaderView::section {
                background-color: #2f3444;
                border: none;
                padding: 4px;
                border-right: 1px solid #3d4356;
                font-size: 15px;
            }
            QTextEdit {
                background-color: #1e2331;
                border-radius: 8px;
                border: 1px solid #2f3546;
                padding: 12px;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 16px;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSpacing(16)

        header = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Решение систем линейных уравнений")
        title_font = QtGui.QFont("Montserrat", 28, QtGui.QFont.Bold)
        title.setFont(title_font)
        subtitle = QtWidgets.QLabel(
            "Заполните коэффициенты матрицы A и вектора b, выберите численный метод и запустите решение."
        )
        subtitle.setStyleSheet("color: #9aa3bd;")
        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(20)

        size_box = QtWidgets.QGroupBox("Размер системы")
        size_layout = QtWidgets.QHBoxLayout(size_box)
        size_layout.setSpacing(12)
        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(2, 12)
        self.size_spin.setValue(3)
        self.size_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.size_spin.setFixedWidth(120)
        size_layout.addWidget(self.size_spin)
        self.resize_button = QtWidgets.QPushButton("Обновить таблицы")
        self.resize_button.setFixedHeight(44)
        size_layout.addWidget(self.resize_button)
        controls.addWidget(size_box)

        method_box = QtWidgets.QGroupBox("Метод решения")
        method_layout = QtWidgets.QVBoxLayout(method_box)
        method_layout.setSpacing(10)
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.setMinimumWidth(420)
        for key in self._method_keys:
            self.method_combo.addItem(f"{key.upper()} — {AVAILABLE_METHODS[key]}", key)
        method_layout.addWidget(self.method_combo)
        self.spd_hint = QtWidgets.QLabel(
            "Методы наискорейшего спуска и сопряжённых градиентов требуют симметрично положительно определённую матрицу."
        )
        self.spd_hint.setWordWrap(True)
        self.spd_hint.setStyleSheet("color: #7a88a8;")
        method_layout.addWidget(self.spd_hint)
        controls.addWidget(method_box, stretch=2)
        layout.addLayout(controls)

        self.params_box = QtWidgets.QGroupBox("Параметры метода")
        params_form = QtWidgets.QFormLayout(self.params_box)
        params_form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        params_form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        params_form.setHorizontalSpacing(18)
        params_form.setVerticalSpacing(12)
        self.params_form = params_form

        self.tau_edit = QtWidgets.QLineEdit()
        self.tau_edit.setPlaceholderText("Авто")
        row_tau = params_form.rowCount()
        params_form.addRow("τ:", self.tau_edit)

        self.tol_edit = QtWidgets.QLineEdit("1e-9")
        row_tol = params_form.rowCount()
        params_form.addRow("Точность:", self.tol_edit)

        self.max_iter_spin = QtWidgets.QSpinBox()
        self.max_iter_spin.setRange(1, 1_000_000)
        self.max_iter_spin.setValue(10_000)
        self.max_iter_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        row_iter = params_form.rowCount()
        params_form.addRow("Максимум итераций:", self.max_iter_spin)

        self.param_widgets: Dict[str, tuple[QtWidgets.QWidget, QtWidgets.QWidget]] = {}
        self.param_widgets["tau"] = (
            params_form.itemAt(row_tau, QtWidgets.QFormLayout.LabelRole).widget(),
            params_form.itemAt(row_tau, QtWidgets.QFormLayout.FieldRole).widget(),
        )
        self.param_widgets["tol"] = (
            params_form.itemAt(row_tol, QtWidgets.QFormLayout.LabelRole).widget(),
            params_form.itemAt(row_tol, QtWidgets.QFormLayout.FieldRole).widget(),
        )
        self.param_widgets["iter"] = (
            params_form.itemAt(row_iter, QtWidgets.QFormLayout.LabelRole).widget(),
            params_form.itemAt(row_iter, QtWidgets.QFormLayout.FieldRole).widget(),
        )

        layout.addWidget(self.params_box)

        matrix_box = QtWidgets.QGroupBox("Система Ax = b")
        matrix_layout = QtWidgets.QHBoxLayout(matrix_box)
        layout.addWidget(matrix_box, stretch=1)

        self.matrix_table = QtWidgets.QTableWidget()
        self.matrix_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.matrix_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.matrix_table.setAlternatingRowColors(False)
        self.matrix_table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: #3b4156;
                selection-background-color: #5a87ff;
                selection-color: #ffffff;
            }
            QTableWidget::item {
                padding: 4px;
            }
            """
        )
        matrix_layout.addWidget(self.matrix_table, stretch=3)

        self.vector_table = QtWidgets.QTableWidget()
        self.vector_table.setColumnCount(1)
        self.vector_table.setHorizontalHeaderLabels(["b"])
        self.vector_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.vector_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.vector_table.setAlternatingRowColors(False)
        matrix_layout.addWidget(self.vector_table, stretch=1)

        actions_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #9aa3bd;")
        actions_layout.addWidget(self.status_label)

        self.solve_button = QtWidgets.QPushButton("Решить систему")
        self.solve_button.setFixedWidth(200)
        actions_layout.addWidget(self.solve_button, alignment=QtCore.Qt.AlignRight)
        layout.addLayout(actions_layout)

        self.result_box = QtWidgets.QGroupBox("Результат")
        result_layout = QtWidgets.QVBoxLayout(self.result_box)
        self.result_text = QtWidgets.QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("После запуска решения здесь появятся значения x₀, x₁, ...")
        result_layout.addWidget(self.result_text)
        layout.addWidget(self.result_box, stretch=1)

        self.resize_button.clicked.connect(self._resize_tables)
        self.size_spin.valueChanged.connect(self._update_status_message)
        self.method_combo.currentIndexChanged.connect(self._update_parameter_state)
        self.solve_button.clicked.connect(self._solve_system)

    def _update_status_message(self) -> None:
        self.status_label.setText("")

    def _resize_tables(self) -> None:
        size = self.size_spin.value()
        self.matrix_table.setRowCount(size)
        self.matrix_table.setColumnCount(size)
        for i in range(size):
            self.matrix_table.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem(f"a{i}"))
            self.matrix_table.setVerticalHeaderItem(i, QtWidgets.QTableWidgetItem(str(i)))
            for j in range(size):
                item = QtWidgets.QTableWidgetItem("0")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setFont(QtGui.QFont("Consolas", 16))
                self.matrix_table.setItem(i, j, item)

        self.vector_table.setRowCount(size)
        for i in range(size):
            item = QtWidgets.QTableWidgetItem("0")
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            item.setFont(QtGui.QFont("Consolas", 16))
            self.vector_table.setItem(i, 0, item)

        self.matrix_table.verticalHeader().setDefaultSectionSize(60)
        self.vector_table.verticalHeader().setDefaultSectionSize(60)

        self.status_label.setText(f"Размер системы обновлён: {size}×{size}")

    def _update_parameter_state(self) -> None:
        key = self.method_combo.currentData()
        iterative_methods = {"simple", "jacobi", "seidel", "steepest", "cg"}
        tau_needed = key == "simple"
        iterations_needed = key in iterative_methods

        self._set_param_row_visible("tau", tau_needed)
        self._set_param_row_visible("tol", iterations_needed)
        self._set_param_row_visible("iter", iterations_needed)

        self.tau_edit.setEnabled(tau_needed)
        if not tau_needed:
            self.tau_edit.clear()
            self.tau_edit.setPlaceholderText("—")
        else:
            self.tau_edit.setPlaceholderText("Авто")

        self.tol_edit.setEnabled(iterations_needed)
        self.max_iter_spin.setEnabled(iterations_needed)

        self.params_box.setVisible(tau_needed or iterations_needed)

        self._update_status_message()
    
    def _set_param_row_visible(self, key: str, visible: bool) -> None:
        label, field = self.param_widgets[key]
        label.setVisible(visible)
        field.setVisible(visible)

    def _collect_matrix(self) -> List[List[float]]:
        rows = self.matrix_table.rowCount()
        cols = self.matrix_table.columnCount()
        matrix: List[List[float]] = []
        for i in range(rows):
            row_values: List[float] = []
            for j in range(cols):
                item = self.matrix_table.item(i, j)
                row_values.append(self._parse_float(item, f"A[{i}][{j}]"))
            matrix.append(row_values)
        return matrix

    def _collect_vector(self) -> List[float]:
        rows = self.vector_table.rowCount()
        values: List[float] = []
        for i in range(rows):
            item = self.vector_table.item(i, 0)
            values.append(self._parse_float(item, f"b[{i}]"))
        return values

    def _collect_params(self) -> Dict[str, float | int]:
        params: Dict[str, float | int] = {}
        key = self.method_combo.currentData()

        if key == "simple":
            tau_text = self.tau_edit.text().strip()
            if tau_text:
                params["tau"] = self._parse_text_float(tau_text, "τ")

        if key in {"simple", "jacobi", "seidel", "steepest", "cg"}:
            tol_text = self.tol_edit.text().strip()
            if not tol_text:
                raise ValueError("Поле точности не должно быть пустым.")
            tol_value = self._parse_text_float(tol_text, "точность")
            if tol_value <= 0:
                raise ValueError("Точность должна быть положительной.")
            params["tolerance"] = tol_value
            params["max_iterations"] = int(self.max_iter_spin.value())

        return params

    def _solve_system(self) -> None:
        try:
            matrix = self._collect_matrix()
            vector = self._collect_vector()
            params = self._collect_params()
            method = self.method_combo.currentData()

            start_numpy = perf_counter()
            baseline = np.linalg.solve(np.asarray(matrix, dtype=float), np.asarray(vector, dtype=float))
            numpy_time = perf_counter() - start_numpy

            solver = LinearSystemSolver(matrix, vector)
            start_method = perf_counter()
            result = solver.solve(method=method, **params)
            method_time = perf_counter() - start_method

            diff = np.abs(baseline - np.asarray(result, dtype=float))
            title = AVAILABLE_METHODS.get(method, method.upper())

            lines: List[str] = [
                f"NumPy solve (время: {numpy_time * 1e3:.3f} мс)",
            ]
            for idx, value in enumerate(baseline):
                lines.append(f"  x[{idx}] = {float(value):.10f}")

            lines.append("")
            lines.append(f"{title} (время: {method_time * 1e3:.3f} мс)")
            for idx, value in enumerate(result):
                lines.append(f"  x[{idx}] = {value:.10f}    Δ={diff[idx]:.3e}")

            lines.append("")
            lines.append(f"Максимальное отклонение: {diff.max():.3e}")

            self.result_text.setPlainText("\n".join(lines))
            self.status_label.setText(f"Решение найдено методом {method.upper()}.")
            self.status_label.setStyleSheet("color: #7ed493;")
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Ошибка решения", str(error))
            self.status_label.setText(f"Ошибка: {error}")
            self.status_label.setStyleSheet("color: #ff8a8a;")
            self.result_text.clear()

    @staticmethod
    def _parse_float(item: QtWidgets.QTableWidgetItem | None, field: str) -> float:
        if item is None or not item.text().strip():
            raise ValueError(f"Поле {field} должно быть заполнено.")
        text = item.text().replace(",", ".").strip()
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Поле {field} содержит некорректное число: '{text}'.") from exc

    @staticmethod
    def _parse_text_float(value: str, field: str) -> float:
        text = value.replace(",", ".")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Поле {field} содержит некорректное число: '{value}'.") from exc


def main() -> None:
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = SLAEWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
