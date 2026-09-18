"""GUI entry point."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtCore import QLocale
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    argv = sys.argv if argv is None else argv
    QLocale.setDefault(QLocale.c())   # always use '.' as decimal separator
    app = QApplication(argv)
    app.setApplicationName("revdpd")
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    args = argv[1:]
    if args:
        if args[0].endswith(".json"):
            w.load_project(args[0])
        else:
            w.ed_cg.setText(args[0])
            w.load_cg()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
