"""Optional offline Qt viewer for ``.rlk-drift`` cross-config bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rl_engine.validation.reports.drift_report import load_drift_bundle

_LANES = [
    ("Audit", None),
    ("Training audit", "Training audit"),
    ("Rollout samples", "Rollout samples"),
    ("Execution", None),
    ("Operator / backend", "Operator / backend"),
    ("Token comparison", "Token comparison"),
    ("Validation", None),
    ("Drift markers", "Drift markers"),
]
_COLORS = {
    "pass": "#54a64f",
    "warning": "#e0a000",
    "failure": "#d53f3f",
    "info": "#72a6c8",
}


def _load_qt() -> tuple[Any, Any, Any]:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:  # pragma: no cover - depends on workstation extras
        raise RuntimeError(
            "The offline viewer requires the optional GUI dependency. "
            "Install it with: pip install 'rl-engine[drift-viewer]'"
        ) from exc
    return QtCore, QtGui, QtWidgets


def _format_event(event: dict[str, Any]) -> str:
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    payload = {
        "id": event.get("id"),
        "lane": event.get("lane"),
        "label": event.get("label"),
        "status": event.get("status"),
        "start": event.get("start"),
        "end": event.get("end"),
        "details": details,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def _run_qt(bundle_path: Path) -> int:  # pragma: no cover - GUI path
    QtCore, QtGui, QtWidgets = _load_qt()
    bundle = load_drift_bundle(bundle_path)
    report = bundle["report"]
    events = [event for event in report.get("events", []) if isinstance(event, dict)]
    title = str(report.get("title", "RL-Kernel cross-config drift report"))
    status = str(report.get("status", "info"))
    status_color = _COLORS.get(status, _COLORS["info"])

    class EventItem(QtWidgets.QGraphicsRectItem):  # type: ignore[name-defined]
        def __init__(self, rect: Any, event: dict[str, Any], color: str) -> None:
            super().__init__(rect)
            self.event = event
            self.setBrush(QtGui.QColor(color))
            self.setPen(QtGui.QPen(QtGui.QColor(color)))
            self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
            self.setToolTip(str(event.get("label", event.get("id", "event"))))

    class TimelineView(QtWidgets.QGraphicsView):  # type: ignore[name-defined]
        def __init__(self, scene: Any) -> None:
            super().__init__(scene)
            self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setStyleSheet("QGraphicsView { border: 0; background: #ffffff; }")

        def wheelEvent(self, event: Any) -> None:
            factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
            self.scale(factor, 1.0)
            event.accept()

    class Window(QtWidgets.QMainWindow):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"RL-Kernel Cross-Config Drift - {title}")
            self.resize(1500, 900)
            self._scene_width = 2200.0
            self._row_height = 52.0
            self._left_label_width = 8.0
            self._span = max(
                1.0,
                max(
                    (float(event.get("end", 1.0) or 1.0) for event in events),
                    default=1.0,
                ),
            )
            self._scene = QtWidgets.QGraphicsScene(self)
            self._scene.setBackgroundBrush(QtGui.QColor("#ffffff"))
            self._view = TimelineView(self._scene)
            self._tree = QtWidgets.QTreeWidget()
            self._tree.setHeaderLabel("Tracks")
            self._tree.setMinimumWidth(245)
            self._tree.setStyleSheet(
                "QTreeWidget { background: #f4f5f6; border: 0; "
                "font-family: 'Segoe UI'; font-size: 10pt; }"
            )
            self._details = QtWidgets.QPlainTextEdit()
            self._details.setReadOnly(True)
            self._details.setPlaceholderText("Select an event to inspect its audit details.")
            self._details.setStyleSheet(
                "QPlainTextEdit { background: #ffffff; border-top: 1px solid #c9ced3; "
                "font-family: Consolas; font-size: 10pt; padding: 8px; }"
            )
            self._build_tree()
            self._build_scene()
            self._scene.selectionChanged.connect(self._show_selected)
            self._tree.itemSelectionChanged.connect(self._select_lane)
            self._build_layout()

        def _build_tree(self) -> None:
            groups: dict[str, Any] = {}
            for label, event_lane in _LANES:
                if event_lane is None:
                    item = QtWidgets.QTreeWidgetItem([label])
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
                    self._tree.addTopLevelItem(item)
                    groups[label] = item
                    continue
                parent = groups.get(
                    "Audit" if event_lane in {"Training audit", "Rollout samples"} else "Execution"
                )
                if event_lane == "Drift markers":
                    parent = groups.get("Validation")
                item = QtWidgets.QTreeWidgetItem([label])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, event_lane)
                if parent is not None:
                    parent.addChild(item)
                else:
                    self._tree.addTopLevelItem(item)
            self._tree.expandAll()

        def _x_for(self, value: float) -> float:
            return 120.0 + max(0.0, min(self._span, value)) / self._span * self._scene_width

        def _build_scene(self) -> None:
            font = QtGui.QFont("Segoe UI", 9)
            muted = QtGui.QColor("#69737c")
            line = QtGui.QColor("#d6dade")
            grid = QtGui.QColor("#e8eaec")
            for tick in range(0, 17):
                x = self._x_for(self._span * tick / 16.0)
                self._scene.addLine(
                    x,
                    0,
                    x,
                    len(_LANES) * self._row_height,
                    QtGui.QPen(line if tick % 4 == 0 else grid),
                )
                label = self._scene.addText(f"{self._span * tick / 16.0:g}", font)
                label.setDefaultTextColor(muted)
                label.setPos(x + 3, -24)

            lane_index = {name: index for index, (_, name) in enumerate(_LANES) if name is not None}
            for index, (label, event_lane) in enumerate(_LANES):
                y = index * self._row_height
                self._scene.addLine(
                    0,
                    y + self._row_height,
                    self._scene_width + 140,
                    y + self._row_height,
                    QtGui.QPen(line),
                )
                if event_lane is None:
                    block = self._scene.addRect(
                        0,
                        y,
                        self._scene_width + 140,
                        self._row_height,
                        QtGui.QPen(),
                        QtGui.QBrush(QtGui.QColor("#eef0f2")),
                    )
                    block.setZValue(-2)
                else:
                    text = self._scene.addText(label, font)
                    text.setDefaultTextColor(QtGui.QColor("#252a2f"))
                    text.setPos(12, y + 17)

            for event in events:
                event_lane = str(event.get("lane", "Drift markers"))
                if event_lane not in lane_index:
                    continue
                y = lane_index[event_lane] * self._row_height + 16
                start = float(event.get("start", 0.0) or 0.0)
                end = float(event.get("end", start) or start)
                x0 = self._x_for(start)
                x1 = max(x0 + 10.0, self._x_for(end))
                color = _COLORS.get(str(event.get("status", "info")), _COLORS["info"])
                if event.get("kind") == "marker":
                    item = EventItem(QtCore.QRectF(x0 - 5, y + 5, 10, 10), event, color)
                else:
                    item = EventItem(QtCore.QRectF(x0, y, x1 - x0, 18), event, color)
                self._scene.addItem(item)
                if x1 - x0 > 130 and event.get("kind") != "marker":
                    text = self._scene.addText(str(event.get("label", "event")), font)
                    text.setDefaultTextColor(QtGui.QColor("#ffffff"))
                    text.setPos(x0 + 8, y - 1)
                    text.setZValue(1)
            self._scene.setSceneRect(
                0, -28, self._scene_width + 140, len(_LANES) * self._row_height + 40
            )

        def _build_layout(self) -> None:
            toolbar = QtWidgets.QToolBar()
            toolbar.setMovable(False)
            toolbar.setStyleSheet(
                "QToolBar { background: #ffffff; border-bottom: 1px solid #c9ced3; }"
            )
            status_label = QtWidgets.QLabel(f"  {title}    STATUS: {status.upper()}  ")
            status_label.setStyleSheet(f"color: {status_color}; font-weight: 600; padding: 5px;")
            toolbar.addWidget(status_label)
            fit_button = QtWidgets.QAction("Fit", self)
            fit_button.triggered.connect(
                lambda: self._view.fitInView(
                    self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio
                )
            )
            toolbar.addAction(fit_button)
            toolbar.addWidget(QtWidgets.QLabel(f"  {report.get('timeline_note', '')}"))
            self.addToolBar(toolbar)

            right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            right.addWidget(self._view)
            right.addWidget(self._details)
            right.setStretchFactor(0, 4)
            right.setStretchFactor(1, 1)
            root = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            root.addWidget(self._tree)
            root.addWidget(right)
            root.setStretchFactor(1, 1)
            self.setCentralWidget(root)

        def _select_lane(self) -> None:
            items = self._tree.selectedItems()
            if not items:
                return
            lane = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
            if not lane:
                return
            for item in self._scene.items():
                if isinstance(item, EventItem):
                    item.setOpacity(1.0 if item.event.get("lane") == lane else 0.28)

        def _show_selected(self) -> None:
            selected = self._scene.selectedItems()
            self._details.setPlainText(_format_event(selected[0].event) if selected else "")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    window._view.fitInView(window._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open an RL-Kernel drift bundle in the offline desktop viewer."
    )
    parser.add_argument("bundle", type=Path, help=".rlk-drift bundle")
    args = parser.parse_args(argv)
    if args.bundle.suffix.lower() != ".rlk-drift":
        parser.error("the offline viewer expects a .rlk-drift bundle")
    try:
        return _run_qt(args.bundle)
    except RuntimeError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
