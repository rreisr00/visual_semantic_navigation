import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from python_qt_binding.QtCore import QEvent, Qt
from python_qt_binding.QtGui import QKeyEvent
from python_qt_binding.QtWidgets import QApplication, QWidget

from semantic_evaluation.operator_gui import ArrowKeyFilter


class _ShortcutWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.voice_events = []
        self.voice_active = False
        self.voice_enabled = True
        self.direction_events = []
        self.released = 0

    def voice_shortcut_enabled(self):
        return self.voice_enabled

    def set_voice_push_to_talk(self, pressed):
        self.voice_active = pressed
        self.voice_events.append(pressed)

    def voice_shortcut_active(self):
        return self.voice_active

    def teleop_keys_enabled(self):
        return True

    def set_direction(self, direction, pressed):
        self.direction_events.append((direction, pressed))

    def release_keyboard_controls(self):
        self.released += 1


def _application():
    return QApplication.instance() or QApplication([])


def test_space_press_and_release_drive_push_to_talk():
    application = _application()
    window = _ShortcutWindow()
    event_filter = ArrowKeyFilter(window)

    pressed = QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier)
    released = QKeyEvent(QEvent.KeyRelease, Qt.Key_Space, Qt.NoModifier)

    assert event_filter.eventFilter(window, pressed)
    window.voice_enabled = False
    assert event_filter.eventFilter(window, released)
    assert window.voice_events == [True, False]
    assert application is QApplication.instance()


def test_arrow_keys_keep_press_and_release_semantics():
    application = _application()
    window = _ShortcutWindow()
    event_filter = ArrowKeyFilter(window)

    pressed = QKeyEvent(QEvent.KeyPress, Qt.Key_Up, Qt.NoModifier)
    released = QKeyEvent(QEvent.KeyRelease, Qt.Key_Up, Qt.NoModifier)

    assert event_filter.eventFilter(window, pressed)
    assert event_filter.eventFilter(window, released)
    assert window.direction_events == [('forward', True), ('forward', False)]
    assert application is QApplication.instance()
