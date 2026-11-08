"""Small shared widgets for the desktop workspace."""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QListView


def flat_tabs(widget):
    """A continuous workspace background and quiet, underline-only tabs."""
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
    widget.setStyleSheet('''
        QTabWidget {background:#242b36;}
        QTabWidget::pane {border:0;background:#242b36;}
        QTabBar {background:transparent;}
        QTabBar::tab {background:transparent;color:#9aabc0;border:0;border-bottom:2px solid transparent;padding:7px 10px;}
        QTabBar::tab:selected {background:transparent;color:#f5f8ff;border-bottom-color:#7ac7e5;}
        QTabBar::tab:hover {color:#f5f8ff;}
        QStackedWidget, QScrollArea {background:#242b36;border:0;}
        QGroupBox {background:transparent;border:0;}
        QLabel, QSlider {background:transparent;}
    ''')


def view_button(icon,title,callback):
    """Small, unobtrusive view actions for image headings."""
    from .desktop_project import icon_button
    from .desktop_slice_tools import tool_icon
    from qtpy.QtCore import QSize
    button=icon_button(icon,title,callback)
    button.setIcon(tool_icon(icon,color='#99c0d1e2'))
    button.setFixedSize(23,23)
    button.setIconSize(QSize(16,16))
    button.setStyleSheet('''QToolButton {background:transparent;border:0;padding:2px;}
        QToolButton:hover {background:rgba(122,199,229,28);}
        QToolButton:pressed {background:rgba(122,199,229,55);}''')
    return button


class PopupComboBox(QComboBox):
    """Keep popup labels readable even when the sidebar field is narrow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # The macOS menu delegate miscalculates its width under our stylesheet:
        # the checkmark consumes space reserved for the text. A standard item
        # view uses the same text metrics for layout and painting.
        view = QListView(self)
        view.setObjectName('choicePopup')
        view.setTextElideMode(Qt.TextElideMode.ElideNone)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setView(view)

    def showPopup(self):
        view = self.view()
        view.setMinimumWidth(max(self.width(), view.sizeHintForColumn(0)
                                 + 2 * view.frameWidth()
                                 + view.verticalScrollBar().sizeHint().width()))
        super().showPopup()
