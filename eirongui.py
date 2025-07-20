from gui.app import EironGUI
from PyQt5.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    window = EironGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
