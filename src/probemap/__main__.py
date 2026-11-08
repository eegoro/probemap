"""Launch ProbeMap with python -m probemap or the probemap command."""

def main(argv=None):
    import os
    # Select the installed backend before desktop_app imports qtpy.
    os.environ.setdefault("QT_API", "pyqt6")
    from .registration.desktop_app import main as run
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
