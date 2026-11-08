"""Exercise saved-track compatibility tools without restoring removed nav pages."""
def open_legacy(window,old_step):
    from qtpy.QtWidgets import QApplication
    from test_desktop_tracks import wait_until
    app=QApplication.instance()
    wait_until(app,lambda:not window.busy)
    window.go_step(5)
    workspace=window.track_workspace
    wait_until(app,lambda:not window.busy and workspace.track_native is not None and not workspace._running)
    workspace.open(window.project,window.reference,old_step-4)
