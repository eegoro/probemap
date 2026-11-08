@echo off
setlocal
pushd "%~dp0.." || exit /b 1
set "QT_API=pyqt6"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
  echo Run scripts\install_probemap_windows.cmd first.
  pause
  popd
  exit /b 1
)
set "PROBEMAP_PYTHON=%CD%\.venv\Scripts\python.exe"
"%PROBEMAP_PYTHON%" -m probemap %*
set "PROBEMAP_EXIT=%ERRORLEVEL%"
if not "%PROBEMAP_EXIT%"=="0" pause
popd
exit /b %PROBEMAP_EXIT%
