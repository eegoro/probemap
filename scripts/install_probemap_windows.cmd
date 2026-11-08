@echo off
setlocal
pushd "%~dp0.." || exit /b 1
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -c "import struct; assert struct.calcsize('P') == 8, 'Install 64-bit Python 3.12'"
if errorlevel 1 goto failed
py -3.12 -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install .
if errorlevel 1 goto failed
echo Installation completed. Open scripts\start_probemap_windows.cmd to launch ProbeMap.
pause
popd
exit /b 0
:failed
echo Installation failed. The error details are shown above.
echo ProbeMap requires 64-bit Python 3.12 and an Internet connection for installation.
pause
popd
exit /b 1
