# Setup

Use Python **3.12** in a new virtual environment. Installation needs internet access.
Download the repository (**Code → Download ZIP**) and unzip it. The source folder
is the folder containing `pyproject.toml`, `src`, and `scripts`.
The commands below install all required Python dependencies automatically.

## macOS and Linux

Install [Python 3.12 for macOS](https://www.python.org/downloads/macos/), or use
your Linux package manager. Open a terminal and use `cd` to enter the source
folder (for example, `cd ~/Downloads/probemap-main`; use your actual folder name).
Check `python3.12 --version`, then run:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
.venv/bin/python -m probemap
```

Run the last command each time you want to open ProbeMap. To open a saved project:

```sh
.venv/bin/python -m probemap --project /path/to/project.json
```

On Ubuntu, install `python3.12-venv` if creating the environment fails.
Use a desktop session with a working graphics driver for 3D views.
If Qt reports missing `libEGL.so.1` or OpenGL libraries on Ubuntu, run:

```sh
sudo apt-get update
sudo apt-get install -y libegl1 libopengl0 libgl1
```

On macOS, you can also run `scripts/start_probemap_macos.command` after installation.
For a local app shortcut, run `python3.12 scripts/build_macos_app.py`.
Keep the resulting `ProbeMap.app` in this source folder; it uses the local environment.

## Windows

Install [64-bit Python 3.12](https://www.python.org/downloads/windows/), including the Python launcher.
Unzip the source to a short path, such as `C:\ProbeMap`.

1. Open `scripts` and double-click `install_probemap_windows.cmd` once.
2. Double-click `start_probemap_windows.cmd` in the same folder to open the app.

Keep `scripts` inside the source folder. The installer creates `.venv` beside
`pyproject.toml`; the launcher uses that environment.

Or open PowerShell in the source folder (not inside `scripts`) and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m probemap
```

You do not need to activate the environment or change PowerShell execution policy.

## Optional ANTs tools

Landmark alignment, reconstruction, and ephys work with the base installation.
For ANTs transform export and the Python registration tools, run this after
installation, from the source folder:

**macOS / Linux**:

```sh
.venv/bin/python -m pip install ".[ants]"
```

**Windows (PowerShell)**:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[ants]"
```

On Windows, ANTs may also need the
[Microsoft Visual C++ runtime](https://github.com/ANTsX/ANTsPy/wiki/Installing-ANTsPy).

## Moving or updating

Copy the project folder **and its source images**, keeping their relative locations.
Open `project.json` on the new computer. Copy prepared `ephys` folders too if needed.
Cached plots work without the raw binary; preparing new data or using original-source
analysis needs access to that binary. Reconnect server paths on each computer.

Create a fresh environment on each OS. A macOS `.app` or `.venv` cannot run on Windows or Linux.
To update, replace the source files with the new download, keeping `.venv`.
On Mac/Linux, run `.venv/bin/python -m pip install .` from that folder.
On Windows, run `scripts/install_probemap_windows.cmd` again. If using ANTs, repeat
the optional ANTs command too.

## If the app does not open

Run it from a terminal to see the error. Check the Python version and graphics driver.
On Linux, missing Qt/OpenGL libraries depend on the distribution; see
[napari troubleshooting](https://napari.org/stable/troubleshooting.html).
For a bug report, include the OS, Python version, terminal error, and steps to reproduce.
