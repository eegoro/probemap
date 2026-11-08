"""Build a macOS launcher for this checkout; the Python environment stays local."""
from pathlib import Path
import plistlib
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from probemap import __version__


def main():
    contents=ROOT/'ProbeMap.app/Contents'
    executable=contents/'MacOS/ProbeMap'
    resources=contents/'Resources'
    executable.parent.mkdir(parents=True,exist_ok=True)
    resources.mkdir(parents=True,exist_ok=True)
    executable.write_text('''#!/bin/zsh
set -eu
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
if [[ "$#" -gt 0 && "$1" == -psn_* ]]; then shift; fi
PROBEMAP_LOG_DIR="${TMPDIR:-/tmp}/probemap"
mkdir -p "$PROBEMAP_LOG_DIR"
exec >> "$PROBEMAP_LOG_DIR/launcher.log" 2>&1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export QT_API=pyqt6
for PROBEMAP_PYTHON in "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/.venv-registration/bin/python"; do
  if [[ -x "$PROBEMAP_PYTHON" ]]; then
    exec "$PROBEMAP_PYTHON" -m probemap "$@"
  fi
done
exec python3 -m probemap "$@"
''')
    executable.chmod(0o755)
    shutil.copy2(ROOT/'src/probemap/resources/icon.icns',resources/'ProbeMap.icns')
    info=dict(CFBundleName='ProbeMap',CFBundleDisplayName='ProbeMap',
              CFBundleIdentifier='org.probemap.app',CFBundleVersion=__version__,
              CFBundleShortVersionString=__version__,CFBundlePackageType='APPL',
              CFBundleExecutable='ProbeMap',CFBundleIconFile='ProbeMap.icns',
              NSHighResolutionCapable=True)
    with (contents/'Info.plist').open('wb') as handle: plistlib.dump(info,handle)
    print(ROOT/'ProbeMap.app')


if __name__=='__main__': main()
