#!/bin/zsh
set -eu
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export QT_API=pyqt6
for PROBEMAP_PYTHON in "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/.venv-registration/bin/python"; do
  if [[ -x "$PROBEMAP_PYTHON" ]]; then
    exec "$PROBEMAP_PYTHON" -m probemap "$@"
  fi
done
exec python3 -m probemap "$@"
