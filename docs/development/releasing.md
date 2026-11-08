# Development and release

## Files

| Path | Contents |
| --- | --- |
| `src/probemap/` | Application code and icons |
| `tests/` | Tests and small data fixtures |
| `docs/` | Guides and screenshots |
| `docs/development/` | Calculation notes and release checks |
| `scripts/` | Windows installer and Mac/Windows launchers |
| `pyproject.toml`, `MANIFEST.in` | Dependencies and package contents |
| `LICENSE` | MIT license |

Generated files (`build/`, `dist/`, `*.egg-info`, and caches) are excluded from Git.
They can be deleted and recreated. Local environments, `ProbeMap.app`, data, and
`outputs/` are also excluded. Keep the environment and app if you use them locally.

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/install_probemap_windows.cmd` | Create `.venv` and install or update ProbeMap |
| `scripts/start_probemap_windows.cmd` | Open ProbeMap on Windows |
| `scripts/start_probemap_macos.command` | Open ProbeMap from the source folder on Mac |
| `scripts/build_macos_app.py` | Create a local Mac app shortcut |

The Mac shortcut needs this source folder and its Python environment.
It is not a standalone installer.

## Run tests and build

In your virtual environment, install the test and build tools:

```sh
python -m pip install ".[test,ants]" build
python -m pytest -m "not gui"
python -m pytest -m gui
python -m build
```

Full GUI tests need a display and OpenGL. GitHub Actions runs core tests and
offscreen ephys tests on macOS, Windows, and Linux.

## Before release

1. Confirm permission to publish the code. Include `LICENSE` in the release.
2. Check GitHub Actions results.
3. On each OS, create and reopen a project. Check alignment, 3D views, tracks,
   probes, reconstruction, ephys, and exports.
4. On Windows, prepare and cancel a small recording. Check file replacement
   and network paths if used.

## Last check: 8 November 2026

On macOS, 175 core tests and 4 offscreen ephys tests passed. A clean source copy
built a source archive and wheel. The installed wheel passed `--version` and
`--help` checks. The archive included docs, screenshots, fixtures, and MIT license.
Local data and environments were excluded. Documentation links resolved.

Full native GUI tests were not verified in the restricted runner: napari could
not write its user cache, and a later combined run crashed. Native Windows/Linux
desktop checks and GitHub Actions results remain to be checked.
