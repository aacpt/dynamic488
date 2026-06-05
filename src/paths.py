"""
Resolve repo-relative directories for data and figures.

The library lives flat in src/ and the runnable scripts in
scans_and_plots/; both import this module as a top-level module (with
src/ on the path). The repo root resolves in this order:
  1. the DYN488_ROOT environment variable, if set;
  2. the nearest ancestor of this module (src/paths.py) that contains a
     pyproject.toml - i.e. the repo root, independent of the current
     working directory;
  3. the nearest ancestor of the current working directory containing a
     pyproject.toml (fallback for weird installs);
  4. the current working directory.
"""

import os
from pathlib import Path


def repo_root():
    env = os.environ.get('DYN488_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    # Anchor on this file's location (src/paths.py) first, so the root is found
    #   no matter where the process was launched from; fall back to the cwd
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for d in (start, *start.parents):
            if (d / 'pyproject.toml').exists():
                return d
    return Path.cwd().resolve()


def results_dir(*parts):
    p = repo_root() / 'results'
    for part in parts:
        p = p / part
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir(*parts):
    p = repo_root() / 'figures'
    for part in parts:
        p = p / part
    p.mkdir(parents=True, exist_ok=True)
    return p


def circuits_dir():
    return repo_root() / 'circuits'


RESULTS = repo_root() / 'results'
FIGURES = repo_root() / 'figures'
CIRCUITS = repo_root() / 'circuits'