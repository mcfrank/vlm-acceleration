#!/usr/bin/env bash
# Render the book with the project venv. One file per call is most reliable:
#   ./render.sh            whole book
#   ./render.sh 03-bouts.qmd
set -euo pipefail
# Chapters execute in the Jupyter kernel "vlm-acceleration", which must point at this repo's
# .venv. Register it once with:
#   .venv/bin/python -m ipykernel install --user --name vlm-acceleration
cd "$(dirname "$0")"
export QUARTO_PYTHON="$(cd .. && pwd)/.venv/bin/python"
quarto render "$@"
