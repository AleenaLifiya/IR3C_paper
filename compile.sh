#!/bin/bash
# Rebuild every table and figure from the archived results, then typeset the manuscript.
#   ./compile.sh          full rebuild
#   ./compile.sh --tex    typeset only
set -eu
cd "$(dirname "$0")"
PY=/home/aleena/miniforge3/envs/xdiff/bin/python
if [ "${1:-}" != "--tex" ]; then
  "$PY" scripts/make_tables.py
  "$PY" scripts/make_figures.py 2>&1 | grep -v -i "deprecat\|rank_zero" || true
fi
PKG_ROOT=/home/aleena/miniforge3/pkgs
LIB_PATH=""
for d in icu-78.3-hcab7f73_0 harfbuzz-14.2.1-h8af1aa0_1 graphite2-1.3.15-hfae3067_0 \
         freetype-2.14.3-h8af1aa0_1 libfreetype-2.14.3-h8af1aa0_1; do
  [ -d "$PKG_ROOT/$d/lib" ] && LIB_PATH="$LIB_PATH:$PKG_ROOT/$d/lib"
done
LD_LIBRARY_PATH="${LIB_PATH#:}:/home/aleena/miniforge3/lib" \
  "$PKG_ROOT/tectonic-0.17.0-h9c1a3b4_0/bin/tectonic" -X compile --keep-logs main.tex
cp main.pdf IR3C_MedIA_draft2.pdf
echo "built IR3C_MedIA/IR3C_MedIA_draft2.pdf"
