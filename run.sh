#!/usr/bin/env sh
# Rattil: set up and run with one command on macOS / Linux. See README.md.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then exec .venv/bin/python run.py "$@"; fi
for py in python3.12 python3.11 python3.13 python3.10 python3 python python3.14; do
  if command -v "$py" >/dev/null 2>&1 &&
     "$py" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 14))' 2>/dev/null; then
    exec "$py" run.py "$@"
  fi
done
echo "Rattil needs Python 3.10-3.13 (3.12 recommended): https://www.python.org/downloads/" >&2
exit 1
