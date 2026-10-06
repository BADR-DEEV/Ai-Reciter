#!/usr/bin/env bash
# Restart Rattil after the deploy workflow (.github/workflows/deploy.yml) has checked out the new commit,
# then wait until both servers answer. run.py reinstalls packages and rebuilds the web app only when
# requirements.txt, package-lock.json or the web sources changed. Runs as root through Systems Manager.
set -uo pipefail
systemctl restart rattil
for _ in $(seq 1 120); do
  sleep 5
  if curl -sf -o /dev/null http://127.0.0.1:8000/health && curl -sf -o /dev/null http://127.0.0.1:3000/; then
    echo "Rattil is up at $(git -c safe.directory='*' -C "$(dirname "$0")/.." log -1 --format='%h %s')"
    exit 0
  fi
  if ! systemctl is-active --quiet rattil; then
    echo "Rattil stopped while starting:"; journalctl -u rattil -n 40 --no-pager -o cat; exit 1
  fi
done
echo "Rattil did not answer within 10 minutes:"; journalctl -u rattil -n 40 --no-pager -o cat; exit 1
