#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
check_only=false
for arg in "$@"; do
  case "$arg" in --check|--help|-h) check_only=true ;; esac
done
if [[ "$check_only" == true || $# == 0 || $EUID == 0 ]]; then
  exec /usr/bin/python3 "$root/scripts/install.py" "$@"
fi
exec sudo /usr/bin/python3 "$root/scripts/install.py" "$@"
