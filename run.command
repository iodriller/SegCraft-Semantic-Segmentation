#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
./run.sh "$@"
status=$?
if [ "$status" -ne 0 ] && [ "$status" -ne 130 ] && [ "$status" -ne 143 ]; then
  echo "SegCraft did not start. Review the error above."
  read -r -p "Press Return to close." _ || true
fi
exit "$status"
