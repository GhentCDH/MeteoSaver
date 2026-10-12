#!/usr/bin/env bash
# Sequential training ladder; each run saves runs/<name>/merged and prints val CER.
set -u
cd "$(dirname "$0")"
PY="$HOME/Documents/Visual Code Repos/BelHisFirm---HisTableFinder/.venv-glm/bin/python"
for cfg in "500 n500" "2000 n2000" "8000 n8000" "0 full"; do
  set -- $cfg
  "$PY" -u train.py --n-train "$1" --name "$2" > "train_$2.log" 2>&1
  echo "finished $2 (exit $?)" >> ladder.log
done
