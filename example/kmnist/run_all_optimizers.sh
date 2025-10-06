#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KMNIST_DIR="$PROJECT_ROOT/example/kmnist"

pushd "$PROJECT_ROOT" >/dev/null

LOGS_DIR="$PROJECT_ROOT/logs/kmnist"
mkdir -p "$LOGS_DIR"

SEEDS=${SEEDS:-"1 2 3"}

OPTIMIZERS=$(KMNIST_DIR="$KMNIST_DIR" python - <<'PY'
import importlib.util
import os
from pathlib import Path

kmnist_dir = Path(os.environ["KMNIST_DIR"])
spec = importlib.util.spec_from_file_location("kmnist_optimizers", kmnist_dir / "optimizers.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore[attr-defined]
print(" ".join(module.optimizer_names()))
PY
)

for SEED in $SEEDS; do
    for OPT in $OPTIMIZERS; do
        echo "Running optimizer: $OPT (seed $SEED)"
        python "$KMNIST_DIR/main.py" --optimizer "$OPT" --seed "$SEED"
    done
done

PLOT_OUTPUT="$LOGS_DIR/all_optimizers.png"
python "$KMNIST_DIR/plot_metrics.py" --logs-dir "$LOGS_DIR" --output "$PLOT_OUTPUT"
echo "Saved combined plot to $PLOT_OUTPUT"

popd >/dev/null


