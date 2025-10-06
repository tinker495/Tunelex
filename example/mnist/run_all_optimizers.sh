#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MNIST_DIR="$PROJECT_ROOT/example/mnist"

pushd "$PROJECT_ROOT" >/dev/null

LOGS_DIR="$PROJECT_ROOT/logs/mnist"
mkdir -p "$LOGS_DIR"

OPTIMIZERS=$(MNIST_DIR="$MNIST_DIR" python - <<'PY'
import importlib.util
import os
from pathlib import Path

mnist_dir = Path(os.environ["MNIST_DIR"])
spec = importlib.util.spec_from_file_location("mnist_optimizers", mnist_dir / "optimizers.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore[attr-defined]
print(" ".join(module.optimizer_names()))
PY
)

for OPT in $OPTIMIZERS; do
    echo "Running optimizer: $OPT"
    python "$MNIST_DIR/main.py" --optimizer "$OPT"
done

PLOT_OUTPUT="$LOGS_DIR/all_optimizers.png"
python "$MNIST_DIR/plot_metrics.py" --logs-dir "$LOGS_DIR" --output "$PLOT_OUTPUT"
echo "Saved combined plot to $PLOT_OUTPUT"

popd >/dev/null
