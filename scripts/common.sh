#!/usr/bin/env bash
# Shared helpers for the IMUPoser shell scripts (sourced, not executed).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
IMU4D_DATA_ROOT="${IMU4D_DATA_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)/data}"

# Keep only the dataset specs whose WDS manifest exists; warn about the rest.
filter_available_datasets() {
    local kept=()
    for spec in "$@"; do
        if [ -f "$IMU4D_DATA_ROOT/processed/$spec/wds/manifest.json" ]; then
            kept+=("$spec")
        else
            echo "[skip] $spec: no $IMU4D_DATA_ROOT/processed/$spec/wds/manifest.json" >&2
            echo "       download with: hf download TianhangCheng7/IMU4DData --repo-type dataset --local-dir \"$IMU4D_DATA_ROOT/processed\" --include \"${spec%%/*}/**\"" >&2
        fi
    done
    [ ${#kept[@]} -gt 0 ] || { echo "no dataset available" >&2; exit 1; }
    local IFS=,
    echo "${kept[*]}"
}

# Best checkpoint of the newest run whose directory name starts with $1 (reads best_model.txt).
latest_best_checkpoint() {
    local run_dir
    run_dir="$(ls -d "$REPO_ROOT"/checkpoints/"$1"-* 2>/dev/null | sort | tail -1)"
    [ -n "$run_dir" ] || { echo "no run under $REPO_ROOT/checkpoints/$1-*" >&2; exit 1; }
    [ -f "$run_dir/best_model.txt" ] || { echo "$run_dir has no best_model.txt (training unfinished?)" >&2; exit 1; }
    head -1 "$run_dir/best_model.txt"
}
