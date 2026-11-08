# Paths shared by every run script.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
RUNS_DIR="$REPO_ROOT/runs"
MLFLOW_URI="http://localhost:5000"
