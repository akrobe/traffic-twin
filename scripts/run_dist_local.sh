# scripts/run_dist_local.sh
#!/usr/bin/env bash
set -euo pipefail

# Ensure Homebrew mpirun and compilers are on PATH (macOS ARM)
export PATH="/opt/homebrew/bin:${PATH}"

make dist

# 1 controller, 1 predictor, 1 aggregator, 1 ingestor
# --oversubscribe is handy on laptops where logical cores < ranks
mpirun --oversubscribe -np 4 ./bin/dist_twin