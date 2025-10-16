# scripts/run_smp.sh
#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:${PATH}"

make smp
./bin/smp_twin