# scripts/run_seq.sh
#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:${PATH}"

make seq
./bin/seq_twin