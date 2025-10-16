# README.md

# traffic-twin

## How to run

This repository builds three variants of the traffic digital twin:

- **seq** — single-process baseline (sanity check)
- **smp** — threaded pipeline on one machine (OpenMP + lock-free queues)
- **dist** — distributed micro-pipeline with MPI ranks:
  - `Controller` (C), one or more `Predictor` workers (P), `Aggregator` (A), `Ingestor` (I)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
---

>>>>>>> Stashed changes
=======
---

>>>>>>> Stashed changes
=======
---

>>>>>>> Stashed changes
=======
---

>>>>>>> Stashed changes
### Prerequisites (macOS, Apple Silicon)

```bash
# One-time setup
brew install libomp open-mpi pocl clinfo

# Ensure brew binaries are on PATH in each new shell
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
export PATH=/opt/homebrew/bin:$PATH
=======
export PATH=/opt/homebrew/bin:$PATH
>>>>>>> Stashed changes
=======
export PATH=/opt/homebrew/bin:$PATH
>>>>>>> Stashed changes
=======
export PATH=/opt/homebrew/bin:$PATH
>>>>>>> Stashed changes
=======
export PATH=/opt/homebrew/bin:$PATH
>>>>>>> Stashed changes
