#!/usr/bin/env bash
set -e
make dist
# 1 controller, 1 predictor, 1 aggregator, 1 ingestor
mpirun -np 4 ./bin/dist_twin