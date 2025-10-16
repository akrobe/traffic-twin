# Makefile

# ------- OS/Toolchain detection -------
UNAME_S := $(shell uname -s)
BREW_PREFIX ?= /opt/homebrew

CXX    ?= c++
MPICXX ?= $(BREW_PREFIX)/bin/mpic++

CXXFLAGS = -O3 -std=gnu++20 -Wall -Wextra -march=native -I./ -I./common

# OpenMP with Apple Clang: use libomp's kegged paths
OMP_CFLAGS  = -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
OMP_LDFLAGS = -L$(BREW_PREFIX)/opt/libomp/lib -lomp

# OpenCL (macOS framework vs. generic libOpenCL on Linux)
ifeq ($(UNAME_S),Darwin)
  OPENCL_LIB = -framework OpenCL
else
  OPENCL_LIB = -lOpenCL
endif

SEQ_SRC  = seq/main.cpp \
           ingest/ingest.cpp aggregate/aggregate.cpp predict/predict.cpp control/control.cpp
SMP_SRC  = smp/main.cpp \
           ingest/ingest.cpp aggregate/aggregate.cpp predict/predict.cpp control/control.cpp
DIST_SRC = dist/main.cpp \
           ingest/ingest.cpp aggregate/aggregate.cpp predict/predict.cpp control/control.cpp

# Ensure Homebrew binaries on PATH for tasks/shells that don't inherit user env
export PATH := $(BREW_PREFIX)/bin:$(PATH)

all: seq smp dist

seq:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(SEQ_SRC) $(OPENCL_LIB) -o bin/seq_twin

smp:
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $(SMP_SRC) $(OPENCL_LIB) $(OMP_LDFLAGS) -o bin/smp_twin

dist:
	mkdir -p bin
	$(MPICXX) $(CXXFLAGS) $(OMP_CFLAGS) $(DIST_SRC) $(OPENCL_LIB) $(OMP_LDFLAGS) -o bin/dist_twin

clean:
	rm -rf bin results *.o **/*.o

# ----- Convenience run targets (useful for demos) -----
run-seq: seq
	./bin/seq_twin

run-smp: smp
	./bin/smp_twin

run-dist: dist
	# 1 controller, 1 predictor, 1 aggregator, 1 ingestor
	mpirun --oversubscribe -np 4 ./bin/dist_twin

.PHONY: all seq smp dist clean run-seq run-smp run-dist