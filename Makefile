# Makefile
UNAME_S := $(shell uname -s)
BREW_PREFIX ?= /opt/homebrew

CXX    ?= c++
MPICXX ?= $(BREW_PREFIX)/bin/mpic++

CXXFLAGS    = -O3 -std=gnu++20 -Wall -Wextra -march=native -I./ -I./common
OMP_CFLAGS  = -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
OMP_LDFLAGS = -L$(BREW_PREFIX)/opt/libomp/lib -lomp
OPENCL_LIB  = -framework OpenCL

SEQ_SRC  = seq/main.cpp ingest/ingest.cpp aggregate/aggregate.cpp predict/predict.cpp control/control.cpp
SMP_SRC  = smp/main.cpp ingest/ingest.cpp aggregate/aggregate.cpp predict/predict.cpp control/control.cpp
DIST_SRC = dist/main.cpp ingest/ingest.cpp aggregate/aggregate.cpp predict/predict.cpp control/control.cpp

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

.PHONY: all seq smp dist clean
