# Build:  make            (binaries land in ./build)
# If this folder is mounted noexec, build elsewhere: make B=/tmp/lpu-build
# Run:    make run-tests | make run-resnet | make run-roofline   (outputs to ../results)
# Everything at once (build, tests, every run, figures, results/REPORT.md): `make` in the repo root.
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra
B        ?= build
CORE     := compiler.cpp model.cpp lpusim.cpp sched.cpp ifetch.cpp
HDRS     := $(wildcard *.h) ../modules_full.h

all: $(B)/tests $(B)/tests_layers $(B)/resnet $(B)/roofline $(B)/viz_layers

$(B):
	mkdir -p $(B)
$(B)/tests: tests.cpp lpusim.cpp sched.cpp $(HDRS) | $(B)
	$(CXX) $(CXXFLAGS) -o $@ tests.cpp lpusim.cpp sched.cpp
$(B)/tests_layers: tests_layers.cpp $(CORE) $(HDRS) | $(B)
	$(CXX) $(CXXFLAGS) -o $@ tests_layers.cpp $(CORE)
$(B)/resnet: main_resnet.cpp report.cpp $(CORE) $(HDRS) | $(B)
	$(CXX) $(CXXFLAGS) -o $@ main_resnet.cpp report.cpp $(CORE)
$(B)/viz_layers: viz_layers.cpp $(CORE) $(HDRS) | $(B)
	$(CXX) $(CXXFLAGS) -o $@ viz_layers.cpp $(CORE)
$(B)/roofline: roofline.cpp $(CORE) $(HDRS) | $(B)
	$(CXX) $(CXXFLAGS) -o $@ roofline.cpp $(CORE)

# The binaries resolve ../planning and ../results relative to the working dir, so run from here.
LAT := ../planning/instruction-timing-table.csv
run-tests: all
	$(B)/tests && $(B)/tests_layers
run-resnet: $(B)/resnet
	mkdir -p ../results
	$(B)/resnet --depth 50 --latency $(LAT) --out ../results
	for d in 101 152; do $(B)/resnet --depth $$d --alias-weights --latency $(LAT) --out ../results || exit 1; done
run-roofline: $(B)/roofline
	mkdir -p ../results
	$(B)/roofline --latency $(LAT) --out ../results
clean:
	rm -rf $(B)
.PHONY: all run-tests run-resnet run-roofline clean
