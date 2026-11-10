# Build:  make            (binaries land in ./build)
# If this folder is mounted noexec, build elsewhere: make B=/tmp/lpu-build
# Run:    make run-tests | make run-resnet | make run-roofline   (outputs to ../../results)
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
$(B)/resnet: main_resnet.cpp $(CORE) $(HDRS) | $(B)
	$(CXX) $(CXXFLAGS) -o $@ main_resnet.cpp $(CORE)
$(B)/viz_layers: viz_layers.cpp $(CORE) $(HDRS) | $(B)
	$(CXX) $(CXXFLAGS) -o $@ viz_layers.cpp $(CORE)
$(B)/roofline: roofline.cpp $(CORE) $(HDRS) | $(B)
	$(CXX) $(CXXFLAGS) -o $@ roofline.cpp $(CORE)

# binaries resolve ../../planning and ../../results relative to the working dir, so run from here
run-tests: all
	$(B)/tests && $(B)/tests_layers
run-resnet: $(B)/resnet
	for d in 50 101 152; do $(B)/resnet --depth $$d --alias-weights --latency ../../planning/instruction-timing-table.csv --out ../../results || exit 1; done
run-roofline: $(B)/roofline
	$(B)/roofline --out ../../results
clean:
	rm -rf $(B)
.PHONY: all run-tests run-resnet run-roofline clean
