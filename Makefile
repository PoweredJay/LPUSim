# One command: `make` rebuilds the binaries, runs every test and every run, and regenerates
# results/ (CSVs, results/figures/roofline.png, results/REPORT.md). See scripts/reproduce.sh.
#   make B=/tmp/lpu-build            build dir (default ./build, or ~/.cache/lpusim-build if this
#                                    checkout's path has spaces: make cannot build targets there)
#   make PYTHON=/path/to/python3     interpreter with matplotlib (pip install -r requirements.txt)
ifneq ($(word 2,$(CURDIR)),)
B      ?= $(HOME)/.cache/lpusim-build
else
B      ?= $(CURDIR)/build
endif
PYTHON ?= python3
# expand a leading ~ (zsh leaves it in PYTHON=~/...)
PY     := $(strip $(patsubst ~/%,$(HOME)/%,$(PYTHON)))

reproduce:
	B="$(B)" PYTHON="$(PYTHON)" bash scripts/reproduce.sh

build:
	$(MAKE) -C src B="$(B)" all

test: build
	cd src && "$(B)/tests" && "$(B)/tests_layers"

# Just the three ResNets (about 15 s): results/resnet{50,101,152}_summary.json and CSVs, plus a
# one-line summary each. ResNet-101/152 do not fit on-chip, so they run with --alias-weights
# (timing-only projections). Extra flags go to every run, e.g. make resnet ARGS="--no-layer-overlap"
resnet: build
	@mkdir -p results
	@cd src && for d in 50 101 152; do \
	  a=""; [ $$d -ne 50 ] && a="--alias-weights"; \
	  out=$$("$(B)/resnet" --depth $$d $$a --latency ../planning/instruction-timing-table.csv --out ../results $(ARGS) 2>&1); \
	  rc=$$?; printf '%s\n' "$$out" > ../results/resnet$$d.log; \
	  if [ $$rc -ne 0 ]; then printf '%s\n' "$$out" | tail -5; exit 1; fi; \
	  printf 'ResNet-%-3s ' $$d; printf '%s\n' "$$out" | grep -E '^simulated' | cut -d: -f1; \
	  printf '           '; printf '%s\n' "$$out" | grep -E '^latency' | cut -d';' -f1; \
	  if printf '%s\n' "$$out" | grep -q PROJECTION; then \
	    printf '           timing-only projection: '; printf '%s\n' "$$out" | grep PROJECTION | cut -d' ' -f2-4 | sed 's/$$/ weight tiles aliased/'; \
	  else printf '           '; printf '%s\n' "$$out" | grep 'logits vs'; fi; \
	done
	@if "$(PY)" -c "import matplotlib" 2>/dev/null; then "$(PY)" -I scripts/plots.py --only resnet; \
	  else echo "(chart skipped: $(PY) has no matplotlib; use make resnet PYTHON=/path/to/python3)"; fi
	@echo "Full outputs: results/resnet{50,101,152}_summary.json, results/resnet*.log, results/figures/resnet_ips.png"

# Latency sensitivity (ResNet-50): halve and double each assumed [A] latency alone and draw the
# tornado chart. Writes results/latency_sensitivity.csv and results/figures/latency_sensitivity.png
sensitivity: build
	"$(PY)" -I scripts/sensitivity.py --bin "$(B)/resnet"
	@if "$(PY)" -c "import matplotlib" 2>/dev/null; then "$(PY)" -I scripts/plots.py --only sensitivity; \
	  else echo "(chart skipped: $(PY) has no matplotlib; use make sensitivity PYTHON=/path/to/python3)"; fi

clean:
	rm -rf "$(B)"

.PHONY: reproduce build test resnet sensitivity clean
