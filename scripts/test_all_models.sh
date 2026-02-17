#!/bin/bash
# Test all PET models: export, run C++ inference, compare with PyTorch reference
#
# Usage:
#   ./scripts/test_all_models.sh [--model <name>] [--energy-only]
#
# By default tests energy + forces. Use --energy-only to skip forces.
# If --model is given, only test that one model. Otherwise test all.

# Portable timeout wrapper (macOS lacks GNU timeout).
# Runs command, captures stdout+stderr to a file, kills after $secs seconds.
# Usage: run_with_timeout <secs> <outfile> <cmd> [args...]
#   Exit code: command's exit code, or 124 on timeout.
run_with_timeout() {
    local secs=$1; shift
    local outfile=$1; shift
    "$@" > "$outfile" 2>&1 &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watchdog=$!
    wait "$pid" 2>/dev/null
    local ret=$?
    kill "$watchdog" 2>/dev/null
    wait "$watchdog" 2>/dev/null
    # If killed by signal, return 124 (matching GNU timeout convention)
    if [[ $ret -gt 128 ]]; then
        return 124
    fi
    return $ret
}

ENERGY_ONLY=""
FILTER_MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --energy-only) ENERGY_ONLY="1"; shift ;;
        --model) FILTER_MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Models to test (all available small/xs HuggingFace models)
MODELS=(
    "pet-mad-s"
    "pet-omad-xs"
    "pet-omad-s"
    "pet-omat-xs"
    "pet-omat-s"
    "pet-spice-s"
)

# Filter to single model if requested
if [[ -n "$FILTER_MODEL" ]]; then
    MODELS=("$FILTER_MODEL")
fi

# Geometries to test
GEOMETRIES=(
    "geometries/water.xyz"
    "geometries/urea.xyz"
    "geometries/urea_molecule.xyz"
    "geometries/si.xyz"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

FORCES_FLAG=""
if [[ -z "$ENERGY_ONLY" ]]; then
    FORCES_FLAG="--forces"
    echo "Testing energy + forces"
else
    echo "Testing energy only"
fi

echo "========================================"
echo "PET Model Comparison: C++ vs PyTorch"
echo "========================================"
echo ""

# Create temp directory for intermediate files
TEST_TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/test_pet_XXXXXX")
trap "rm -rf '$TEST_TMPDIR'" EXIT

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for MODEL in "${MODELS[@]}"; do
    echo -e "${YELLOW}[$MODEL]${NC}"

    EXPORT_DIR="/tmp/test_${MODEL//-/_}"

    # Export model (with --forces unless energy-only)
    # Timeout: 300s for export (model download + tracing can be slow)
    echo "  Exporting..."
    run_with_timeout 300 "$TEST_TMPDIR/export_out.txt" uv run scripts/export_pytorch/export_pet_full.py --model "$MODEL" $FORCES_FLAG -o "$EXPORT_DIR"
    if [[ $? -ne 0 ]]; then
        echo -e "  ${RED}EXPORT FAILED${NC}"
        tail -5 "$TEST_TMPDIR/export_out.txt" | sed 's/^/    /'
        ((FAIL_COUNT++))
        echo ""
        continue
    fi

    for GEOM in "${GEOMETRIES[@]}"; do
        GEOM_NAME=$(basename "$GEOM")

        if [[ ! -f "$GEOM" ]]; then
            echo "  $GEOM_NAME: SKIP (not found)"
            ((SKIP_COUNT++))
            continue
        fi

        # Run C++ inference (timeout: 120s per geometry)
        if [[ -n "$FORCES_FLAG" ]]; then
            run_with_timeout 120 "$TEST_TMPDIR/cpp_out.txt" ./build/bin/graph_inference "$EXPORT_DIR" "$GEOM" --forces
        else
            run_with_timeout 120 "$TEST_TMPDIR/cpp_out.txt" ./build/bin/graph_inference "$EXPORT_DIR" "$GEOM"
        fi
        CPP_EXIT=$?
        CPP_OUTPUT=$(cat "$TEST_TMPDIR/cpp_out.txt")

        if [[ $CPP_EXIT -ne 0 ]]; then
            echo -e "  $GEOM_NAME: ${RED}C++ FAILED${NC}"
            tail -3 "$TEST_TMPDIR/cpp_out.txt" | sed 's/^/    /'
            ((FAIL_COUNT++))
            continue
        fi

        CPP_ENERGY=$(echo "$CPP_OUTPUT" | grep "Total energy:" | awk '{print $3}')
        CPP_TIME=$(echo "$CPP_OUTPUT" | grep "Compute time:" | awk '{print $3}')

        # Run Python reference (timeout: 120s; forces on by default, skip stress)
        run_with_timeout 120 "$TEST_TMPDIR/py_out.txt" uv run scripts/calc_energy_pytorch.py "$GEOM" --model "$MODEL" --no-stress
        PY_OUTPUT=$(cat "$TEST_TMPDIR/py_out.txt")
        PY_ENERGY=$(echo "$PY_OUTPUT" | grep "^Energy:" | head -1 | awk '{print $2}')
        PY_TIME=$(echo "$PY_OUTPUT" | grep "^Time:" | awk '{print $2}' | sed 's/s$//')

        # Compare energies
        if [[ -z "$CPP_ENERGY" ]] || [[ -z "$PY_ENERGY" ]]; then
            # Check if Python failed due to unsupported species
            if echo "$PY_OUTPUT" | grep -q "does not support the atomic type"; then
                echo "  $GEOM_NAME: SKIP (unsupported species)"
                ((SKIP_COUNT++))
            else
                echo -e "  $GEOM_NAME: ${RED}ERROR${NC} - Could not parse energies"
                echo "    C++ output: ${CPP_ENERGY:-'(none)'}"
                echo "    Python output: ${PY_ENERGY:-'(none)'}"
                ((FAIL_COUNT++))
            fi
            continue
        fi

        # Calculate energy difference
        EDIFF=$(python3 -c "print(f'{abs($CPP_ENERGY - ($PY_ENERGY)):.6f}')")
        # Energy tolerance: 0.01 eV accounts for float32 accumulation differences
        # between GGML's graph interpreter and PyTorch's eager evaluation.
        # Typical diffs are <10 μeV; 0.01 eV catches gross errors.
        EPASS=$(python3 -c "print('PASS' if abs($CPP_ENERGY - ($PY_ENERGY)) < 0.01 else 'FAIL')")

        # Compare forces if enabled
        FPASS="SKIP"
        FMAE=""
        FMAX_DIFF=""
        if [[ -z "$ENERGY_ONLY" ]]; then
            FORCE_RESULT=$(python3 -c "
import re, sys

def parse_forces(text):
    forces = []
    in_forces = False
    for line in text.split('\n'):
        if 'Forces' in line:
            in_forces = True
            continue
        if in_forces:
            m = re.match(r'\s*Atom\s+\d+\s*(?:\([^)]*\))?\s*:\s*\[([^\]]+)\]', line)
            if m:
                vals = [float(x.strip()) for x in m.group(1).split(',')]
                forces.append(vals)
            elif forces and not line.strip().startswith('Atom'):
                break
    return forces

cpp_text = open('$TEST_TMPDIR/cpp_out.txt').read()
py_text = open('$TEST_TMPDIR/py_out.txt').read()

cpp_forces = parse_forces(cpp_text)
py_forces = parse_forces(py_text)

if not cpp_forces or not py_forces:
    print(f'PARSE_ERROR cpp={len(cpp_forces)} py={len(py_forces)}')
    sys.exit(0)

if len(cpp_forces) != len(py_forces):
    print(f'LENGTH_MISMATCH cpp={len(cpp_forces)} py={len(py_forces)}')
    sys.exit(0)

total_ae = 0.0
max_ae = 0.0
count = 0
for cf, pf in zip(cpp_forces, py_forces):
    for cv, pv in zip(cf, pf):
        ae = abs(cv - pv)
        total_ae += ae
        max_ae = max(max_ae, ae)
        count += 1

mae = total_ae / count if count > 0 else 0.0
# Force tolerance: 0.05 eV/A max component error. Backward pass through
# decomposed layer norm and attention accumulates more error than the
# forward pass. Typical max diffs are <0.01 eV/A.
status = 'PASS' if max_ae < 0.05 else 'FAIL'
print(f'{status} {mae:.6f} {max_ae:.6f}')
")
            FPASS=$(echo "$FORCE_RESULT" | awk '{print $1}')
            FMAE=$(echo "$FORCE_RESULT" | awk '{print $2}')
            FMAX_DIFF=$(echo "$FORCE_RESULT" | awk '{print $3}')
        fi

        # Print results
        if [[ "$EPASS" == "PASS" ]] && { [[ "$FPASS" == "PASS" ]] || [[ "$FPASS" == "SKIP" ]]; }; then
            echo -e "  $GEOM_NAME: ${GREEN}PASS${NC} (E diff: ${EDIFF} eV)"
            ((PASS_COUNT++))
        else
            echo -e "  $GEOM_NAME: ${RED}FAIL${NC} (E diff: ${EDIFF} eV)"
            ((FAIL_COUNT++))
        fi

        echo "    E:  C++=${CPP_ENERGY}  Py=${PY_ENERGY} eV"

        if [[ -n "$FMAE" ]] && [[ "$FPASS" != "PARSE_ERROR" ]] && [[ "$FPASS" != "LENGTH_MISMATCH" ]]; then
            if [[ "$FPASS" == "PASS" ]]; then
                echo -e "    F:  ${GREEN}PASS${NC} MAE=${FMAE} max=${FMAX_DIFF} eV/A"
            else
                echo -e "    F:  ${RED}FAIL${NC} MAE=${FMAE} max=${FMAX_DIFF} eV/A"
            fi
        elif [[ "$FPASS" == "PARSE_ERROR" ]] || [[ "$FPASS" == "LENGTH_MISMATCH" ]]; then
            echo -e "    F:  ${RED}${FORCE_RESULT}${NC}"
        fi

        # Timing
        if [[ -n "$CPP_TIME" ]] && [[ -n "$PY_TIME" ]]; then
            SPEEDUP=$(python3 -c "
cpp_s = $CPP_TIME / 1000.0
py_s = $PY_TIME
if cpp_s > 0:
    print(f'{py_s/cpp_s:.1f}')
else:
    print('inf')
")
            echo -e "    ${CYAN}Time: C++=${CPP_TIME}ms  Py=${PY_TIME}s  (${SPEEDUP}x)${NC}"
        elif [[ -n "$CPP_TIME" ]]; then
            echo -e "    ${CYAN}Time: C++=${CPP_TIME}ms${NC}"
        fi
    done

    # Cleanup
    rm -rf "$EXPORT_DIR"
    echo ""
done

echo "========================================"
echo "Summary: $PASS_COUNT passed, $FAIL_COUNT failed, $SKIP_COUNT skipped"
echo "========================================"

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
