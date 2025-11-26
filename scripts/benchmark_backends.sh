#!/bin/bash
# Benchmark all available backends and compare results
#
# Usage: ./scripts/benchmark_backends.sh <model.gguf> [options]
#
# Options are passed through to backend_benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK="$PROJECT_DIR/build/bin/backend_benchmark"

if [ ! -f "$BENCHMARK" ]; then
    echo "Error: backend_benchmark not found at $BENCHMARK"
    echo "Run: ninja -C build"
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model.gguf> [options]"
    echo ""
    echo "Options (passed to backend_benchmark):"
    echo "  --no-forces     Energy only (fastest)"
    echo "  --nc-forces     Non-conservative forces via forward pass (fast)"
    echo "  (default)       Gradient forces via backprop (slow)"
    echo "  --warmup N      Warmup iterations (default: 2)"
    echo "  --iterations N  Timed iterations (default: 3)"
    echo ""
    echo "Examples:"
    echo "  $0 pet-mad.gguf --no-forces          # Fast energy-only benchmark"
    echo "  $0 pet-mad-1.1.0.gguf --nc-forces    # NC-forces benchmark (v1.1.0+)"
    exit 1
fi

MODEL="$1"
shift
EXTRA_ARGS="$@"

# Backends to try (in order of preference)
BACKENDS=(metal cuda hip cpu vulkan sycl)

# Determine mode for display
MODE="Energy + Forces (gradient)"
if [[ "$EXTRA_ARGS" == *"--no-forces"* ]]; then
    MODE="Energy only"
elif [[ "$EXTRA_ARGS" == *"--nc-forces"* ]]; then
    MODE="Energy + NC-Forces (forward)"
fi

echo "=========================================="
echo "Backend Benchmark Comparison"
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "=========================================="
echo ""

# Collect CSV data
TMPFILE=$(mktemp)
echo "backend,atoms,time_ms,energy" > "$TMPFILE"

AVAILABLE_BACKENDS=()

for backend in "${BACKENDS[@]}"; do
    echo -n "Testing $backend backend... "
    if $BENCHMARK "$MODEL" --backend "$backend" --csv $EXTRA_ARGS 2>/dev/null | tail -n +2 >> "$TMPFILE"; then
        AVAILABLE_BACKENDS+=("$backend")
        echo "OK"
    else
        echo "Not available"
    fi
done

echo ""
echo "=========================================="
echo "Results (time in ms, lower is better)"
echo "=========================================="
echo ""

# Print header
printf "%8s" "Atoms"
for backend in "${AVAILABLE_BACKENDS[@]}"; do
    printf "%12s" "$backend"
done
echo ""
printf "%8s" "--------"
for backend in "${AVAILABLE_BACKENDS[@]}"; do
    printf "%12s" "--------"
done
echo ""

# Get unique atom counts
ATOMS=$(tail -n +2 "$TMPFILE" | cut -d',' -f2 | sort -n | uniq)

for n_atoms in $ATOMS; do
    printf "%8s" "$n_atoms"
    for backend in "${AVAILABLE_BACKENDS[@]}"; do
        time_ms=$(grep "^$backend,$n_atoms," "$TMPFILE" | cut -d',' -f3)
        if [ -n "$time_ms" ]; then
            printf "%12s" "$time_ms"
        else
            printf "%12s" "-"
        fi
    done
    echo ""
done

echo ""

# Speedup table if we have multiple backends
if [ ${#AVAILABLE_BACKENDS[@]} -gt 1 ]; then
    # Use first backend as baseline
    BASELINE="${AVAILABLE_BACKENDS[0]}"

    echo "=========================================="
    echo "Speedup vs $BASELINE (higher is better)"
    echo "=========================================="
    echo ""

    printf "%8s" "Atoms"
    for backend in "${AVAILABLE_BACKENDS[@]}"; do
        if [ "$backend" != "$BASELINE" ]; then
            printf "%12s" "$backend"
        fi
    done
    echo ""
    printf "%8s" "--------"
    for backend in "${AVAILABLE_BACKENDS[@]}"; do
        if [ "$backend" != "$BASELINE" ]; then
            printf "%12s" "--------"
        fi
    done
    echo ""

    for n_atoms in $ATOMS; do
        printf "%8s" "$n_atoms"
        baseline_time=$(grep "^$BASELINE,$n_atoms," "$TMPFILE" | cut -d',' -f3)
        for backend in "${AVAILABLE_BACKENDS[@]}"; do
            if [ "$backend" != "$BASELINE" ]; then
                backend_time=$(grep "^$backend,$n_atoms," "$TMPFILE" | cut -d',' -f3)
                if [ -n "$backend_time" ] && [ -n "$baseline_time" ]; then
                    speedup=$(echo "scale=2; $baseline_time / $backend_time" | bc)
                    printf "%11sx" "$speedup"
                else
                    printf "%12s" "-"
                fi
            fi
        done
        echo ""
    done
fi

rm -f "$TMPFILE"
