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
    echo "Usage: $0 <model.gguf> [--no-forces] [--warmup N] [--iterations N]"
    exit 1
fi

MODEL="$1"
shift
EXTRA_ARGS="$@"

# Backends to try (in order of preference)
BACKENDS=(cpu metal cuda hip vulkan sycl)

echo "=========================================="
echo "Backend Benchmark Comparison"
echo "Model: $MODEL"
echo "=========================================="
echo ""

# Collect CSV data
TMPFILE=$(mktemp)
echo "backend,atoms,time_ms,energy" > "$TMPFILE"

AVAILABLE_BACKENDS=()

for backend in "${BACKENDS[@]}"; do
    echo "Testing $backend backend..."
    if $BENCHMARK "$MODEL" --backend "$backend" --csv $EXTRA_ARGS 2>/dev/null | tail -n +2 >> "$TMPFILE"; then
        AVAILABLE_BACKENDS+=("$backend")
        echo "  OK"
    else
        echo "  Not available"
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

# Speedup table if CPU is available
if [[ " ${AVAILABLE_BACKENDS[*]} " =~ " cpu " ]]; then
    echo "=========================================="
    echo "Speedup vs CPU (higher is better)"
    echo "=========================================="
    echo ""

    printf "%8s" "Atoms"
    for backend in "${AVAILABLE_BACKENDS[@]}"; do
        if [ "$backend" != "cpu" ]; then
            printf "%12s" "$backend"
        fi
    done
    echo ""
    printf "%8s" "--------"
    for backend in "${AVAILABLE_BACKENDS[@]}"; do
        if [ "$backend" != "cpu" ]; then
            printf "%12s" "--------"
        fi
    done
    echo ""

    for n_atoms in $ATOMS; do
        printf "%8s" "$n_atoms"
        cpu_time=$(grep "^cpu,$n_atoms," "$TMPFILE" | cut -d',' -f3)
        for backend in "${AVAILABLE_BACKENDS[@]}"; do
            if [ "$backend" != "cpu" ]; then
                backend_time=$(grep "^$backend,$n_atoms," "$TMPFILE" | cut -d',' -f3)
                if [ -n "$backend_time" ] && [ -n "$cpu_time" ]; then
                    speedup=$(echo "scale=2; $cpu_time / $backend_time" | bc)
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
