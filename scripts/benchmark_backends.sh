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
TORCH_BENCHMARK="$PROJECT_DIR/local/benchmark_torch.py"

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
    echo "  --iterations N  Timed iterations (default: 10)"
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
echo "backend,atoms,time_ms,us_per_atom,energy" > "$TMPFILE"

AVAILABLE_BACKENDS=()

for backend in "${BACKENDS[@]}"; do
    # Capture output and only add backend if it produced data
    output=$($BENCHMARK "$MODEL" --backend "$backend" --csv $EXTRA_ARGS 2>/dev/null | tail -n +2)
    if [ -n "$output" ]; then
        echo "$output" >> "$TMPFILE"
        AVAILABLE_BACKENDS+=("$backend")
    fi
done

# Try torch baseline
if [ -f "$TORCH_BENCHMARK" ]; then
    # Extract warmup and iterations from args if present
    TORCH_ARGS="--csv"
    if [[ "$EXTRA_ARGS" =~ --warmup[[:space:]]+([0-9]+) ]]; then
        TORCH_ARGS="$TORCH_ARGS --warmup ${BASH_REMATCH[1]}"
    fi
    if [[ "$EXTRA_ARGS" =~ --iterations[[:space:]]+([0-9]+) ]]; then
        TORCH_ARGS="$TORCH_ARGS --iterations ${BASH_REMATCH[1]}"
    fi
    if [[ "$EXTRA_ARGS" == *"--no-forces"* ]]; then
        TORCH_ARGS="$TORCH_ARGS --no-forces"
    elif [[ "$EXTRA_ARGS" == *"--nc-forces"* ]]; then
        TORCH_ARGS="$TORCH_ARGS --nc-forces"
    fi
    output=$(uv run "$TORCH_BENCHMARK" $TORCH_ARGS 2>/dev/null | tail -n +2)
    if [ -n "$output" ]; then
        echo "$output" >> "$TMPFILE"
        AVAILABLE_BACKENDS+=("torch-cpu")
    fi
fi

echo ""
echo "=========================================="
echo "Results (time in ms, lower is better)"
echo "=========================================="
echo ""

# Print header
printf "%8s" "Atoms"
for backend in "${AVAILABLE_BACKENDS[@]}"; do
    printf "%10s %8s" "$backend" "us/atom"
done
echo ""
printf "%8s" "--------"
for backend in "${AVAILABLE_BACKENDS[@]}"; do
    printf "%10s %8s" "--------" "--------"
done
echo ""

# Get unique atom counts
ATOMS=$(tail -n +2 "$TMPFILE" | cut -d',' -f2 | sort -n | uniq)

for n_atoms in $ATOMS; do
    printf "%8s" "$n_atoms"
    for backend in "${AVAILABLE_BACKENDS[@]}"; do
        time_ms=$(grep "^$backend,$n_atoms," "$TMPFILE" | cut -d',' -f3)
        us_per_atom=$(grep "^$backend,$n_atoms," "$TMPFILE" | cut -d',' -f4)
        if [ -n "$time_ms" ]; then
            printf "%10s %8s" "$time_ms" "$us_per_atom"
        else
            printf "%10s %8s" "-" "-"
        fi
    done
    echo ""
done

echo ""

# Speedup table if we have CPU and at least one other backend
if [ ${#AVAILABLE_BACKENDS[@]} -gt 1 ] && grep -q "^cpu," "$TMPFILE"; then
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
