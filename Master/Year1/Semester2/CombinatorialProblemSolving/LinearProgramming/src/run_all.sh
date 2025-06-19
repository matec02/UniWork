#!/bin/bash

INPUT_DIR="./instances"
OUTPUT_DIR="./out"
TIMEOUT_DURATION=60

mkdir -p "$OUTPUT_DIR"

total_generated=0
total_passed=0

for input_file in "$INPUT_DIR"/*.inp; do
    filename=$(basename "$input_file" .inp)
    output_file="$OUTPUT_DIR/$filename.out"
    tmp_output="tmp_output.txt"

    echo "Running: $filename"

    start_time=$(date +%s)

    # Run automaton and store output in a temp file
    if timeout "${TIMEOUT_DURATION}s" ./automaton_lp < "$input_file" > "$tmp_output"; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        mv "$tmp_output" "$output_file"
        echo "Generated $output_file (took ${elapsed}s)"
        ((total_generated++))
    else
        echo "TIMEOUT or error during automaton run for $filename"
        rm -f "$tmp_output"
        continue
    fi

    # Run checker
    if ./checker < "$output_file"; then
        echo "Checker PASSED for $filename"
        ((total_passed++))
    else
        echo "Checker FAILED for $filename"
    fi

    echo "-----------------------------"
done

echo ""
echo "Summary:"
echo "  Total solved under ${TIMEOUT_DURATION}s: $total_generated"
echo "  Total passed checker: $total_passed"
