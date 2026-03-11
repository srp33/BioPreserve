#!/bin/bash
# Dry run all Snakemake workflows in the project (in parallel)
# Usage: ./dryrun_all.sh

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# List of workflows to test
WORKFLOWS=(
    "prepdata"
    "simulate_batches"
    "adjust_paired_datasets"
    "adjust_deep_learning"
    "classify_er_all_datasets"
    "classify_er_paired_datasets"
    "bayesian_shift_scale_adjuster"
    "evaluate_basis"
    "validate_adjusters"
)

echo "========================================"
echo "Snakemake Dry Run - All Workflows"
echo "========================================"
echo "Running ${#WORKFLOWS[@]} workflows in parallel..."
echo ""

# Create temp directory for results
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Function to test a single workflow
test_workflow() {
    local workflow=$1
    local result_file="$TEMP_DIR/${workflow}.result"
    local output_file="$TEMP_DIR/${workflow}.output"
    
    if [ ! -f "$workflow/Snakefile" ]; then
        echo "SKIP" > "$result_file"
        echo "No Snakefile found" > "$output_file"
        return
    fi
    
    # Run dry-run and capture output
    if output=$(pixi run snakemake --dry-run --directory "$workflow" --snakefile "$workflow/Snakefile" 2>&1); then
        echo "SUCCESS" > "$result_file"
        echo "$output" > "$output_file"
    else
        # Check for specific error types
        if echo "$output" | grep -q "Nothing to be done"; then
            echo "SUCCESS" > "$result_file"
        elif echo "$output" | grep -q "ProtectedOutputException"; then
            # Check if it's a permissions issue (files owned by another user)
            if echo "$output" | grep -q "Write-protected output files"; then
                echo "WARNING:Permission denied" > "$result_file"
            else
                echo "FAILED" > "$result_file"
            fi
        else
            echo "FAILED" > "$result_file"
        fi
        echo "$output" > "$output_file"
    fi
}

export -f test_workflow
export TEMP_DIR
export GREEN RED YELLOW NC

# Function to monitor and print results as they complete
monitor_results() {
    local completed=0
    local total=${#WORKFLOWS[@]}
    declare -A printed
    
    while [ $completed -lt $total ]; do
        for workflow in "${WORKFLOWS[@]}"; do
            # Skip if already printed
            if [ "${printed[$workflow]}" = "1" ]; then
                continue
            fi
            
            result_file="$TEMP_DIR/${workflow}.result"
            output_file="$TEMP_DIR/${workflow}.output"
            
            # Check if result is ready
            if [ -f "$result_file" ]; then
                result=$(cat "$result_file")
                
                echo "Testing: $workflow"
                echo "----------------------------------------"
                
                case "$result" in
                    SUCCESS)
                        echo -e "${GREEN}✓ $workflow - SUCCESS${NC}"
                        ;;
                    SKIP)
                        echo -e "${YELLOW}⊘ $workflow - No Snakefile found${NC}"
                        ;;
                    WARNING:*)
                        reason="${result#WARNING:}"
                        echo -e "${YELLOW}⚠ $workflow - $reason${NC}"
                        ;;
                    FAILED)
                        echo -e "${RED}✗ $workflow - FAILED${NC}"
                        cat "$output_file" | tail -n 10
                        ;;
                esac
                echo ""
                
                printed[$workflow]=1
                ((completed++))
            fi
        done
        sleep 0.2
    done
}

# Run all workflows in parallel
for workflow in "${WORKFLOWS[@]}"; do
    test_workflow "$workflow" &
done

# Monitor and print results as they complete
monitor_results

# Wait for all background jobs to complete
wait

# Collect results for summary
declare -a SUCCESSFUL=()
declare -a FAILED=()
declare -a WARNINGS=()

for workflow in "${WORKFLOWS[@]}"; do
    result_file="$TEMP_DIR/${workflow}.result"
    
    if [ ! -f "$result_file" ]; then
        continue
    fi
    
    result=$(cat "$result_file")
    
    case "$result" in
        SUCCESS)
            SUCCESSFUL+=("$workflow")
            ;;
        WARNING:*)
            reason="${result#WARNING:}"
            WARNINGS+=("$workflow: $reason")
            ;;
        FAILED)
            FAILED+=("$workflow")
            ;;
    esac
done

# Summary
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo -e "${GREEN}Successful: ${#SUCCESSFUL[@]}${NC}"
for wf in "${SUCCESSFUL[@]}"; do
    echo "  ✓ $wf"
done

if [ "${#WARNINGS[@]}" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Warnings: ${#WARNINGS[@]}${NC}"
    for wf in "${WARNINGS[@]}"; do
        echo "  ⚠ $wf"
    done
    echo ""
    echo "To fix permission warnings:"
    echo "  1. Check file ownership: ls -l <output_directory>"
    echo "  2. Request write access from file owner, or"
    echo "  3. Change output directory in workflow config, or"
    echo "  4. Use --unlock and --rerun-incomplete flags when running"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed: ${#FAILED[@]}${NC}"
    for wf in "${FAILED[@]}"; do
        echo "  ✗ $wf"
    done
    exit 1
fi

echo ""
echo "All workflows validated successfully!"
