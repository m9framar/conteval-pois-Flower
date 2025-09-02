#!/bin/bash
# filepath: /home/marci/2024-balazs-frank-marcell-poisoning-shapley/run_experiments.sh

# Usage: ./run_experiments.sh [dataset] [num_runs] [enable_wandb] [dist_modes] [contribution_method]    # Run experiments without attacks
# Example: ./run_experiments.sh tabular 5 true "iid dirichlet_0.1" shapley "targeted" "2,3,4,5 2 3 4 5"

# Default values
DATASET=${1:-"tabular"}
NUM_RUNS=${2:-10}
ENABLE_WANDB=${3:-"false"}
DIST_MODES=${4:-"iid dirichlet_1.0"}  # Default distribution modes
CONTRIB_METHOD=${5:-"loo"}  # Default contribution method: "loo" or "shapley"
ATTACK_TYPES=${6:-"targeted"}  # Default attack type: "targeted" or "self-promotion" or both, space-separated
ATTACK_ROUND_CONFIGS=${7:-"2 3 4 5"}  # Default attack round configurations, space-separated

# Project structure - using relative paths
PROJECT_ROOT="$(pwd)"
# Create timestamped results directory for this batch run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${PROJECT_ROOT}/results_${TIMESTAMP}"

# Create metadata file for this batch run
create_metadata() {
    local metadata_file="${RESULTS_DIR}/batch_metadata.txt"
    cat > "$metadata_file" << EOF
Batch Experiment Metadata
========================
Timestamp: $(date +"%Y-%m-%d %H:%M:%S")
Dataset: $DATASET
Number of runs per configuration: $NUM_RUNS
Contribution method: $CONTRIB_METHOD
Data distributions: $DIST_MODES
Attack types: $ATTACK_TYPES
Attack round configurations: $ATTACK_ROUND_CONFIGS
WandB logging enabled: $ENABLE_WANDB

Command used:
./run_experiments.sh "$DATASET" "$NUM_RUNS" "$ENABLE_WANDB" "$DIST_MODES" "$CONTRIB_METHOD" "$ATTACK_TYPES" "$ATTACK_ROUND_CONFIGS"

Results directory structure:
EOF
    echo "Metadata file created at: $metadata_file"
}

# Create results directories based on dataset and distribution
create_dirs() {
    local dataset=$1
    local dist_modes=($2)
    local contrib_method=$3
    local attack_types=($4)
    local attack_round_configs=($5)
    
    for dist_mode in "${dist_modes[@]}"; do
        mkdir -p "${RESULTS_DIR}/${dataset}/${contrib_method}/${dist_mode}/no_attack"
        for attack_type in "${attack_types[@]}"; do
            for attack_rounds in "${attack_round_configs[@]}"; do
                # Create safe directory name from attack_rounds (replace commas with underscores)
                attack_rounds_dir=$(echo "$attack_rounds" | sed 's/,/_/g')
                mkdir -p "${RESULTS_DIR}/${dataset}/${contrib_method}/${dist_mode}/${attack_type}/rounds_${attack_rounds_dir}"
            done
        done
    done
    
    echo "Created directories for ${dataset} dataset with distributions: $2, using contribution method: $3, attack types: $4, and attack rounds: $5"
}

# Configure dataset-specific settings
configure_dataset() {
    local dataset=$1
    
    case "$dataset" in
        "tabular")
            DATASET_DIR="fl-tabular"
            NUM_ROUNDS=5
            ;;
        "nlp")
            DATASET_DIR="quickstart-huggingface"
            NUM_ROUNDS=3
            ;;
        "fashion")
            DATASET_DIR="fl-vision-2"
            NUM_ROUNDS=5
            ;;
        *)
            echo "Unknown dataset: $dataset"
            echo "Supported datasets: tabular, nlp, fashion"
            exit 1
            ;;
    esac
}

# Extract alpha value from distribution mode string
get_alpha_value() {
    local dist_mode=$1
    if [[ $dist_mode == "iid" ]]; then
        echo ""  # No alpha for IID
    else
        # Extract number after "dirichlet_"
        echo "${dist_mode#dirichlet_}"
    fi
}

# Function to run experiments for a specific dataset and distribution
run_experiments() {
    local dataset=$1
    local attack_mode=$2  # boolean: true for attack, false for no attack
    local output_subdir=$3  # no_attack or attack_type/rounds_X
    local dist_mode=$4
    local contrib_method=$5
    local attack_type=$6  # "targeted" or "self-promotion" or empty if no attack
    local attack_rounds=$7  # attack rounds string like "2,3,4,5" or "3"
    local output_base_dir="${RESULTS_DIR}/${dataset}/${contrib_method}/${dist_mode}/${output_subdir}"
    
    # Extract distribution type and alpha value
    local alpha_value=$(get_alpha_value "$dist_mode")
    local dist_type=$([ "$dist_mode" == "iid" ] && echo "iid" || echo "dirichlet")
    
    for i in $(seq 1 $NUM_RUNS); do
        # Create a unique output directory for this specific run
        run_dir="${output_base_dir}/run_${i}"
        mkdir -p "$run_dir"
        
        echo "======================================================="
        echo "Running ${dataset} experiment $i"
        echo "Distribution: ${dist_mode}"
        echo "Contribution method: ${contrib_method}"
        echo "Attack mode: ${attack_mode}"
        if [ "$attack_mode" == "true" ]; then
            echo "Attack type: ${attack_type}"
            echo "Attack rounds: ${attack_rounds}"
        fi
        echo "WandB enabled: ${ENABLE_WANDB}"
        echo "Output directory: $run_dir"
        echo "======================================================="
        
        # Change to the dataset directory
        cd "${PROJECT_ROOT}/${DATASET_DIR}"
        
        # Format run config - include attack-type and attack-rounds only if attack is enabled
        if [ "$attack_mode" == "true" ]; then
            if [ "$dist_type" == "iid" ]; then
                run_config='enable-attacks='${attack_mode}' attack-type="'${attack_type}'" attack-rounds="'${attack_rounds}'" use-wandb="'${ENABLE_WANDB}'" partition-type="iid" contribution-method="'${contrib_method}'"'
            else
                run_config='enable-attacks='${attack_mode}' attack-type="'${attack_type}'" attack-rounds="'${attack_rounds}'" use-wandb="'${ENABLE_WANDB}'" partition-type="dirichlet" dirichlet-alpha='${alpha_value}' contribution-method="'${contrib_method}'"'
            fi
        else
            if [ "$dist_type" == "iid" ]; then
                run_config='enable-attacks='${attack_mode}' use-wandb="'${ENABLE_WANDB}'" partition-type="iid" contribution-method="'${contrib_method}'"'
            else
                run_config='enable-attacks='${attack_mode}' use-wandb="'${ENABLE_WANDB}'" partition-type="dirichlet" dirichlet-alpha='${alpha_value}' contribution-method="'${contrib_method}'"'
            fi
        fi
        
        echo "Run config: ${run_config}"
        
        # Execute command with the proper formatting
        flwr run . --run-config "${run_config}" 2>&1 | tee "${run_dir}/run.log"
        
        # Check for results.json in default location and copy if present
        if [ -f "output/results.json" ]; then
            cp output/results.json "${run_dir}/"
            echo "Copied results.json to ${run_dir}"
        fi
        
        # Return to the project root
        cd "$PROJECT_ROOT"
        
        echo "Experiment $i completed. Output saved to ${run_dir}"
        echo ""
        
        # Pause between runs
        sleep 3
    done
}

# Main execution

# Configure for the selected dataset
configure_dataset "$DATASET"

# Convert space-separated string to arrays
IFS=' ' read -r -a DIST_MODES_ARRAY <<< "$DIST_MODES"
IFS=' ' read -r -a ATTACK_TYPES_ARRAY <<< "$ATTACK_TYPES"
IFS=' ' read -r -a ATTACK_ROUND_CONFIGS_ARRAY <<< "$ATTACK_ROUND_CONFIGS"

# Create directories for each distribution mode
create_dirs "$DATASET" "$DIST_MODES" "$CONTRIB_METHOD" "$ATTACK_TYPES" "$ATTACK_ROUND_CONFIGS"

# Create metadata file
create_metadata

# Validate contribution method
if [[ "$CONTRIB_METHOD" != "loo" && "$CONTRIB_METHOD" != "shapley" ]]; then
    echo "Warning: Invalid contribution method specified: $CONTRIB_METHOD. Using default 'loo' method."
    CONTRIB_METHOD="loo"
fi

# Validate attack types
for attack_type in "${ATTACK_TYPES_ARRAY[@]}"; do
    if [[ "$attack_type" != "targeted" && "$attack_type" != "self-promotion" ]]; then
        echo "Warning: Invalid attack type specified: $attack_type. Valid options are 'targeted' and 'self-promotion'."
        ATTACK_TYPES="targeted"
        IFS=' ' read -r -a ATTACK_TYPES_ARRAY <<< "$ATTACK_TYPES"
        break
    fi
done

# Display summary of experiment configuration
echo "======================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "Dataset: $DATASET"
echo "Number of runs: $NUM_RUNS"
echo "Contribution method: $CONTRIB_METHOD"
echo "Data distributions: $DIST_MODES"
echo "Attack types: $ATTACK_TYPES"
echo "Attack round configurations: $ATTACK_ROUND_CONFIGS"
echo "WandB logging: $ENABLE_WANDB"
echo "Results will be saved in: $RESULTS_DIR"
echo "======================================================================"

# Run experiments for each distribution mode
for dist_mode in "${DIST_MODES_ARRAY[@]}"; do
    echo "======================================================================"
    echo "Starting experiments with distribution: ${dist_mode}"
    echo "======================================================================"
    
    # Run experiments without attacks
    echo "Starting ${NUM_RUNS} runs WITHOUT attacks for ${DATASET} dataset (${dist_mode})"
    run_experiments "$DATASET" false "no_attack" "$dist_mode" "$CONTRIB_METHOD" "" ""
    
    # Run experiments with each attack type and each attack round configuration
    for attack_type in "${ATTACK_TYPES_ARRAY[@]}"; do
        for attack_rounds in "${ATTACK_ROUND_CONFIGS_ARRAY[@]}"; do
            # Create safe directory name from attack_rounds (replace commas with underscores)
            attack_rounds_dir=$(echo "$attack_rounds" | sed 's/,/_/g')
            output_subdir="${attack_type}/rounds_${attack_rounds_dir}"
            
            echo "Starting ${NUM_RUNS} runs WITH ${attack_type} attack (rounds: ${attack_rounds}) for ${DATASET} dataset (${dist_mode})"
            run_experiments "$DATASET" true "${output_subdir}" "$dist_mode" "$CONTRIB_METHOD" "${attack_type}" "${attack_rounds}"
        done
    done
done

echo "All experiments completed!"
echo "Results saved in: ${RESULTS_DIR}/${DATASET}/${CONTRIB_METHOD}/"