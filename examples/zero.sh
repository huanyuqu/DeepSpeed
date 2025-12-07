#!/bin/bash

#SBATCH --job-name=zero
#SBATCH --partition=g_h100
#SBATCH -G 8
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --output=zero_output.%j.out
#SBATCH --error=zero_output.%j.err

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# This script runs the zero_param_sweep.py experiment using DeepSpeed for multiple ZeRO strategies, batch sizes, sequence lengths, and strategy parameters.
DO_SWEEP=$1  # "true" or "false"
MODEL_PATH="/root/highspeedstorage/ZeRO/Qwen3-8B"

if [ "$DO_SWEEP" = "true" ]; then
    ZERO_STRATEGIES=(2 3 3.5 4)
    CHECKPOINTING_OPTS=("false" "true")
    BATCH_SEQ_PAIRS_NO_CHECKPOINTING=("1 512" "1 1024" "1 2048" "2 2048")
    BATCH_SEQ_PAIRS_CHECKPOINTING=("4 2048" "8 2048" "16 2048" "32 2048")
    STRATEGY_PARAMS_35=("(0,0)" "(1,1)" "(1,2)" "(2,1)" "(1,3)" "(3,1)")
    STRATEGY_PARAMS_4=("(1,1)" "(1,2)" "(2,1)" "(1,3)" "(3,1)" "(1,4)" "(4,1)")
    MEMORY_SNAPSHOT=false
    PROFILE=false
else
    ZERO_STRATEGIES=(2 3 3.5 4)
    CHECKPOINTING_OPTS=("false")
    BATCH_SEQ_PAIRS_NO_CHECKPOINTING=("1 2048" "2 1536")
    BATCH_SEQ_PAIRS_CHECKPOINTING=("16 2048" "32 2048")
    STRATEGY_PARAMS_35=("(0,0)")
    STRATEGY_PARAMS_4=("(3,1)")
    MEMORY_SNAPSHOT=true
    PROFILE=false
fi

set -e

export PATH="/root/.local/bin:$PATH"
export PATH="/usr/local/cuda/bin:$PATH"

eval "$(conda shell.bash hook)"

if [ ! -d "/root/.triton/autotune" ]; then
    mkdir -p "/root/.triton/autotune"
fi

if ! conda activate /root/highspeedstorage/ZeRO/envs/zero; then
    echo "Failed to activate /root/highspeedstorage/ZeRO/envs/zero, trying zero..."
    if ! conda activate zero; then
        echo "Failed to activate zero environment, terminating..."
        exit 1
    fi
fi

cd ..

if [ ! -d examples/results ]; then
    mkdir -p examples/results
fi
OUTPUT_DIR="examples/results/$(date +%Y%m%d_%H%M%S)"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

if [ ! -d examples/log ]; then
    mkdir -p examples/log
fi
LOG_DIR="examples/log/$(date +%Y%m%d_%H%M%S)"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

if [ ! -d examples/profile ]; then
    mkdir -p examples/profile
fi
PROFILE_DIR="examples/profile/$(date +%Y%m%d_%H%M%S)"
if [ ! -d "$PROFILE_DIR" ]; then
    mkdir -p "$PROFILE_DIR"
fi

for ZERO_STRATEGY in "${ZERO_STRATEGIES[@]}"; do
    for CHECKPOINTING in "${CHECKPOINTING_OPTS[@]}"; do
        # choose batch-seq pairs depending on checkpointing option
        if [ "$CHECKPOINTING" = "true" ]; then
            BATCH_SEQ_PAIRS=("${BATCH_SEQ_PAIRS_CHECKPOINTING[@]}")
        else
            BATCH_SEQ_PAIRS=("${BATCH_SEQ_PAIRS_NO_CHECKPOINTING[@]}")
        fi

        for BATCH_SEQ in "${BATCH_SEQ_PAIRS[@]}"; do
            read -r BATCH_SIZE SEQ_LEN <<< "$BATCH_SEQ"

            if [ "$ZERO_STRATEGY" = "3.5" ]; then
                STRATEGY_PARAMS=("${STRATEGY_PARAMS_35[@]}")
            elif [ "$ZERO_STRATEGY" = "4" ]; then
                STRATEGY_PARAMS=("${STRATEGY_PARAMS_4[@]}")
            else
                STRATEGY_PARAMS=("")
            fi

            for STRATEGY_PARAM in "${STRATEGY_PARAMS[@]}"; do
                # normalize strategy param for filenames
                STRATEGY_PARAM_NAME="$STRATEGY_PARAM"
                STRATEGY_PARAM_NAME="${STRATEGY_PARAM_NAME//,/-}"
                STRATEGY_PARAM_NAME="${STRATEGY_PARAM_NAME//[()]/}"

                LOGFILE="$LOG_DIR/zero${ZERO_STRATEGY}_b${BATCH_SIZE}_s${SEQ_LEN}_param${STRATEGY_PARAM_NAME}_ck${CHECKPOINTING}_$(date +%Y%m%d_%H%M%S)_output.log"

                echo "Running ZeRO-$ZERO_STRATEGY, Batch size: $BATCH_SIZE, Seq len: $SEQ_LEN, Strategy param: $STRATEGY_PARAM, Checkpointing: $CHECKPOINTING, Memory Snapshot: $MEMORY_SNAPSHOT, Profile: $PROFILE, Model: $MODEL_PATH, Output: $OUTPUT_DIR, Profile dir: $PROFILE_DIR, Logfile: $LOGFILE"

                # checkpointing flag for deepspeed and suffix for logfile
                CHECKPOINTING_ARG=""
                if [ "$CHECKPOINTING" = "true" ]; then
                    CHECKPOINTING_ARG="--checkpointing"
                fi

                # Additional arguments
                ADDITIONAL_ARGS=""
                if [ "$MEMORY_SNAPSHOT" = "true" ]; then
                    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --memory_snapshot"
                fi
                if [ "$PROFILE" = "true" ]; then
                    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --profile"
                fi

                # Write the run info to LOGFILE as well
                echo "Running ZeRO-$ZERO_STRATEGY, Batch size: $BATCH_SIZE, Seq len: $SEQ_LEN, Strategy param: $STRATEGY_PARAM, Checkpointing: $CHECKPOINTING, Memory Snapshot: $MEMORY_SNAPSHOT, Profile: $PROFILE, Model: $MODEL_PATH, Output: $OUTPUT_DIR, Profile dir: $PROFILE_DIR, Logfile: $LOGFILE" >> "$LOGFILE"

                if [ -n "$STRATEGY_PARAM" ]; then
                    if ! deepspeed examples/zero_param_sweep.py \
                        --model $MODEL_PATH \
                        --zero $ZERO_STRATEGY \
                        --batch_size $BATCH_SIZE \
                        --seq_len $SEQ_LEN \
                        --steps 16 \
                        --warmup 3 \
                        $CHECKPOINTING_ARG \
                        --strategy_param $STRATEGY_PARAM \
                        --output_dir "$OUTPUT_DIR" \
                        --profile_dir "$PROFILE_DIR" \
                        $ADDITIONAL_ARGS >> "$LOGFILE" 2>&1; then
                        echo "[ERROR] ZeRO-$ZERO_STRATEGY b${BATCH_SIZE} s${SEQ_LEN} p${STRATEGY_PARAM_NAME} ck${CHECKPOINTING} ms${MEMORY_SNAPSHOT} pr${PROFILE} Model: $MODEL_PATH Output: $OUTPUT_DIR Profile: $PROFILE_DIR failed — see $LOGFILE"
                    fi
                else
                    if ! deepspeed examples/zero_param_sweep.py \
                        --model $MODEL_PATH \
                        --zero $ZERO_STRATEGY \
                        --batch_size $BATCH_SIZE \
                        --seq_len $SEQ_LEN \
                        --steps 16 \
                        --warmup 3 \
                        $CHECKPOINTING_ARG \
                        --output_dir "$OUTPUT_DIR" \
                        --profile_dir "$PROFILE_DIR" \
                        $ADDITIONAL_ARGS >> "$LOGFILE" 2>&1; then
                        echo "[ERROR] ZeRO-$ZERO_STRATEGY b${BATCH_SIZE} s${SEQ_LEN} ck${CHECKPOINTING} ms${MEMORY_SNAPSHOT} pr${PROFILE} Model: $MODEL_PATH (no param) Output: $OUTPUT_DIR Profile: $PROFILE_DIR failed — see $LOGFILE"
                    fi
                fi
            done
        done
    done
done
