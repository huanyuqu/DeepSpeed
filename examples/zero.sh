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

ZERO_STRATEGIES=(2 3 3.5 4)
CHECKPOINTING_OPTS=("false" "true")
BATCH_SEQ_PAIRS_NO_CHECKPOINTING=("1 512" "1 1024" "1 2048" "2 2048")
BATCH_SEQ_PAIRS_CHECKPOINTING=("4 2048" "8 2048" "16 2048" "32 2048")
STRATEGY_PARAMS_35=("(0,0)" "(1,1)" "(1,2)" "(2,1)" "(1,3)" "(3,1)")
STRATEGY_PARAMS_4=("(1,1)" "(1,2)" "(2,1)" "(1,3)" "(3,1)" "(1,4)" "(4,1)")


set -e

export NCCL_P2P_DISABLE=1

eval "$(conda shell.bash hook)"

conda activate deepspeed

if [ ! -d log ]; then
    mkdir log
fi

cd ..

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
                echo "Running ZeRO-$ZERO_STRATEGY, Batch size: $BATCH_SIZE, Seq len: $SEQ_LEN, Strategy param: $STRATEGY_PARAM, Checkpointing: $CHECKPOINTING"

                STRATEGY_PARAM_NAME="$STRATEGY_PARAM"
                STRATEGY_PARAM_NAME="${STRATEGY_PARAM_NAME//,/-}"
                STRATEGY_PARAM_NAME="${STRATEGY_PARAM_NAME//[()]/}"

                # checkpointing flag for deepspeed and suffix for logfile
                CHECKPOINTING_ARG=""
                if [ "$CHECKPOINTING" = "true" ]; then
                    CHECKPOINTING_ARG="--checkpointing"
                fi

                LOGFILE="examples/log/zero${ZERO_STRATEGY}_b${BATCH_SIZE}_s${SEQ_LEN}_p${STRATEGY_PARAM_NAME}_ck${CHECKPOINTING}_$(date +%Y%m%d_%H%M%S)_output.log"

                if [ -n "$STRATEGY_PARAM" ]; then
                    if ! deepspeed examples/zero_param_sweep.py \
                        --model examples/Qwen3-8B \
                        --zero $ZERO_STRATEGY \
                        --batch_size $BATCH_SIZE \
                        --seq_len $SEQ_LEN \
                        --steps 16 \
                        --warmup 3 \
                        $CHECKPOINTING_ARG \
                        --strategy_param $STRATEGY_PARAM \
                        --output_dir examples/results >> "$LOGFILE" 2>&1; then
                        echo "[ERROR] ZeRO-$ZERO_STRATEGY b${BATCH_SIZE} s${SEQ_LEN} p${STRATEGY_PARAM_NAME} ck${CHECKPOINTING} failed — see $LOGFILE"
                    fi
                else
                    if ! deepspeed examples/zero_param_sweep.py \
                        --model examples/Qwen3-8B \
                        --zero $ZERO_STRATEGY \
                        --batch_size $BATCH_SIZE \
                        --seq_len $SEQ_LEN \
                        --steps 16 \
                        --warmup 3 \
                        $CHECKPOINTING_ARG \
                        --output_dir examples/results >> "$LOGFILE" 2>&1; then
                        echo "[ERROR] ZeRO-$ZERO_STRATEGY b${BATCH_SIZE} s${SEQ_LEN} ck${CHECKPOINTING} (no param) failed — see $LOGFILE"
                    fi
                fi
            done
        done
    done
done
