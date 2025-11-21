#!/bin/bash

DATA_DIR="../data"
TRAIN="$DATA_DIR/ensup"
DEV="$DATA_DIR/endev"

RESULTS_DIR="results_fast"
mkdir -p $RESULTS_DIR

RNN_DIMS=(0 1 3 5)
LRS=(0.01 0.001)

MAX_STEPS=1500
EVAL_INTERVAL=200
BATCH=8
REG=0.0

for d in "${RNN_DIMS[@]}"; do
    for lr in "${LRS[@]}"; do

        MODEL_NAME="model_d${d}_lr${lr}.pkl"
        LOGFILE="$RESULTS_DIR/log_d${d}_lr${lr}.txt"

        echo "Running: d=$d lr=$lr" | tee $LOGFILE

        python3 tag.py "$DEV" \
            --train "$TRAIN" \
            --crf \
            --rnn_dim $d \
            --lr $lr \
            --batch_size $BATCH \
            --reg $REG \
            --model "$RESULTS_DIR/$MODEL_NAME" \
            --eval_interval $EVAL_INTERVAL \
            --max_steps $MAX_STEPS \
            | tee -a $LOGFILE

    done
done

echo "Done."
