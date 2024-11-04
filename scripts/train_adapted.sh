#!/bin/sh

TRAIN_CODE=train.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
GPU=None


while getopts "p:d:c:n:w:g:r:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

echo "Experiment name: $EXP_NAME"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $GPU"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py


echo "Loading config in:" $CONFIG_DIR

echo " =========> RUN TASK <========="

if [ "${WEIGHT}" = "None" ]
then
    python tools/train.py \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR"
else
    python tools/train.py \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT"
fi