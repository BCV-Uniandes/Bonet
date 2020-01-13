#!/bin/bash
START_EPOCH=0
NUM_EPOCHS=60
LR=0.0001
PATIENCE=2
BATCH_SIZE=20
NUM_WORKERS=0
NUM_GPUS=1
GPUS=0

SSD_LOCATION=
DATASET="RSNA" #RHPE
EXPERIMENT_NAME="Experiment"

DATA_TRAIN=
ANN_PATH_TRAIN=
ROIS_PATH_TRAIN=

DATA_VAL=
ANN_PATH_VAL=
ROIS_PATH_VAL=

DATA_TEST=
ANN_PATH_TEST=
ANN_PATH_TEST=

SAVE_FOLDER=$SSD_LOCATION"/TRAIN/"$EXPERIMENT_NAME
mkdir -p $SAVE_FOLDER

SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_snapshot.pth"
OPTIM_SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_optim.pth"


CUDA_VISIBLE_DEVICES=$GPUS mpirun -np $NUM_GPUS -H localhost:$NUM_GPUS -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x HOROVOD_CUDA_HOME=/usr/local/cuda-10.0 -mca pml ob1 -mca btl ^openib python -m pdb -c continue train.py --data-train $DATA_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --eval-first --workers $NUM_WORKERS #>> $SAVE_FOLDER"/log.txt" #Uncomment if you want a log of your training
