#!/usr/bin/env bash
# docker run -itd --gpus all -p 6006:6006 -v /home/gopalan_iyengar/SuperMapNet/:/workspace/SuperMapNet -v /media/wolfrush/data/samba-rd-data/:/workspace/SuperMapNet/data/ gopalan/supermapnet

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh train-continue pivotnet_nuscenes_swint_dense 30 1 8 /workspace/SuperMapNet/outputs/pivotnet_nuscenes_swint_dense/pivotnet_nuscenes_swint_dense_e30_b1_g8-2026-01-23T04:58:43/dump_model/checkpoint_epoch_14.pth
# CUDA_VISIBLE_DEVICES=0 bash run.sh test pivotnet_nuscenes_swint_dense outputs/pivotnet_nuscenes_swint_dense/latest/dump_model/checkpoint_epoch_29.pth
# CUDA_VISIBLE_DEVICES=0 bash run.sh test pivotnet_nuscenes_swint_dense weights/60_30_ckpt_29.pth
# CUDA_VISIBLE_DEVICES=1 bash run.sh test pivotnet_nuscenes_swint weights/120_30_ckpt_29.pth

# tmux new -s data_gen_bezier
# tmus ls 
# ctrl+b d
# tmux attach -t data_gen_bezier

#export PYTHONPATH=$(pwd)
export PYTHONPATH=$(pwd):$PYTHONPATH:/usr/lib/python3.8/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/torch/lib

case "$1" in
    "train")
        CONFIG_NAME=$2
        NUM_EPOCHS=$3
        BATCH_SIZE=$4
        NUM_GPU=$5
        python3  configs/"${CONFIG_NAME}".py --experiment_name ${CONFIG_NAME}_e${NUM_EPOCHS}_b${BATCH_SIZE}_g${NUM_GPU} -d 0-$((NUM_GPU-1)) -b ${BATCH_SIZE} -e ${NUM_EPOCHS} --sync_bn ${NUM_GPU} --no-clearml
        # python3  configs/"${CONFIG_NAME}".py -d 0-3 -b 1 -e ${NUM_EPOCHS} --sync_bn 4 --no-clearml
        ;;
    "test")
        CONFIG_NAME=$2
        CKPT=$3
        python3 configs/"${CONFIG_NAME}".py -d 3 --eval --ckpt "${CKPT}"
        ;;
    "train-continue")
        CONFIG_NAME=$2
        NUM_EPOCHS=$3
        BATCH_SIZE=$4
        NUM_GPU=$5
        CKPT=$6
        python3 configs/"${CONFIG_NAME}".py -d 0-$((NUM_GPU-1)) -b ${BATCH_SIZE} -e 30 --sync_bn ${NUM_GPU} --no-clearml --ckpt "${CKPT}"
        ;;
    "pipeline")
        CONFIG_NAME=$2
        NUM_EPOCHS=$3
        CKPT_ID=$((NUM_EPOCHS-1))
        bash run.sh train ${CONFIG_NAME} ${NUM_EPOCHS}
        bash run.sh test ${CONFIG_NAME} outputs/${CONFIG_NAME}/latest/dump_model/checkpoint_epoch_${CKPT_ID}.pth
        ;;
        
    "reproduce")
        CONFIG_NAME=$2
        bash run.sh pipeline ${CONFIG_NAME} 30
        bash run.sh pipeline ${CONFIG_NAME} 110
        ;;
    *)
        echo "error"
esac
