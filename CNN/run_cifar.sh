# DATA_PATH: path to the CIFAR-10 dataset directory
DATA_PATH=${DATA_PATH:-./data/cifar10}

python -u distill.py \
    --gpus 0 \
    --model resnet18_1w1a \
    --results_dir ./results/rnt18c10 \
    --data_path ${DATA_PATH} \
    --dataset cifar10 \
    --cl 2 \
    --kd 1 \
    --epochs 1200 \
    --workers 8 \
    --lr 0.1 \
    -b 512 -bt 128 \
    --warm_up \
    --save run_cifar10
