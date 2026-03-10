export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:0' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 \
--rank 0 \
--lr 2e-2 \
--epochs 300 \
--scale 1e-1 \
--b  256 \
--name 'Imagenet_ver0_1e-10lr_200_0.1scale_256' \


