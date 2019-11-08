CUDA_VISIBLE_DEVICES=0 python quan_weight_main.py -a resnet18 -b 256 -d imagenet \
    --img_size 224 -j 16 --weight-decay 1e-4 \
    --lr 0.001 --temperature 20 \
    --offline_biases resnet18-w-1  \
    --step_size 25 --decay_step 5 --epochs 35 \
    --start_save 0 --print-info 1 \
    --pretrained ./pre-models/resnet18.pth \
    --logs-dir logs/quan-weight/resnet18-quan-w-1
