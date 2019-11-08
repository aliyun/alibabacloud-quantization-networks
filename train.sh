CUDA_VISIBLE_DEVICES=0,1 python main.py -a resnet18 -b 256 -d imagenet \
    --img_size 224 -j 16 --weight-decay 1e-4 --lr 0.1 \
    --step_size 40 --decay_step 25 --epochs 110 \
    --start_save 0 --print-info 1 \
    --logs-dir logs/baseline/resnet18-relu6-low-aug
    #--resume logs/baseline/alexnet/epoch_79.pth.tar \
