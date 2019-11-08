# Quantization Networks

### Overview
This repository contains the training code of Quantization Networks introduced in our CVPR 2019 paper: [*Quantization Networks*](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Quantization_Networks_CVPR_2019_paper.pdf).

In this work, we provide a **simple and uniform way** for weights and activations quantization by formulating it as a differentiable non-linear function.
The quantization function is represented as a linear combination of several
Sigmoid functions with learnable biases and scales that
could be learned in a lossless and end-to-end manner via
continuous relaxation of the steepness of Sigmoid functions.

Extensive experiments on image classification and object
detection tasks show that our quantization networks outperform state-of-the-art methods.

### Run environment

+ Python 3.5
+ Python bindings for OpenCV
+ Pytorch 0.3.0

### Usage

Download the ImageNet dataset and decompress into the structure like

    dir/
      train/
        n01440764_10026.JPEG
        ...
      val/
        ILSVRC2012_val_00000001.JPEG
        ...

To train a weight quantization model of ResNet-18, simply run

    sh quan-weight.sh

After the training, the result model will be stored in `./logs/quan-weight/resnet18-quan-w-1`.

Other training processes can be found in the paper.

### License
+ Apache License 2.0


### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{yang2019quantization,
  title={Quantization Networks},
  author={Yang Jiwei, Shen Xu, Xing Jun, Tian Xinmei, Li Houqiang, Deng Bing, Huang Jianqiang and Hua Xian-sheng},
  booktitle={CVPR},
  year={2019}
}
```
