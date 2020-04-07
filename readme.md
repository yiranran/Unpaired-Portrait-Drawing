
# Unpaired Portrait Drawing Generation via Asymmetric Cycle Mapping

We provide PyTorch implementations for our CVPR 2020 paper "Unpaired Portrait Drawing Generation via Asymmetric Cycle Mapping".

This project generates artistic portrait drawings from face photos using a GAN-based model.


## Our Proposed Framework
 
<img src = 'imgs/architecture.png'>

## Sample Results
From left to right: input, output(style1), output(style2), output(style3)
<img src = 'imgs/results.jpg'>

## Citation
If you use this code for your research, please cite our paper.

[paper](https://yiranran.github.io/files/CVPR2020_Unpaired%20Portrait%20Drawing%20Generation%20via%20Asymmetric%20Cycle%20Mapping.pdf), [suppl](https://yiranran.github.io/files/CVPR2020_Unpaired%20Portrait%20Drawing%20Generation%20via%20Asymmetric%20Cycle%20Mapping%20Suppl.pdf).
```
@inproceedings{YiLLR20,
  title     = {Unpaired Portrait Drawing Generation via Asymmetric Cycle Mapping},
  author    = {Yi, Ran and Liu, Yong-Jin and Lai, Yu-Kun and Rosin, Paul L},
  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR '20)},
  year      = {2020}
}
```

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


## Installation
- Install PyTorch 1.1.0 and torchvision from http://pytorch.org and other dependencies (e.g., [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)). You can install all the dependencies by
```bash
pip install -r requirements.txt
```

## Apply a Pre-trained Model

- Download a pre-trained model from [BaiduYun](https://pan.baidu.com/s/1_9Fy8mRpTQp6AvqhHsfQAQ)(extract code:c9h7) or [GoogleDrive](https://drive.google.com/drive/folders/1FzOcdlMYhvK_nyLCe8wnwotMphhIoiYt?usp=sharing) and put it in `checkpoints/pretrained`.

- Then generate artistic portrait drawings for example photos in `examples` using
``` bash
python test_seq_style.py
```
The test results will be saved to a html file here: `./results/pretrained/test_200/indexstylex-x-x.html`.

- You could also test on your photos. The photos need to be square since the program will load it and resized as 512x512. An optional preprocess is [here](preprocess/readme.md). Modify the 5th line in [test_seq_style.py](test_seq_style.py) to your test folder and run the above command again.

You can contact email yr16@mails.tsinghua.edu.cn for any questions.

## Colab
A colab demo is [here](https://colab.research.google.com/drive/1U1fPXD1JukuKPOrhGMX1iaJC-d8_RUYr).

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
