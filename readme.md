
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

[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yi_Unpaired_Portrait_Drawing_Generation_via_Asymmetric_Cycle_Mapping_CVPR_2020_paper.pdf), [suppl](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yi_Unpaired_Portrait_Drawing_CVPR_2020_supplemental.pdf).
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

- 1. Download pre-trained models from [BaiduYun](https://pan.baidu.com/s/1_9Fy8mRpTQp6AvqhHsfQAQ)(extract code:c9h7) or [GoogleDrive](https://drive.google.com/drive/folders/1FzOcdlMYhvK_nyLCe8wnwotMphhIoiYt?usp=sharing) and rename the folder to `checkpoints`.

- 2. Then generate artistic portrait drawings for example photos in the folder `./examples` using
``` bash
# with GPU
python test_seq_style.py
# without GPU
python test_seq_style.py --gpu -1
```
The test results will be saved to a html file here: `./results/pretrained/test_200/index3styles.html`.
The result images are saved in `./results/pretrained/test_200/images3styles`,
where `real`, `fake1`, `fake2`, `fake3` correspond to input face photo, style1 drawing, style2 drawing, style3 drawing respectively.

<img src = 'imgs/how_to_crop.jpg'>
- 3. To test on your own photos, the photos need to be square (since the program will load it and resized as 512x512). You can use an image editor to crop a square area of your photo that contains face (or use an optional preprocess [here](preprocess/readme.md)). Then specify the folder that contains test photos using `--dataroot`, specify save folder name using `--savefolder` and run the above command again:

``` bash
# with GPU
python test_seq_style.py --dataroot [input_folder] --savefolder [save_folder_name]
# without GPU
python test_seq_style.py --gpu -1 --dataroot [input_folder] --savefolder [save_folder_name]
# E.g.
python test_seq_style.py --gpu -1 --dataroot ./imgs/test1 --savefolder 3styles_test1
```
The test results will be saved to a html file here: `./results/pretrained/test_200/index[save_folder_name].html`.
The result images are saved in `./results/pretrained/test_200/images[save_folder_name]`.

You can contact email yr16@mails.tsinghua.edu.cn for any questions.

## Colab
A colab demo is [here](https://colab.research.google.com/drive/1U1fPXD1JukuKPOrhGMX1iaJC-d8_RUYr).

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
