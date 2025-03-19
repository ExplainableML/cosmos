# COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training
[![Paper](https://img.shields.io/badge/paper-arxiv.2412.03561-B31B1B.svg)](https://arxiv.org/abs/2412.01814)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FLAIR-FFD700?logo=huggingface&logoColor=yellow)](https://huggingface.co/sankim2/cosmos)


**Authors:** [Sanghwan Kim](https://kim-sanghwan.github.io/), [Rui Xiao](https://www.eml-munich.de/people/rui-xiao), [Mariana-Iuliana Georgescu](https://lilygeorgescu.github.io/), [Stephan Alaniz](https://www.eml-munich.de/people/stephan-alaniz), [Zeynep Akata](https://www.eml-munich.de/people/zeynep-akata)

### Abstract
Vision-Language Models (VLMs) trained with contrastive loss have achieved significant advancements in various vision and language tasks. However, the global nature of contrastive loss makes VLMs focus predominantly on foreground objects, neglecting other crucial information in the image, which limits their effectiveness in downstream tasks. To address these challenges, we propose COSMOS: CrOSs-MOdality Self-distillation for vision-language pre-training that integrates a novel text-cropping strategy and cross-attention module into a self-supervised learning framework. We create global and local views of images and texts (i.e., multi-modal augmentations), which are essential for self-distillation in VLMs. We further introduce a cross-attention module, enabling COSMOS to learn comprehensive cross-modal representations optimized via a cross-modality self-distillation loss. COSMOS consistently outperforms previous strong baselines on various zero-shot downstream tasks, including retrieval, classification, and semantic segmentation. Additionally, it surpasses CLIP-based models trained on larger datasets in visual perception and contextual understanding tasks. 

## Methodology
![](assets/framework.png "An overview of COSMOS")

## Pre-trained Models

We released the pre-trained COSMOS models on [Huggingface](https://huggingface.co/sankim2/cosmos). The pre-trained models, their corresponding pre-trained datasets, R@1 retrieval results on COCO and Flickr, and Top-1 classification results on ImageNet are listed below. For the full results please see the [paper](https://arxiv.org/abs/2412.01814).

| **Checkpoints**                                                                                            | **Architecture** | **Pre-trained Datasets** | **COCO I2T** | **COCO T2I** | **Flickr I2T** | **Flickr T2I** | **IN Top-1** |
|------------------------------------------------------------------------------------------------------------|------------------|--------------|--------------|----------------|----------------|----------------|----------------|
| [cosmos_vitb16_cc3m]         | ViT-B/16 |       CC3M-recap         | 53.1         | 40.1         | 84.1           | 68.6           |37.1           |
| [cosmos_vitb16_cc12m]        | ViT-B/16 | CC12M-recap              | 64.2         | 48.9         | 91.4           | 76.2           |51.4           |
| [cosmos_vitb16_yfcc15m]      | ViT-B/16 | YFCC15M-recap            | 67.5         | 50.9         | 92.6           | 79.6           |52.4           |
| [cosmos_vitb16_merged30m]    | ViT-B/16 | Merged30M                | 68.0         | 52.5         | 92.9           | 80.3           |57.6           |
| [cosmos_vitb16_pixelprose]   | ViT-B/16 | PixelProse               | 62.4         | 43.4         | 89.9           | 73.6           |59.6           |
| [cosmos_vitb32_cc3m]         | ViT-B/32 |  CC3M-recap              | 47.6         | 33.5         | 74.3           | 59.2           |33.0           |
| [cosmos_vitb32_cc12m]        | ViT-B/32 | CC12M-recap              | 59.6         | 43.0         | 86.5           | 69.8           |46.7           |
| [cosmos_vitb32_yfcc15m]      | ViT-B/32 | YFCC15M-recap            | 64.5         | 46.0         | 90.2           | 73.3           |48.1           |
| [cosmos_vitb32_merged30m]    | ViT-B/32 | Merged30M                | 64.3         | 48.4         | 89.9           | 76.1           |53.4           |
| [cosmos_vitb32_pixelprose]   | ViT-B/32 | PixelProse               | 57.2         | 38.9         | 85.6           | 66.3           |54.3           |

## Dependencies
The following small tutorial helps you set up a simple python virtual environment to run our code. Since our main dependency is [OpenCLIP](https://github.com/mlfoundations/open_clip), which is still updated frequently, you could always check their repo for a detailed tutorial on creating an environment that is best suited for your system. A conda environment is also possible with the same Python and PyTorch version.
### 1. Create a Virtual Environment
First, navigate to the project’s root directory `flair/` and create a virtual environment using Python 3.12:
```bash
cd flair/
python3.12 -m venv flair_env
```
### 2. Activate and Navigate to src/
Activate the virtual environment and navigate to `src/`
```bash
source flair_env/bin/activate
cd src/
```

### 3. Install Dependencies
Our code mainly involves installing `open_clip_torch` and `open_clip_torch[training]`.
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Inference Datasets Preparation
Check to prepare all the inference datasets. For clarity, we provide an example datasets folder with annotation files in `datasets/`. However, all datasets don't have to be stored in the same directory, you could specify them freely by changing the arguments in `src/inference.sh`.

## Training FLAIR
In order to train COSMOS from scratch, synthetic long caption datasets should be downloaded from [DreamLIP](https://github.com/ant-research/DreamLIP)'s recaptioned [CC3M-recap](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions), [CC12M-recap](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions), [YFCC15M-recap](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions) and combined(Merged-30M), and [PixelProse](https://huggingface.co/datasets/tomg-group-umd/pixelprose). Notably, COSMOS requires all pre-training dataset to be processed into the [webdataset](https://github.com/webdataset/webdataset) format, to achieve higher I/O efficiency for large-scale training. In the pre-training dataset preparation step, we will take [CC3M-recap](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions) as an example to demonstrate how to prepare the pretraining data. The preparation for other datasets should be similar.


## Qualitative Results

We visualize the attention weights of image and text cross-attention modules. Patch-wise (image) and token-wise (caption) attention weights are both normalized between 0 and 1.

![](assets/qualitative_results_supp.png "Qualitative Results")

## Acknowledgements
We thank [OpenCLIP](https://github.com/mlfoundations/open_clip) for providing the amazing code base. Meanwhile, we acknowledge [DreamLIP](https://github.com/zyf0619sjtu/DreamLIP) and [PixelProse](https://huggingface.co/datasets/tomg-group-umd/pixelprose) for providing us with various pre-training datasets with captions from MLLMs. We are also greateful for [SCLIP](https://github.com/wangf3014/SCLIP) for providing the the detailed scheme for semantic segmentation task.

## Citations
If you find our work useful, please star this repo and cite:

```bibtex
@article{kim2024cosmos,
  title={COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training},
  author={Kim, Sanghwan and Xiao, Rui and Georgescu, Mariana-Iuliana and Alaniz, Stephan and Akata, Zeynep},
  journal={arXiv preprint arXiv:2412.01814},
  year={2024}
}
