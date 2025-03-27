# [CVPR 2025] COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training
[![Paper](https://img.shields.io/badge/paper-arxiv.2412.03561-B31B1B.svg)](https://arxiv.org/abs/2412.01814)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-COSMOS-FFD700?logo=huggingface&logoColor=yellow)](https://huggingface.co/sankim2/cosmos)


**Authors:** [Sanghwan Kim](https://kim-sanghwan.github.io/), [Rui Xiao](https://www.eml-munich.de/people/rui-xiao), [Mariana-Iuliana Georgescu](https://lilygeorgescu.github.io/), [Stephan Alaniz](https://www.eml-munich.de/people/stephan-alaniz), [Zeynep Akata](https://www.eml-munich.de/people/zeynep-akata)

### Abstract
Vision-Language Models (VLMs) trained with contrastive loss have achieved significant advancements in various vision and language tasks. However, the global nature of contrastive loss makes VLMs focus predominantly on foreground objects, neglecting other crucial information in the image, which limits their effectiveness in downstream tasks. To address these challenges, we propose COSMOS: CrOSs-MOdality Self-distillation for vision-language pre-training that integrates a novel text-cropping strategy and cross-attention module into a self-supervised learning framework. We create global and local views of images and texts (i.e., multi-modal augmentations), which are essential for self-distillation in VLMs. We further introduce a cross-attention module, enabling COSMOS to learn comprehensive cross-modal representations optimized via a cross-modality self-distillation loss. COSMOS consistently outperforms previous strong baselines on various zero-shot downstream tasks, including retrieval, classification, and semantic segmentation. Additionally, it surpasses CLIP-based models trained on larger datasets in visual perception and contextual understanding tasks. 

## Methodology
![](assets/framework.png "An overview of COSMOS")

## Pre-trained Model Weights

We released the pre-trained COSMOS models on [Huggingface](https://huggingface.co/sankim2/cosmos). Our pre-trained models and their corresponding performances on COCO (I2T R@1 and T2I R@1), Flickr (I2T R@1 and T2I R@1) and ImageNet (Top-1) are reported below. For the full results, please refer to our [paper](https://arxiv.org/abs/2412.01814).

| **Checkpoints**                                                                                            | **Arch.** | **Datasets** | **COCO I2T** | **COCO T2I** | **Flickr I2T** | **Flickr T2I** | **IN Top-1** |
|------------------------------------------------------------------------------------------------------------|------------------|--------------|--------------|----------------|----------------|----------------|----------------|
| [cosmos_vitb16_cc3m](https://huggingface.co/sankim2/cosmos/resolve/main/cosmos_vitb16_cc3m.pt?download=true)         | ViT-B/16 |       CC3M-recap         | 53.1         | 40.1         | 84.1           | 68.6           |37.1           |
| [cosmos_vitb16_cc12m](https://huggingface.co/sankim2/cosmos/resolve/main/cosmos_vitb16_cc12m.pt?download=true)        | ViT-B/16 | CC12M-recap              | 64.2         | 48.9         | 91.4           | 76.2           |51.4           |
| [cosmos_vitb16_yfcc15m](https://huggingface.co/sankim2/cosmos/resolve/main/cosmos_vitb16_yfcc15m.pt?download=true)      | ViT-B/16 | YFCC15M-recap            | 67.5         | 50.9         | 92.6           | 79.6           |52.4           |
| [cosmos_vitb16_merged30m](https://huggingface.co/sankim2/cosmos/resolve/main/cosmos_vitb16_merged30m.pt?download=true)    | ViT-B/16 | Merged30M                | 68.0         | 52.5         | 92.9           | 80.3           |57.6           |
| [cosmos_vitb16_pixelprose](https://huggingface.co/sankim2/cosmos/resolve/main/cosmos_vitb16_pixelprose.pt?download=true)   | ViT-B/16 | PixelProse               | 62.4         | 43.4         | 89.9           | 73.6           |59.6           |
| [cosmos_vitb32_cc3m](https://huggingface.co/sankim2/cosmos/resolve/main/cosmos_vitb32_cc3m.pt?download=true)         | ViT-B/32 |  CC3M-recap              | 47.6         | 33.5         | 74.3           | 59.2           |33.0           |
| [cosmos_vitb32_cc12m](https://huggingface.co/sankim2/cosmos/resolve/main/cosmos_vitb32_cc12m.pt?download=true)        | ViT-B/32 | CC12M-recap              | 59.6         | 43.0         | 86.5           | 69.8           |46.7           |
| [cosmos_vitb32_yfcc15m](https://huggingface.co/sankim2/cosmos/resolve/main/cosmos_vitb32_yfcc15m.pt?download=true)      | ViT-B/32 | YFCC15M-recap            | 64.5         | 46.0         | 90.2           | 73.3           |48.1           |
| [cosmos_vitb32_merged30m](https://huggingface.co/sankim2/cosmos/resolve/main/cosmos_vitb32_merged30m.pt?download=true)    | ViT-B/32 | Merged30M                | 64.3         | 48.4         | 89.9           | 76.1           |53.4           |
| [cosmos_vitb32_pixelprose](https://huggingface.co/sankim2/cosmos/resolve/main/cosmos_vitb32_pixelprose.pt?download=true)   | ViT-B/32 | PixelProse               | 57.2         | 38.9         | 85.6           | 66.3           |54.3           |

⚠️ You don't need to manually download the pre-trained weights to run the inference, the pre-trained weights will be automatically downloaded by specifying the `--huggingface-model-name` and `--huggingface-repo-name` during inference. 
Optionally, you could download each weight separately and set  `--resume path/to/pretrained_weights` flag in inference code.

## Dependencies
You can set up your virtual environment following the below instructions. We built our code repository upon [OpenCLIP](https://github.com/mlfoundations/open_clip), which is still updated frequently. We recommend you to check their repo for a detailed tutorial on creating an environment that is best suited for your system. A conda environment is also possible with the same Python and PyTorch version.

### 1. Download our github Repository
First, download the COSMOS github repo and navigate to the project’s root directory `cosmos/`.
```bash
git clone https://github.com/ExplainableML/cosmos.git
cd cosmos/
```

### 2. Create a Virtual Environment
Create a virtual environment using Python 3.12 and activate the virtual environment.
```bash
python3.12 -m venv cosmos_env
source cosmos_env/bin/activate
```

### 3. Install Dependencies
Install all requirements via pip.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
If you want to conduct semantic segmentation tasks, please follow [SCLIP](https://github.com/wangf3014/SCLIP) to install their dependencies as well. We wrote down their command below for completeness.
```bash
pip install openmim
mim install mmcv==2.0.1 mmengine==0.8.4 mmsegmentation==1.1.1
pip install ftfy regex yapf==0.40.1
```

### [Optional] Anaconda Environment
One can optionally use anaconda to set up the environment.
```bash
conda create --name cosmos_env python=3.12
conda activate cosmos_env
```
Then, install all dependencies as follows.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Inference Datasets Preparation
Check [datasets/README.md](datasets/README.md) to prepare all the inference datasets for retrieval, classification, and segmentation tasks.

## Inference with COSMOS
To reproduce the results of downstream tasks (image-text retrieval, image classification, semantic segmentation) in the COSMOS paper, we provide an example inference bash script for each task: `src/inference_retrieval.sh`, `src/inference_classification.sh`, and `src/inference_segmentation.sh`.

Here are detailed explanations of important flags.


- `--huggingface-repo-name`: Name of the Huggingface repo where the pre-trained models are stored. Should be fixed as `sankim2/cosmos`.
- `--huggingface-model-name`: Name of the pretrained models. Options include `cosmos_vitb16_cc3m.pt, cosmos_vitb16_cc12m.pt, cosmos_vitb16_yfcc15m.pt, cosmos_vitb16_merged30m.pt, cosmos_vitb16_pixelprose.pt` for ViT-B/16 and `cosmos_vitb32_cc3m.pt, cosmos_vitb32_cc12m.pt, cosmos_vitb32_yfcc15m.pt, cosmos_vitb32_merged30m.pt, cosmos_vitb32_pixelprose.pt` for ViT-B/32.
- `--model`: Model architecture should be matched with `--huggingface-model-name`. Options include `ViT-B-16` and `ViT-B-32`.
- `--precision`: Defualt as `amp` in our paper.
- `--workers`: Adjustable according to your system.

### Image-Text Retrieval Task
`--data-root-dir` should denote your directory which contains COCO and Flickr30k validation set. Please refer to [/src/inference_retrieval.sh](/src/inference_retrieval.sh) for running inference on retrieval task.

### Image Classification Task
`--imagenet-val` should denote your directory which contains ImageNet validation set. Please refer to [/src/inference_classification.sh](/src/inference_classification.sh) for running inference on classification task.

### Semantic Segmentation Task
`--seg-w-background` denotes a flag whether to evaluate on segmentation benchmarks with background. If `--use-csa` is included, the model will use Correlative Self-Attention (CSA) block from SCLIP for segmentation. Please refer to [/src/inference_segmentation.sh](/src/inference_segmentation.sh) for running inference on segmentation task.

## Training COSMOS
In order to train COSMOS from scratch, synthetic long caption datasets should be downloaded from [DreamLIP](https://github.com/ant-research/DreamLIP)'s recaptioned [CC3M-recap](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions), [CC12M-recap](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions), [YFCC15M-recap](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions) and combined (Merged-30M), and [PixelProse](https://huggingface.co/datasets/tomg-group-umd/pixelprose). Notably, COSMOS requires all pre-training dataset to be processed into the [webdataset](https://github.com/webdataset/webdataset) format, to achieve higher I/O efficiency for large-scale training. In the pre-training dataset preparation step, we take [CC3M-recap](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions) as an example to demonstrate how to prepare the pretraining data. The preparation for other datasets should be similar. We share the same pre-training dataset as [FLAIR](https://arxiv.org/abs/2412.03561). Please check their [repo](https://github.com/ExplainableML/flair) as well if you find it interesting!

### Prepare Pre-training Data
1. Download DreamLIP's annotations for CC3M-recap:
`wget https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/resolve/main/cc3m_3long_3short_1raw_captions_url.csv`
2. Scrape the images based on the url links using [img2dataset](https://github.com/rom1504/img2dataset).

### Training Script
COSMOS is trained with Slurm GPU Cluster on 16 NVIDIA A100s 40GB (on CC3M) or 128 NVIDIA A100s 40GB (on other larger datasets). In `src/`, we provide example slurm training scripts for each of the datasets: `train_cc3m.sh, train_cc12m.sh, train_yfcc15m.sh, train_merged30m.sh, train_pixelprose.sh`. 

Important flags are described below:
   - `--train-data`: Root dir of where the training data (shards) is stored.
   - `--train-num-samples`: the total number of training samples. This should be adjustable based on your available data.
   - `--use-imagecrop-aug`: Using multi-crop image augmentation described in the paper. 
   - `--global-crops-number`: Number of global crop of image. Fixed as 2.
   - `--local-crops-number`: Number of local crop of image. 
   - `--crop-scale`: Determine the scale <i>s</i> of global and local crop images. (0.05, s) for local crops and (s, 1.0) for global crops. Fixed as 0.4
   - `--caption-sampling-mode`: Determine how captions are sampled. Fixed as `textcrop` or `textcrop_pixelprose`.
   - `--num-sampled-captions`: Total number of captions (global+local)
   - `--momentum-teacher`: Initial momentum value. This should be adjusted based on batch size. We used 0.999 for 1k batch and 0.99 for 4k batch.
   - `--fix-momentum`: Fix momentum value during training.
   - `--output-all`: Output both patch (or word) tokens and [cls] (or [eot]) tokens.
   - `--attentional-pool`: Set cross-attention module in model.
   - `--cosmos`: Use COSMOS loss during training.

## Qualitative Results
We visualize the attention weights of image and text cross-attention modules. Patch-wise (image) and token-wise (caption) attention weights are both normalized between 0 and 1.

![](assets/qualitative_results_supp.png "Qualitative Results")

## Acknowledgements
We thank [OpenCLIP](https://github.com/mlfoundations/open_clip) for providing the amazing code base. Meanwhile, we acknowledge [DreamLIP](https://github.com/zyf0619sjtu/DreamLIP) and [PixelProse](https://huggingface.co/datasets/tomg-group-umd/pixelprose) for providing us with various pre-training datasets with captions from MLLMs. We are also grateful to [SCLIP](https://github.com/wangf3014/SCLIP) for providing the detailed scheme for semantic segmentation task.

## Citations
If you find our work useful, please star this repo and cite:

```bibtex
@article{kim2025cosmos,
  title={COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training},
  author={Kim, Sanghwan and Xiao, Rui and Georgescu, Mariana-Iuliana and Alaniz, Stephan and Akata, Zeynep},
  journal={CVPR},
  year={2025}
}
```
