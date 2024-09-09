# Two-Stage Fine-tuning CLIP by Introducing Structure Knowledge for Few-shot Classification
This repository contains code for a two-stage fine-tuned multimodal model CLIP on the kilogram dataset and 11 target classification datasets.
# Environment Configuration
We recommend to install the environment through conda and pip. You should make a new environment with python, for example:
```
conda create -n cross_modal python=3.8
```
Next, you can download pytorch from official site
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
# Dataset Installation
In stage Ⅰ, download preprocessed data at: https://huggingface.co/datasets/lil-lab/kilogram/tree/main and create `./data/` to place all the downloaded files. You can also configure the data paths in `./dataloader/data_pathes.py` and `./evaluate.py`.

In stage Ⅱ, the dataset download instruction is modified from official [CoOp repository](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).
# Model Training
## Path Configuration
Default is to save under the current folder. You should modify the paths to dataset and results.
## Stage Ⅰ Fine-tuning
Example for CLIP training:
```
CUDA_VISIBLE_DEVICES=0 \
python3 -m main \
--model_type clip \
--dataset_type part_black \
--optimizer Adam \
--exp_name CLIP_p+b_1_controlled \
--save_folder saved/part+black/model1
```
## Stage Ⅱ Fine-tuning
Import the model weights pre-trained in the stage Ⅰ into：
./

You can use [features.py](features.py) to pre-extract image and text features from a frozen CLIP model. 
```
python features.py --dataset imagenet --train-shot 16 --seed 1 --clip-encoder ViT-B/32 --image-layer-idx 0 --text-augmentation hand_crafted --image-augmentation none --image-views 0
```

To reproduce the experiments in main paper (with flipped view and hand-crafted template), you may run the bash script below to extract for all 11 datasets and 1 seeds.

```
bash features.sh
```
To train model, please refer to [train.py](train.py). 

```
python train.py --modality cross_modal --classifier_head linear --classifier_init zeroshot --logit 4.0 --hyperparams linear --dataset imagenet --train-shot 16 --seed 1 --clip-encoder ViT-B/32 --image-layer-idx 0 --text-augmentation hand_crafted --image-augmentation flip --image-views 1
```

To reproduce the numbers in main paper, please run
```
bash linear_probe.sh
```
To collect results of cross-modal linear probing, please run
```
python eval.py --mode linear --modality cross_modal --classifier_init zeroshot --clip-encoder ViT-B/32 --text-augmentation hand_crafted --image-augmentation flip --image-views 1
```
To reproduce the distribution shift experiments in paper please run [domain_shift.py](domain_shift.py). All the argparse arguments follow that of [train.py](train.py):

```
python domain_shift.py --modality cross_modal --classifier_head linear --classifier_init zeroshot --logit 4.60517 --hyperparams linear --dataset imagenet --train-shot 16 --clip-encoder ViT-B/32 --image-layer-idx 0 --text-augmentation hand_crafted --image-augmentation none --seed 1
```
