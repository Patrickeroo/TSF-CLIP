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
