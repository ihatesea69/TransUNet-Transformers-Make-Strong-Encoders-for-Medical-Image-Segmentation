# Attention-Augmented TransUNet for Synapse Segmentation

PyTorch research repo for extending TransUNet on the Synapse multi-organ segmentation benchmark, with the current focus on CNN-guided attention injection inside the hybrid R50-ViT encoder.

This cleaned version keeps only the research codepath, lightweight utilities, and reproducibility notebooks. AWS/SageMaker and CloudFormation deployment assets were intentionally removed so the repository is easier to read, reproduce, and push to GitHub.

## Research focus

This repo currently centers on two attention variants added on top of the hybrid ResNet-50 + ViT-B/16 encoder:

- `pre_hidden`: refine selected CNN scales and fuse them into the hidden feature before patch projection
- `cnn_fusion`: refine selected CNN skip features and fuse multiple CNN scales back into the hidden feature

## Current result snapshot

Latest evaluated attention run:

- Variant: `cnn_fusion`
- Scales: `1/8,1/4,1/2`
- Mean Dice: `76.61%`
- Mean HD95: `28.80`

Reference baseline from the earlier cleaned reproduction:

- Mean Dice: `77.29%`
- Mean HD95: `30.71`

### Comparison with the original TransUNet paper

| Framework | Encoder | Decoder | Average DSC ↑ | HD ↓ | Aorta | Gallbladder | Kidney (L) | Kidney (R) | Liver | Pancreas | Spleen | Stomach |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (`cnn_fusion`, `1/8,1/4,1/2`)** | R50-ViT | CUP | 76.61 | **28.80** | 86.72 | 57.15 | 79.33 | 75.14 | **94.40** | **57.36** | **86.54** | **76.20** |
| TransUNet (paper) | R50-ViT | CUP | **77.48** | 31.69 | **87.23** | **63.13** | **81.87** | **77.02** | 94.08 | 55.86 | 85.08 | 75.62 |

Bold values indicate the better score between the current attention run and the original paper row. The current `cnn_fusion` run improves HD95 and four organ-wise Dice scores (`Liver`, `Pancreas`, `Spleen`, `Stomach`) while still trailing the original paper on mean Dice.

Relevant implementation files:

- [networks/vit_seg_modeling.py](networks/vit_seg_modeling.py)
- [networks/vit_seg_modeling_resnet_skip.py](networks/vit_seg_modeling_resnet_skip.py)
- [experiment_utils.py](experiment_utils.py)
- [train.py](train.py)
- [test.py](test.py)

## Repository layout

```text
datasets/          dataset package and Synapse loader
splits/            explicit train/test split metadata
networks/          TransUNet model + hybrid encoder attention modules
notebooks/         Colab notebooks for Drive bootstrap and end-to-end experiments
train.py           training entrypoint
test.py            evaluation entrypoint
trainer.py         training loop with epoch-level resume checkpointing
```

## Environment

Recommended:

- Python 3.10 to 3.12
- CUDA-enabled PyTorch
- `pip install -r requirements.txt`

Main Python dependencies are tracked in [requirements.txt](requirements.txt). PyTorch and torchvision should match your CUDA runtime.

## Data

The repo expects preprocessed Synapse data in:

```text
data/
  Synapse/
    train_npz/
    test_vol_h5/
```

Recommended workflow:

- use [notebooks/transunet-drive-data-setup.ipynb](notebooks/transunet-drive-data-setup.ipynb) to cache the dataset to Google Drive for Colab
- or prepare the Synapse layout manually under `data/Synapse`

## Pretrained weights

The hybrid encoder expects the R50-ViT-B/16 ImageNet-21k checkpoint under:

```text
model/vit_checkpoint/imagenet21k/
  R50+ViT-B_16.npz
  R50-ViT-B_16.npz
```

Recommended workflow:

- use [notebooks/transunet-drive-data-setup.ipynb](notebooks/transunet-drive-data-setup.ipynb) to cache the pretrained weight to Google Drive for Colab
- or place the checkpoint manually under `model/vit_checkpoint/imagenet21k/`

Both filename aliases are supported because different codepaths and notebooks reference both forms.

## Training

Example: run the current research default (`cnn_fusion` on `1/8,1/4,1/2`)

```bash
python train.py ^
  --dataset Synapse ^
  --vit_name R50-ViT-B_16 ^
  --attention_mode cnn_fusion ^
  --attention_scales 1/8,1/4,1/2
```

Alternative attention experiment:

```bash
python train.py ^
  --dataset Synapse ^
  --vit_name R50-ViT-B_16 ^
  --attention_mode pre_hidden ^
  --attention_scales 1/8
```

Baseline ablation:

```bash
python train.py --dataset Synapse --vit_name R50-ViT-B_16 --attention_mode none
```

## Evaluation

```bash
python test.py ^
  --dataset Synapse ^
  --vit_name R50-ViT-B_16 ^
  --attention_mode cnn_fusion ^
  --attention_scales 1/8,1/4,1/2
```

Save NIfTI predictions:

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16 --is_savenii
```

## Colab notebooks

For reproducibility on Google Colab:

- [notebooks/transunet-drive-data-setup.ipynb](notebooks/transunet-drive-data-setup.ipynb): prepare the Synapse dataset and pretrained TransUNet weight on Google Drive
- [notebooks/transunet-cnn-attention-research-colab.ipynb](notebooks/transunet-cnn-attention-research-colab.ipynb): run the TransUNet CNN-attention experiment end-to-end on Colab with live logs and checkpoint resume

## Notes

- `trainer.py` saves `latest_checkpoint.pth` every epoch and can resume automatically.
- Package markers were added to `datasets/` and `networks/` so Colab does not confuse them with third-party packages.
- The repo intentionally no longer contains AWS deployment code, CloudFormation templates, or SageMaker helpers.

## Citation

If you use this repo, cite the original TransUNet work and document the attention extension separately in your report or paper.

```bibtex
@article{chen2024transunet,
  title={TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers},
  author={Chen, Jieneng and Mei, Jieru and Li, Xianhang and Lu, Yongyi and Yu, Qihang and Wei, Qingyue and Luo, Xiangde and Xie, Yutong and Adeli, Ehsan and Wang, Yan and others},
  journal={Medical Image Analysis},
  pages={103280},
  year={2024}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
