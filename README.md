# BiEfficient: Bidirectionally Prompting Vision-Language Models for Parameter-Efficient Video Recognition
## Environment
- Python=3.9
- PyTorch=2.0.0
- RandAugment
- tqdm
- dotmap
- decord
- timm
- regex
- ftfy
- einops
## Configuration
Some common configurations (e.g., dataset path, pretrained backbone path) are set in `config.yaml`. We have included example configs in `configs\`.
## Data Preparation
We decoe the videos in on online fashion using `decord`. Please refer to [PaddlePaddle](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/dataset/README.md) repo for the detailed guide of dataset processing.
## Training
```sh
# Finetuning on Kinetics400 using the following command:
sh scripts/run_train.sh configs/k400/k400_train_vitb_16_fa.yaml
```
## Test
```sh
# Run the following command to test the model:
sh scripts/run_test.sh configs/k400/k400_train_vitb_16_fa.yaml your_trained_model.pt
```
## Acknowledgements
This repository is built based on [CLIP](https://github.com/openai/CLIP) and [BIKE](https://github.com/whwu95/BIKE). Sincere thanks to their wonderful works.

