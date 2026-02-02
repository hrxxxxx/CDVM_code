# [Causality-Inspired Dual-branch ViT-based Masking for Unsupervised Industrial Anomaly Detection]




## Download Pretrained Weights and Models


Download checkpoints that are **self-supervised pretrained** on ImageNet-22k:
- ViT-base-16: [beit_base_patch16_224_pt22k](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth)

Download pretrained visual tokenizer(discrite VAE) from: [encoder](https://cdn.openai.com/dall-e/encoder.pkl), [decoder](https://cdn.openai.com/dall-e/encoder.pkl), and put them to the directory ``weights/tokenizer``.

Download pretrained protoflow from: [there](https://huggingface.co/hrxxxxx/CDVM/tree/main), and put it to the directory ``weights/protoflow``.


## Setup
Install all packages with this command:
```
$ python3 -m pip install -U -r requirements.txt
```

Download MVTecAD dataset from [there](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/), put it to the directory ``/mvtec_anomaly_detection``, and then run following code to convert this dataset to ImageNet format.

```
python setup_train_dataset.py --data_path /path/to/dataset
```
This script will create a ImageNet format dataset for training at the ``/Mvtec-ImageNet`` directory.Then please download [foreground masks](https://huggingface.co/xcyao00/PMAD/blob/main/fg_mask.zip), and put it to the directory ``data/Mvtec-ImageNet/fg_mask``.

## Training

Run code for training MVTecAD dataset.
```
bash scripts/train_cross_class.sh  // training for cross-class setting
```
For cross-class setting objects-to-textures, please set ``--from_obj_to_texture`` in ``train_cross_class.sh``. If not setted, the code will run cross-class setting textures-to-objects.


## Testing

Run code for testing MVTecAD dataset.
```
bash scripts/test_cross_class.sh  // testing for cross_class setting
bash scripts/test_cross_class.sh  // testing for cross_class setting
```


We summarize the validation results as follows.

Multi-Class Setting:

| Category | Image/Pixel AUC | Category | Image/Pixel AUC | Category | Image/Pixel AUC |
|:------------:|:--------:|:----------:|:-----:|:-----:|:-------:|
| Carpet | 0.998/0.987 | Bottle | 1.000/0.982 | Pill | 0.958/0.935 |
| Grid | 0.982/0.958 | Cable | 0.972/0.964 | Screw | 0.885/0.964 |
| Leather | 1.000/0.991 | Capsule | 0.912/0.975 | Toothbrush | 0.994/0.983 |
| Tile | 1.000/0.951 | Hazelnut | 1.000/0.995 | Transistor | 0.979/0.963 |
| Wood | 1.000/0.911 | Metal nut | 0.991/0.949 | Zipper | 0.989/0.965 |
| Mean | 0.977/0.965 | 
---
Cross-Class Setting(objects-to-textures):

| Category | Image/Pixel AUC | 
|:------------:|:--------:|
| Carpet | 0.996/0.976 | 
| Grid | 0.979/0.935 | 
| Leather | 1.000/0.985 | 
| Tile | 1.000/0.939 | 
| Wood | 0.960/0.887 | 
| Mean | 0.987/0.945 |
---
Cross-Class Setting(textures-to-objects):

| Category | Image/Pixel AUC | Category | Image/Pixel AUC |
|:----------:|:-----:|:-----:|:-------:|
| Bottle | 1.000/0.950 | Pill | 0.839/0.871 |
| Cable | 0.950/0.950 | Screw | 0.631/0.949 |
| Capsule | 0.731/0.953 | Toothbrush | 0.847/0.962 |
| Hazelnut | 1.000/0.962 | Transistor | 0.962/0.863 |
| Metal nut | 0.971/0.844 | Zipper | 0.947/0.947 |
| Mean | 0.927/0.940 | 
---

## Citation

If you find this repository useful, please consider citing our work:
```
@article{CDVM,
      title={Causality-Inspired Dual-branch ViT-based Masking for Unsupervised Industrial Anomaly Detection}, 
      author={Rongxin Huang and Guangfeng Lin and Zhirong Li},
      year={2026},
      journal={IEEE Transactions on Industrial Informatics},
      note={Submitted for publication},
      primaryClass={cs.CV}}
}
```


## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, the [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repository , the [DALL-E](https://github.com/openai/DALL-E) repository and  the [PMAD](https://github.com/xcyao00/PMAD/blob/main/README.md?plain=1) repository.
