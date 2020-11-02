# *EEMEFN: Low-Light Image Enhancement via Edge-Enhanced Multi-Exposure Fusion Network*

## Introduction
This project page provides TensorFlow 1.X code that implements the following AAAI2019 paper:

**Title:** "EEMEFN: Low-Light Image Enhancement via Edge-Enhanced Multi-Exposure Fusion Network"

## How to use
```
conda env create --prefix ./env --file environment.yml
conda activate .env
pip install h5py
pip install rawpy
```
### Download pretrained model
[Sony and Fuji model](https://mega.nz/file/uBUhDCaS#sl_3X-ceBsConxcytAH-FUmBfbaO2Zh6e4X3GIVf24w)
```
# model directory structure
result
|-- Sony_edge
|-- Sony_merge
|-- Fuji_edge
└-- Fuji_merge
```

### Process data
Please download raw dataset from https://github.com/cchen156/Learning-to-See-in-the-Dark  
And delete misalignment images  
Then process data as follows (about two hours)  
```
python dataset.py
```
```
# dataset directory structure
dataset
|-- Fuji
|   |-- fuji_long.hdf5
|   |-- fuji_short.hdf5
|   |-- long
|   └-- short
└-- Sony
    |-- sony_long.hdf5
    |-- sony_short.hdf5
    |-- long
    └-- short

```
### Training
Please modify the name of target dataset first
```
python train_fusion.py
python train_merge.py
```
### Evaluation
```
python test.py
```

### Performance

| Model                     | Sony (PSNR) | Sony (SSIM) | Fuji (PSNR) | Fuji (SSIM) |
|---------------------------|-------------|-------------|-------------|-------------|
| baseline                  | 29.06       | 0.787       | 26.95       | 0.717       |
| MEF                       | 29.43       | 0.791       | 27.21       | 0.719       |
| EEMEFN (paper)            | 29.60       | 0.795       | 27.38       | 0.723       |
| EEMEFN (pretrained model) | 29.60       | 0.795       | 27.43       | 0.723       |

## License
This code is released under the MIT License (refer to the LICENSE file for details).
