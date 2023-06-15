# IMPORTANT !!!

## RGAE

Everyting regarding the default Robust Graph Autoencoder (RGAE) is taken from [Github Link](https://github.com/FGH00292/Hyperspectral-anomaly-detection-with-RGAE). 
This RGAE is based on the papers "Hyperspectral Anomaly Detection With Robust Graph Autoencoders" and "Hyperspectral Anomaly Detection With Robust Graph Autoencoders", both by G. Fan et al.

## KPCA

Everything regarding KPCA is taken from [Github link](https://github.com/xiangyusong19/SSIIFD_Hyperspectral-Anomaly-Detection/tree/main/Demos_full-pixels_detection?fbclid=IwAR16aahWpTO-_kgc1CuVpv9Y1mBGRn716N_U9lbiHi1m2ZSVMDOF14aAD9g).
The authors of this code are X. Song et al.

## Clustering

Everything regarding clustering is take nfrom [Github link](https://github.com/GatorSense/hsi_toolkit).
The code is written by A. Zare et al.

## My contributions

I have only modified certain aspects of the code. These are:
- PCA -> KPCA
- KPCA on Input of RGAE
- Changed layer setup
- No optimizer -> RMSP/Momentum/ADAM


Different optimizers and layer setups must be chosen in the file "RGAE_mat/utils/RGAE/RGAE.m"

## Datasets
Datasets must be placed in the RGAE_mat/datasets/ folder. If you want to use KPCA, you need to save the KPCA of the datasets you want to use in a folder within dim_red/.

