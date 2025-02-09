Overcoming the Stability Gap in Continual Learning
=====================================
This is a PyTorch implementation of the SGM algorithm from our TMLR-2024 paper. An [arXiv version](https://arxiv.org/abs/2306.01904) of our paper is available.


## Dependencies

The conda environment that we used for SGM has been shared in the GitHub repository. 
The yml file `sgmenv.yml` includes all the libraries. We have tested the code with the packages and versions 
specified in the yml file. We recommend setting up a `conda` environment using the `sgmenv.yml` file:
```
conda env create -f sgmenv.yml
```


# Code for Training SGM and Vanilla Models

Python code to reproduce results presented in the paper. The folder `imagenet_files` contains the data ordering files. To use the code, simply run the following:

- To train our proposed SGM model: `train_sgm.sh`

- To train the vanilla model: `train_vanilla.sh`

To get results for other configures, change the relevant arguments.

In our main experiments, we use ImageNet-1K pretrained ConvNeXt V2 Femto model (i.e.,`convnextv2_femto_1k_224_ema.pt` file in the repository).

You can also find pre-trained weights [here](https://github.com/facebookresearch/ConvNeXt-V2).



## Citation

If using this code, please cite our paper.
```
@article{
harun2024overcoming,
title={Overcoming the Stability Gap in Continual Learning},
author={Md Yousuf Harun and Christopher Kanan},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=o2wEfwUOma},
note={}
}
```