# One-shot-Segmentation-of-Brain-Tissues-via-Image-aligned-Style-Transformation
This is the implementation of the paper "Robust One-shot Segmentation of Brain Tissues via Image-aligned Style Transformation"

## Install
The packages and their corresponding version we used in this repository are listed in below.

- Tensorflow==1.15.4
- Keras==2.3.1
- tflearn==0.5.0

## Training
After configuring the environment, please use this command to train the model. The training process consists of unsupervised training of reg-model (reg0), supervised training of seg-model with image transferred by IST (seg0), and the weakly supervised training of reg-model (reg1).

```sh
python train.py --lr 1e-4  -d ./dataset/OASIS.json -c weights/xxx --clear_steps -g 0 --round 2000 --scheme reg #reg0
python train.py --lr 1e-3  -d ./dataset/OASIS.json -c weights/xxx --clear_steps -g 0 --round 2000 --scheme seg #seg0
python train.py --lr 1e-4  -d ./dataset/OASIS.json -c weights/xxx --clear_steps -g 0 --round 2000 --scheme reg_supervise #reg1
```

## Testing
Use this command to obtain the final segmentation results of test data.
```sh
python predict.py -c weights/xxx -d ./datasets/OASIS.json -g 0 --scheme seg
```

## Pre-processed dataset
We provided the processed two brain MRI dataset, i.e., [OASIS](https://drive.google.com/file/d/124AIYL2Qt39wiZV15s0RIkmU76AJcuIh/view?usp=sharing) and [CANDIShare](https://drive.google.com/file/d/1zjHp6pV_pRYFzW2lDyKaLZFg0S5cgplk/view?usp=sharing), and a heart CT dataset, i.e., [MH-WHS 2017](https://drive.google.com/file/d/194iZ9jHumUwUoscsD84d0kufCjJR6kfc/view?usp=sharing). Please unzip these files, and move them to `/datasets/` folder.

## Pre-trained model
The pre-trained model for OASIS, CANDIShare, MH-WHS 2017 are also [available](https://drive.google.com/file/d/1zpZdjuXgX-VWuTRNyqSAAH0x-YPMdLry/view?usp=sharing). Please unzip and put them into the `/weights/` folder.

## Results
The visualization of our proposed Image-aligned Style Transformation module. The style-transferred atlas w/ IST presents a highly similar appearance with the target unlabeled image, while that w/o IST contains lots of artifacts.

<img src="./Figure/Fourier_vis.png" width="900px">

Boxplots of Dice scores of 35 brain regions for comparison methods on OASIS dataset. The brain regions are presented under the plot and are ranked by the average numbers of the region voxels in decreasing order.

<img src="./Figure/boxplot.png" width="1000px">

The visualization of segmentation results for different dual-model iterative learning methods. From left to right are raw image, UNet trained with 5 atlas, Brainstorm, DeepAtlas, our method, and the ground-truth of segmentation. The implementation of [Brainstorm](https://github.com/xamyzhao/brainstorm) and [DeepAtlas](https://github.com/uncbiag/DeepAtlas) are all used their offical released source code. The Brainstorm, DeepAtlas and our method are all trained with one atlas (labeled image).

<img src="./Figure/visulization.png" width="900px">

Furthermore, we also evaluated the generalization performance of our method on the other modality, i.e., 3D heart CT [MH-WHS 2017](https://zmiclab.github.io/zxh/0/mmwhs/) dataset. We randomly selected three cases from the segmentation results, and visualized them below.

<img src="./Figure/heart_ct.png" width="500px">

## Acknowledgment

Some codes are modified from [RCN](https://github.com/microsoft/Recursive-Cascaded-Networks) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph).
Thanks a lot for their great contribution.


