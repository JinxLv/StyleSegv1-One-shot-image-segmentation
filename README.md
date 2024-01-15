# StyleSeg: One-shot-Segmentation-of-Brain-Tissues-via-Image-aligned-Style-Transformation
This is the implementation of the [StyleSeg]([https://github.com/uncbiag/DeepAtlas](https://ojs.aaai.org/index.php/AAAI/article/view/25276)): Robust One-shot Segmentation of Brain Tissues via Image-aligned Style Transformation

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

## Results
The visualization of our proposed Image-aligned Style Transformation module. The style-transferred atlas w/ IST presents a highly similar appearance with the target unlabeled image, while that w/o IST contains lots of artifacts.

<img src="./Figure/Fourier_vis.png" width="900px">

Boxplots of Dice scores of 35 brain regions for comparison methods on OASIS dataset. The brain regions are presented under the plot and are ranked by the average numbers of the region voxels in decreasing order.

<img src="./Figure/boxplot.png" width="1000px">

The visualization of segmentation results for different dual-model iterative learning methods. From left to right are raw image, UNet trained with 5 atlas, Brainstorm, DeepAtlas, our method, and the ground-truth of segmentation. The implementation of [Brainstorm](https://github.com/xamyzhao/brainstorm) and [DeepAtlas](https://github.com/uncbiag/DeepAtlas) are all used their offical released source code. The Brainstorm, DeepAtlas and our method are all trained with one atlas (labeled image).

<img src="./Figure/visulization.png" width="900px">

Furthermore, we also evaluated the generalization performance of our method on the other modality, i.e., 3D heart CT [MH-WHS 2017](https://zmiclab.github.io/zxh/0/mmwhs/) dataset. One labeled image (atlas) is selected randomly together with the 40 unlabeled images as a training set, the remaining 19 labeled images were utilized as test set. We randomly selected three cases from the segmentation results, and the visualization results are shown below.

<img src="./Figure/heart_ct_1.png" width="500px">

## Citation
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```
@article{Lv_Zeng_Wang_Duan_Wang_Li_2023, number={2}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Lv, Jinxin and Zeng, Xiaoyu and Wang, Sheng and Duan, Ran and Wang, Zhiwei and Li, Qiang}, year={2023}, month={Jun.}, pages={1861-1869} }

```


## Acknowledgment

Some codes are modified from [RCN](https://github.com/microsoft/Recursive-Cascaded-Networks) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph).
Thanks a lot for their great contribution.


