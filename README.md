# One-shot-Segmentation-of-Brain-Tissues-via-Image-aligned-Style-Transformation
This is the implementation of the paper "Robust One-shot Segmentation of Brain Tissues via Image-aligned Style Transformation"

## Install
The packages and their corresponding version we used in this repository are listed in below.

- Tensorflow==1.15.4
- Keras==2.3.1
- tflearn==0.5.0

## Training
After configuring the environment, please use this command to train the model.

```sh
python train.py -g 0 --batch 1 -d datasets/brain.json -b PCNet -n 1 --round 10000 --epoch 10
```

## Testing
Use this command to obtain the testing results.
```sh
python predict.py -g 0 --batch 1 -d datasets/brain.json -c weights/Apr06-1516
```

## Pre-trained model and testing data on LPBA40
The [pre-trained model](https://drive.google.com/file/d/1NndVW8beu-fYjDP2mVsOf-WRnX2NCZUQ/view?usp=sharing) and [testing data](https://drive.google.com/file/d/1tU42wwc1qLlwJEI3IHcP30XqOtW0j7hb/view?usp=sharing) are available. Please unzip these files, and move the `lpba_val.h5` to `/datasets/` folder.

## Results
<img src="./Figure/boxplot.png" width="800px">
<img src="./Figure/Fourier_vis.png" width="800px">
<img src="./Figure/visulization.png" width="800px">

## Acknowledgment

Some codes are modified from [RCN](https://github.com/microsoft/Recursive-Cascaded-Networks) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph).
Thanks a lot for their great contribution.


