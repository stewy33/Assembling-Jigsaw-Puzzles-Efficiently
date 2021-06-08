# Assembling Jigsaw Puzzles Efficiently
Reimplementing the 2016 paper "Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles" with newer and more efficient architectures.

Original implementation of paper (with AlexNet) based off of [this repository](https://github.com/bbrattoli/JigsawPuzzlePytorch).

## Dependencies
- Python 3
- [Pytorch](https://pytorch.org/)
- [Tensorflow](https://www.tensorflow.org/) for logging and training visualization

## Getting the Data
Note: Unfortunately, [ImageNet](http://image-net.org/) was down for a bit, so I ended up using a torrent and downloading the 2012 Object Detection Training Dataset (full ImageNet was just too large to fit on my disk).

1. [Download the ILSVRC 2012 Object Detection Academic Torrent](http://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2)
(1,281,168 images across 1000 imagenet classes)
2. Then download *ILSVRC2012_img_train.tar* using your favorite torrent client.
3. Extract with `tar -xvf ILSVRC2012_img_train.tar`
4. Extract all sub tars into their own folders with
    ```shell
    for f in n*.tar;
        do mkdir "${f%.tar}";
        tar -xf "$f" -C "${f%.tar}";
    done
    ```
5. Move the new folders into *{repository root}/imagenet/all* and run `python imagenet_train_test_split.py`. Data is now ready!

## Training the Network
Fill the path information in *run_jigsaw_training.sh*. 
IMAGENET_FOLD needs to point to the folder containing ImageNet.

```
./run_jigsaw_training.sh [GPU_ID]
```
or call the python script
```
python JigsawTrain.py [*path_to_imagenet*] --checkpoint [*path_checkpoints_and_logs*] --gpu [*GPU_ID*] --batch [*batch_size*]
```
By default the network uses 1000 permutations with maximum hamming distance selected using *select_permutations.py*.

To change the file name loaded for the permutations, open the file *Dataset/JigsawLoader.py* and change the permutation file in the method *retrieve_permutations*

# Details:
- The input of the network should be 64x64, but it is resized to 75x75,
  otherwise the output of conv5 is 2x2 instead of 3x3 like the official architecture
- Jigsaw trained using the approach of the paper: SGD, LRN layers, 70 epochs
- Implemented data augmentation to discourage learning *shortcuts*: spatial jittering, normalize each patch indipendently, color jittering, 30% black&white image


Results pending...
