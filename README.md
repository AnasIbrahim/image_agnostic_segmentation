# Image class-agnostic segmentation (Python code and ROS driver)
This repository contains a pipeline that can segment and compute suction grasps of any non-seen objects
using our category-agnostic CNN.


## Example
This is the result of running our CNN on [NVIDIA hope dataset](https://github.com/swtyree/hope-dataset).
The dataset wasn't used during training neither any of its objects.
![results of our CNN on NVIDIA hope dataset](images/HOPE_dataset_example_segmented.png)

## installing dependencies
Install pre-built detectron2 library from [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
And install other dependencies with pip:
```
pip install open3d opencv-python argparse
```

## Python example
To test the model directly without ROS
```
git clone https://github.com/FLW-TUDO/image_agnostic_segmentation.git
mkdir -p image_agnostic_segmentation/models
cd image_agnostic_segmentation/models
wget https://tu-dortmund.sciebo.de/s/ISdLcDMduHeW1ay/download  -O FAT_trained_Ml2R_bin_fine_tuned.pth
cd ../scripts/image_agnostics_segmentation

# to run the example
python segment_image.py

# To test segmentation only with RGB images
python segment_image.py --compute-no-suction-pts --rgb-image-path RGB_IMAGE_PATH

# To segment an image and compute grasps
python segment_image.py --compute-suction-pts --rgb-image-path RGB_IMAGE_PATH --depth-image-path DEPTH_IMAGE_PATH --depth-scale DEPTH_SCALE -c-matrix FX 0.0 CX 0.0 FY CY 0.0 0.0 1.0
```

The examples shows the following scene:
![grasp computation](images/grasp.gif)


## DoPose Dataset
This CNN model is trained with our Dopose data.
The dataset can be downloaded [here](https://zenodo.org/record/6103779).
The dataset is saved in the [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).
It includes multi-view of storage bin (KLT Euro container) and tabletop scenes.
The annotations include RGB and depth images, 6D pose of each object, segmentation masks, COCO json annotations. Also the dataset includes camera transformations between different views of the same scene (this is extra non-standard to BOP format).