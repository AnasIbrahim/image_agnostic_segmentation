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
pip install open3d opencv-python argparse os-sys
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
python3 scripts/image_agnostic_segmentation/segment_image.py

# To test your own images
python3 scripts/image_agnostic_segmentation/segment_image.py --image-path IMAGE_PATH
```