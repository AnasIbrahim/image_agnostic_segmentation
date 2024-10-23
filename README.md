<h1 align="center">
<img src="./images/dounseen_logo_10.svg" width="300">

Segment & Classify-Anthing for Robotic Grasping
</h1><br>

The DoUnseen package segments and classifies any novel object in just few lines of code. Without any training or fine tuning.

Try it on
<a href="https://huggingface.co/spaces/anas-gouda/dounseen">
HuggingFace
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="50" height="50">
</a>


## Usage modes

1. Standalone
<h1 align="center">
<img src="./images/standalone.png">
</h1><br>

2. Full Segmentation Pipeline (extension to Segment Anything)
<h1 align="center">
<img src="./images/fullpipeline.png">
</h1><br>

## installation

Install dependencies with pip:
```commandline
pip install opencv-python torch torchvision
```

Install Segment Anything 2
```commandline
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```

Install DoUnseen
```commandline
git+https://github.com/AnasIbrahim/image_agnostic_segmentation.git@CASE_release
```

## download pretrained models

TODO: download samv2 and dounseen from huggingface

## How to use 
importing dounseen and setting up the classifier
```python
from dounseen.core import UnseenClassifier
import dounseen.utils as dounseen_utils

unseen_classifier = UnseenClassifier(
        model_path="models/dounseen/vit_b_16_epoch_199_augment.pth",
        gallery_images=None,
        gallery_buffered_path=None,
        augment_gallery=False,
        batch_size=100,
    )
```

1- Standalone mode

TODO

2- Full segmentation pipeline (extension to Segment-Anything)
load SAMv2
```python
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device=DEVICE)
mask_generators[key] = SAM2AutomaticMaskGenerator(model)
```
load and segment the image
```python
image = np.array(image_input.convert("RGB"))
sam2_result = model.generate(image)
```
prepare the output for DoUnseen
```python
# prepare sam2 output for the format expected by DoUnseen
masks = [ann['segmentation'] for ann in sam2_result]
bboxes = [ann['bbox'] for ann in sam2_result]
bboxes = [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] for bbox in bboxes]
# change bboxed from xywh to xyxy
bboxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in bboxes]
segments = dounseen_utils.get_image_segments_from_binary_masks(image, masks, bboxes)
```
To find one object
```python
gallery_dict = {'obj_000001': [np.array(object_image.convert("RGB")) for object_image in [object_image1, object_image2, object_image3, object_image4, object_image5, object_image6]]}
DOUNSEEN_MODEL.update_gallery(gallery_dict)
matched_query, score = DOUNSEEN_MODEL.find_object(segments, obj_name="obj_000001", method="max")
matched_query_ann_image = dounseen_utils.draw_segmented_image(image, [masks[matched_query]], [bboxes[matched_query]], classes_predictions=[0], classes_names=["obj_000001"])
```
To find all gallery objects
```python
# TODO
```


## DoPose Dataset
The unseen object segmentation model was trained with our Dopose data.
The dataset can be downloaded [here](https://zenodo.org/record/6103779).
The dataset is saved in the [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).
It includes multi-view of storage bin (KLT Euro container) and tabletop scenes.
The annotations include RGB and depth images, 6D pose of each object, segmentation masks, COCO json annotations. Also the dataset includes camera transformations between different views of the same scene (this is extra non-standard to BOP format).

Samples from the dataset:
![DoPose dataset sample](images/DoPose.png)

## Papers and Citation

The latest version of DoUnseen is based on the our paper
**Learning Embeddings with Centroid Triplet Loss for Object Identification in Robotic Grasping**
[[Arxiv](https://arxiv.org/abs/2404.06277)].
The model used in DoUnseen is slightly under-trained compared to the model used in the paper.
```
@misc{gouda2024learningembeddingscentroidtriplet,
      title={Learning Embeddings with Centroid Triplet Loss for Object Identification in Robotic Grasping}, 
      author={Anas Gouda and Max Schwarz and Christopher Reining and Sven Behnke and Alice Kirchheim},
      year={2024},
      eprint={2404.06277},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.06277}, 
}
```

A previous version of this repo was based on our original DoUnseen paper
([Arvix](https://arxiv.org/abs/2304.02833)).
The results presented in that paper were barely an improvement due to lack of datasets at that point of time.
```
@misc{gouda2023dounseen,
      title={DoUnseen: Tuning-Free Class-Adaptive Object Detection of Unseen Objects for Robotic Grasping}, 
      author={Anas Gouda and Moritz Roidl},
      year={2023},
      eprint={2304.02833},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Before zero-shot segmentation models like Segment-Anything came out.
This repository offered a similar segmentation method that segmented only household objects.
That was presented and trained using our DoPose dataset.
([Arxiv](https://arxiv.org/abs/2204.13613)).
```
@INPROCEEDINGS{10069586,
  author={Gouda, Anas and Ghanem, Abraham and Reining, Christopher},
  booktitle={2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA)}, 
  title={DoPose-6D dataset for object segmentation and 6D pose estimation}, 
  year={2022},
  volume={},
  number={},
  pages={477-483},
  doi={10.1109/ICMLA55696.2022.00077}}
```


## Latest updates

October 2024: the repo was strongly refactored to be more modular and easier to use
- DoUnseen can be called using few lines of code
- using SAMv2 for segmentation
- Easy installation using pip
- ROS support is removed
- Mask R-CNN model for background removal is removed
- Grasp calculation is removed

Jan 18 2024:
- New classification models were added using ViT and ResNet50 (paper to be added soon)
- classification by calculating centroids of objects was added


### This research is supported by the LAMARR institute
<img src="images/lamarr_logo.png" width="200">