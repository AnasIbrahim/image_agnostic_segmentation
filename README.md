curl -J -O "https://tu-dortmund.sciebo.de/s/qfNQ2vLdCXW8RBS/FAT_trained_Ml2R_bin_fine_tuned.pth"

python3 scripts/image_agnostic_segmentation/segment_image.py --image-path IMAGE_PATH --model-path MODEL_PATH