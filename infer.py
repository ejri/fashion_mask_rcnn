from Mask_RCNN.mrcnn.config import Config

from os import listdir
import json
import PIL.Image as Image
from PIL import Image
import numpy as np
from numpy import zeros
from numpy import asarray
from Mask_RCNN.mrcnn.utils import Dataset
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN
from Mask_RCNN.mrcnn.visualize import display_instances
from Mask_RCNN.mrcnn.utils import extract_bboxes

import matplotlib

import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.color as color
import PIL.ImageDraw as ImageDraw

############
###########
### inference

# define a configuration for the model
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "short_sleeve_shirt_cfg"
    BACKBONE = "resnet50"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # background + 1 class

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.1

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.0


# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode="inference", model_dir="./", config=cfg)
# load model weights. Where it was trained in the earlier step.
# the latest mask has the least error, so load that one
model_path = "/Users/ibrahim/Downloads/fashion_mask_rcnn/mask_rcnn_fashion.h5"

# in case of an error, use below
# model.load_weights(model_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model.load_weights(
    model_path,
    by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
)

model.keras_model.summary()
