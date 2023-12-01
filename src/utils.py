import json
import os
import random
import re
import shutil
import time
from typing import List, Literal, Optional

import cv2
import requests
import supervisely as sly
import tqdm
from dotenv import load_dotenv
from PIL import Image
from supervisely._utils import camel_to_snake
from supervisely.io.fs import archive_directory, get_file_name, mkdir

import dataset_tools as dtools
from dataset_tools.repo import download
from dataset_tools.repo.sample_project import (
    download_sample_image_project,
    get_sample_image_infos,
)
from dataset_tools.templates import DatasetCategory, License
from dataset_tools.text.generate_summary import list2sentence
import src.globals as g


# pseudocode
def get_updated_images(project_info, project_meta):
    updated_images = []
    images_flat = []

    # anns = g.api.annotation.get_list(dataset.id)

    for dataset in g.api.dataset.get_list(project_info.id):
        images_flat += g.api.image.get_list(dataset.id)

    for image in images_flat:
        try:
            image: sly.ImageInfo
            cached = g.IMAGES_CACHE[image.id]
            if image.updated_at != cached.updated_at:
                updated_images.append(image)
        except KeyError:
            g.IMAGES_CACHE[image.id] = image
            updated_images.append(image)

    # for ann in anns:
    #     if change_in_label or new_label_added:
    #         forced_anns.append(ann)

    return updated_images
