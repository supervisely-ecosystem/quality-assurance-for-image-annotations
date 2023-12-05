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


def get_updated_images(project_info: sly.ImageInfo, project_meta: sly.ProjectMeta):
    updated_images = []
    images_flat = []

    for dataset in g.api.dataset.get_list(project_info.id):
        images_flat += g.api.image.get_list(dataset.id)

    if g.META_CACHE.get(project_info.id) is not None:
        if len(g.META_CACHE[project_info.id].obj_classes) != len(
            project_meta.obj_classes
        ):
            sly.logger.warn(
                "Changes in the number of classes detected. Recalculate full stats... #TODO"
            )
            g.META_CACHE[project_info.id] = project_meta
            return images_flat
    g.META_CACHE[project_info.id] = project_meta

    for image in images_flat:
        try:
            image: sly.ImageInfo
            cached = g.IMAGES_CACHE[image.id]
            if image.updated_at != cached.updated_at:
                updated_images.append(image)
                g.IMAGES_CACHE[image.id] = image
        except KeyError:
            updated_images.append(image)
            g.IMAGES_CACHE[image.id] = image

    return updated_images


def get_indexes_dct(project_id):
    for dataset in g.api.dataset.get_list(project_id):
        images_all = g.api.image.get_list(dataset.id)
        images_all = sorted(images_all, key=lambda x: x.id)

        idx_to_infos, infos_to_idx = {}, {}

        for idx, image_batch in enumerate(sly.batched(images_all, g.BATCH_SIZE)):
            identifier = f"chunk_{idx}_{dataset.id}_{project_id}"
            for image in image_batch:
                infos_to_idx[image.id] = identifier
            idx_to_infos[identifier] = image_batch

    return idx_to_infos, infos_to_idx
