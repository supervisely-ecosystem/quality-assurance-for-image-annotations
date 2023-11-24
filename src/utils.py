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


def build_stats(
    self,
    force: Optional[
        List[
            Literal[
                "all",
                "ClassBalance",
                "ClassCooccurrence",
                "ClassesPerImage",
                "ObjectsDistribution",
                "ObjectSizes",
                "ClassSizes",
                "ClassesHeatmaps",
                "ClassesPreview",
                "ClassTreemap",
            ]
        ]
    ] = None,
    settings: dict = {},
):
    sly.logger.info("Starting to build stats...")

    literal_stats = [
        "ClassBalance",
        "ClassCooccurrence",
        "ClassesPerImage",
        "ObjectsDistribution",
        "ObjectSizes",
        "ClassSizes",
        "ClassesHeatmaps",
        "ClassesPreview",
        "ClassesTreemap",
    ]

    if force is None:
        force = []
    elif "all" in force:
        force = literal_stats

    sly.logger.info(f"Following stats are passed with force: {force}")

    cls_prevs_settings = settings.get("ClassesPreview", {})
    heatmaps_settings = settings.get("ClassesHeatmaps", {})
    # previews_settings = settings.get("Previews", {})

    stat_cache = {}
    stats = [
        dtools.ClassBalance(
            self.project_meta, self.project_stats, stat_cache=stat_cache
        ),
        dtools.ClassCooccurrence(self.project_meta),
        dtools.ClassesPerImage(
            self.project_meta, self.project_stats, self.datasets, stat_cache=stat_cache
        ),
        dtools.ObjectsDistribution(self.project_meta),
        dtools.ObjectSizes(self.project_meta, self.project_stats),
        dtools.ClassSizes(self.project_meta),
        dtools.ClassesTreemap(self.project_meta),
    ]
    heatmaps = dtools.ClassesHeatmaps(self.project_meta, self.project_stats)

    if cls_prevs_settings.get("tags") is not None:
        self.classification_task_classes = cls_prevs_settings.pop("tags")

    classes_previews = dtools.ClassesPreview(
        self.project_meta, self.project_info, **cls_prevs_settings
    )
    cls_prevs_settings["tags"] = self.classification_task_classes
    classes_previews_tags = dtools.ClassesPreviewTags(
        self.project_meta, self.project_info, **cls_prevs_settings
    )

    for stat in stats:
        if (
            not sly.fs.file_exists(f"./stats/{stat.basename_stem}.json")
            or stat.__class__.__name__ in force
        ):
            stat.force = True
        if (
            isinstance(stat, dtools.ClassCooccurrence)
            and len(self.project_meta.obj_classes.items()) == 1
        ):
            stat.force = False
    stats = [stat for stat in stats if stat.force]

    vstats = [heatmaps, classes_previews, classes_previews_tags]

    for vstat in vstats:
        if vstat.__class__.__name__ in force:
            vstat.force = True

    if (
        not sly.fs.file_exists(f"./stats/{heatmaps.basename_stem}.png")
        or heatmaps.__class__.__name__ in force
    ):
        heatmaps.force = True
    if (
        not sly.fs.file_exists(
            f"./visualizations/{classes_previews.basename_stem}.webm"
        )
        or classes_previews.__class__.__name__ in force
    ):
        if self.classification_task_classes is None:
            classes_previews.force = True
        else:
            classes_previews_tags.force = True

    vstats = [stat for stat in vstats if stat.force]

    srate = 1
    if settings.get("Other") is not None:
        srate = settings["Other"].get("sample_rate", 1)

    if self.project_stats["images"]["total"]["imagesMarked"] == 0:
        sly.logger.info(
            "This is a classification-only dataset. It has zero annotations. Building only ClassesPreview and Poster."
        )
        if classes_previews_tags.force is not True:
            return
        stats = []
        vstats = [
            vstat for vstat in vstats if isinstance(vstat, dtools.ClassesPreviewTags)
        ]
        heatmaps.force, classes_previews.force, classes_previews_tags.force = (
            False,
            False,
            True,
        )

    dtools.count_stats(
        self.project_id, self.project_stats, stats=stats + vstats, sample_rate=srate
    )

    sly.logger.info("Saving stats...")
    for stat in stats:
        sly.logger.info(f"Saving {stat.basename_stem}...")
        if stat.to_json() is not None:
            with open(f"./stats/{stat.basename_stem}.json", "w") as f:
                json.dump(stat.to_json(), f)
        try:
            stat.to_image(f"./stats/{stat.basename_stem}.png")
        except TypeError:
            pass

    if len(vstats) > 0:
        if heatmaps.force:
            heatmaps.to_image(
                f"./stats/{heatmaps.basename_stem}.png", **heatmaps_settings
            )
        if classes_previews.force:
            classes_previews.animate(
                f"./visualizations/{classes_previews.basename_stem}.webm"
            )
        elif classes_previews_tags.force:  # classification-only dataset
            classes_previews_tags.animate(
                f"./visualizations/{classes_previews.basename_stem}.webm"
            )

    sly.logger.info("Successfully built and saved stats.")
