import json
import os
import time
from tqdm import tqdm
import cv2
import numpy as np

import src.globals as g
import src.utils as u
import supervisely as sly
from supervisely.io.fs import get_file_name_with_ext
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    Editor,
    Empty,
    Image,
    SelectItem,
    Text,
    SelectProject,
)

import dataset_tools as dtools

button_stats = Button(text="Calculate")
# button_save = Button(text="Save settings")
# infotext = Text("Settings saved", "success")
# select_item = SelectItem(dataset_id=None, compact=False)

select_item = SelectProject(g.PROJECT_ID, g.WORKSPACE_ID)


card_1 = Card(
    title="Calculate stats",
    content=Container(
        widgets=[
            select_item,
            button_stats,
        ]
    ),
)

# infotext.hide()


@button_stats.click
def calculate() -> None:
    # settings = json.loads(editor.get_text())

    # item_id = select_item.get_selected_id()
    project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(g.PROJECT_ID))
    project_info = g.api.project.get_info_by_id(g.PROJECT_ID)

    project_stats = g.api.project.get_stats(g.PROJECT_ID)
    datasets = g.api.dataset.get_list(g.PROJECT_ID)

    # image = g.api.image.get_info_by_id(item_id)
    # jann = g.api.annotation.download_json(item_id)
    # ann = sly.Annotation.from_json(jann, project_meta)

    # pseudocode
    def smart_cache(anns, dataset):
        forced_anns = []

        anns = g.api.annotation.get_list(dataset.id)
        for ann in anns:
            if change_in_label or new_label_added:
                forced_anns.append(ann)

        return forced_anns

    stat_cache = {}

    stats = [
        dtools.ClassBalance(project_meta, project_stats, stat_cache=stat_cache),
        dtools.ClassCooccurrence(project_meta),
        dtools.ClassesPerImage(
            project_meta, project_stats, datasets, stat_cache=stat_cache
        ),
        dtools.ObjectsDistribution(project_meta),
        dtools.ObjectSizes(project_meta, project_stats),
        dtools.ClassSizes(project_meta),
        # dtools.ClassesTreemap(project_meta),
        # dtools.AnomalyReport(),  # ?
    ]

    srate = 1

    dtools.count_stats(g.PROJECT_ID, project_stats, stats=stats, sample_rate=srate)

    sly.logger.info("Saving stats...")
    for stat in stats:
        sly.logger.info(f"Saving {stat.basename_stem}...")
        if stat.to_json() is not None:
            with open(f"{g.STORAGE_DIR}/{stat.basename_stem}.json", "w") as f:
                json.dump(stat.to_json(), f)

        stat.to_image(f"{g.STORAGE_DIR}/{stat.basename_stem}.png")

    json_paths = sly.fs.list_files(g.STORAGE_DIR, valid_extensions=[".json"])
    dst_paths = [
        f"{g.DST_TF_DIR}/{get_file_name_with_ext(path)}" for path in json_paths
    ]

    pbar = tqdm(desc="Uploading", total=len(json_paths), unit="B", unit_scale=True)
    g.api.file.upload_bulk(g.TEAM_ID, json_paths, dst_paths, pbar)
