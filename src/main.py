from datetime import datetime
import json, os
import numpy as np

import src.globals as g
import src.utils as u
import supervisely as sly
from tqdm import tqdm
import dataset_tools as dtools
from supervisely.io.fs import get_file_name_with_ext
from pathlib import Path

from fastapi import Response

import src.globals as g
import src.utils as u
import supervisely as sly
from src.ui.input import card_1
from supervisely.app.widgets import Container

layout = Container(widgets=[card_1], direction="vertical")

static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)
server = app.get_server()


# @button_stats.click
# def calculate() -> None:
@server.get("/get-stats", response_class=Response)
async def stats_endpoint(project_id: int):
    json_project_meta = g.api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(json_project_meta)
    project_info = g.api.project.get_info_by_id(project_id)

    updated_images = u.get_updated_images(project_info, project_meta)

    project_stats = g.api.project.get_stats(project_id)
    datasets = g.api.dataset.get_list(project_id)

    # def download_json_batches_from_server(project_id, src_dir: str, dst_dir: str):
    #     g.api.file.download_directory(g.TEAM_ID, g.TF_STATS_DIR, g.STORAGE_DIR)
    #     pass  # download to be able to sew

    # g.api.file.download_directory(g.TEAM_ID, g.TF_STATS_DIR, g.STORAGE_DIR)

    cache = {}
    stats = [
        # dtools.ClassBalance(project_meta, project_stats, stat_cache=cache),
        dtools.ClassCooccurrence(project_meta),
        # dtools.ClassesPerImage(project_meta, project_stats, datasets, stat_cache=cache),
        # dtools.ObjectsDistribution(project_meta),
        # dtools.ObjectSizes(project_meta, project_stats),
        # dtools.ClassSizes(project_meta),
        # dtools.ClassesTreemap(project_meta),
        # dtools.AnomalyReport(),  # ?
    ]

    # images_all = []
    # for dataset in g.api.dataset.get_list(project_id):
    #     images_all += g.api.image.get_list(dataset.id)
    # images_all = sorted(images_all, key=lambda x: x.id)

    with tqdm(desc="Calculating stats", total=len(updated_images)) as pbar:
        for dataset in g.api.dataset.get_list(project_id):
            images_all = g.api.image.get_list(dataset.id)
            images_all = sorted(images_all, key=lambda x: x.id)

            images_upd = [
                image for image in updated_images if image.dataset_id == dataset.id
            ]

            idx_to_infos, infos_to_idx = {}, {}
            for idx, image_batch in enumerate(sly.batched(images_all, 2)):
                for image in image_batch:
                    infos_to_idx[image.id] = idx
                idx_to_infos[idx] = image_batch

            idx_upd = list(set([infos_to_idx[image.id] for image in images_upd]))

            for idx in idx_upd:
                batch = idx_to_infos[idx]
                image_ids = [image.id for image in batch]
                datetime_objects = [
                    datetime.fromisoformat(timestamp[:-1])
                    for timestamp in [image.updated_at for image in batch]
                ]
                latest_datetime = sorted(datetime_objects, reverse=True)[0]

                janns = g.api.annotation.download_json_batch(dataset.id, image_ids)
                anns = [
                    sly.Annotation.from_json(ann_json, project_meta)
                    for ann_json in janns
                ]

                for img, ann in zip(batch, anns):
                    for stat in stats:
                        stat.update(img, ann)

                    pbar.update(1)

                for stat in stats:
                    savedir = f"{g.STORAGE_DIR}/{stat.basename_stem}"
                    os.makedirs(savedir, exist_ok=True)

                    np.save(
                        f"{savedir}/chunk_{idx}_{dataset.id}_{project_id}_{latest_datetime.isoformat()}Z.npy",
                        stat.to_numpy_raw(),
                    )

                    stat.clean()

    for stat in stats:
        stat.sew_chunks(chunks_dir=f"{g.STORAGE_DIR}/{stat.basename_stem}/")
        with open(f"{g.STORAGE_DIR}/{stat.basename_stem}.json", "w") as f:
            json.dump(stat.to_json(), f)

        npy_paths = sly.fs.list_files(
            f"{g.STORAGE_DIR}/{stat.basename_stem}", valid_extensions=[".npy"]
        )
        dst_npy_paths = [
            f"{g.TF_STATS_DIR}/{stat.basename_stem}/{get_file_name_with_ext(path)}"
            for path in npy_paths
        ]

        with tqdm(
            desc=f"Uploading {stat.basename_stem} chunks",
            total=sly.fs.get_directory_size(f"{g.STORAGE_DIR}/{stat.basename_stem}"),
            unit="B",
            unit_scale=True,
        ) as pbar:
            g.api.file.upload_bulk(g.TEAM_ID, npy_paths, dst_npy_paths, pbar)

    json_paths = sly.fs.list_files(f"{g.STORAGE_DIR}", valid_extensions=[".json"])
    dst_json_paths = [
        f"{g.TF_STATS_DIR}/{get_file_name_with_ext(path)}" for path in json_paths
    ]

    with tqdm(
        desc=f"Uploading .json stats",
        total=sly.fs.get_directory_size(f"{g.STORAGE_DIR}"),
        unit="B",
        unit_scale=True,
    ) as pbar:
        g.api.file.upload_bulk(g.TEAM_ID, json_paths, dst_json_paths, pbar)
