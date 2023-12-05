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

from fastapi import Response, HTTPException

import src.globals as g
import src.utils as u
import supervisely as sly
from src.ui.input import card_1
from supervisely.app.widgets import Container

layout = Container(widgets=[card_1], direction="vertical")

static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)
server = app.get_server()


@server.get("/get-stats", response_class=Response)
async def stats_endpoint(project_id: int):
    json_project_meta = g.api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(json_project_meta)
    project_info = g.api.project.get_info_by_id(project_id)

    updated_images = u.get_updated_images(project_info, project_meta)

    project_stats = g.api.project.get_stats(project_id)
    datasets = g.api.dataset.get_list(project_id)

    # with open(f"{g.STORAGE_DIR}/meta.json", "w") as f:
    #     json.dump(json_project_meta, f)

    cache = {}
    stats = [
        dtools.ClassBalance(project_meta, project_stats, stat_cache=cache),
        dtools.ClassCooccurrence(project_meta),
        dtools.ClassesPerImage(project_meta, project_stats, datasets, stat_cache=cache),
        dtools.ObjectsDistribution(project_meta),
        dtools.ObjectSizes(project_meta, project_stats),
        dtools.ClassSizes(project_meta),
        dtools.ClassesTreemap(project_meta),
        # dtools.AnomalyReport(),  # ?
    ]

    if len(updated_images) == 0:
        sly.logger.info("Nothing to update. Skipping stats calculation...")
    else:
        idx_to_infos, infos_to_idx = u.get_indexes_dct(project_id)

        if sly.fs.dir_empty(g.STORAGE_DIR):
            sly.logger.info("The buffer is empty. Calculate full stats")
            if len(updated_images) != project_info.items_count:
                raise ValueError(
                    f"The number of updated images ({len(updated_images)}) should equal to the number of images ({project_info.items_count}) in the project."
                )
        else:
            for stat in stats:
                files = sly.fs.list_files(
                    f"{g.STORAGE_DIR}/{stat.basename_stem}", [".npy"]
                )

                if len(files) != len(idx_to_infos.keys()):
                    sly.logger.warn(
                        f"The number of images in the project has changed. Check chunks in Team Files: {g.STORAGE_DIR}/{stat.basename_stem}",
                    )

        tf_all_paths = [
            info.path
            for info in g.api.file.list2(g.TEAM_ID, g.TF_STATS_DIR, recursive=True)
        ]

        with tqdm(desc="Calculating stats", total=len(updated_images)) as pbar:
            for dataset in g.api.dataset.get_list(project_id):
                images_upd = [
                    image for image in updated_images if image.dataset_id == dataset.id
                ]

                pass

                idx_upd = list(set([infos_to_idx[image.id] for image in images_upd]))

                for identifier in idx_upd:
                    batch = idx_to_infos[identifier]
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

                        tf_stat_chunks = [
                            path
                            for path in tf_all_paths
                            if (stat.basename_stem in path) and (identifier in path)
                        ]
                        if len(tf_stat_chunks) > 0:
                            timestamps = [
                                sly.fs.get_file_name(path).split("_")[-1]
                                for path in tf_stat_chunks
                            ]
                            datetime_objects = [
                                datetime.fromisoformat(timestamp)
                                for timestamp in timestamps
                            ]
                            if (
                                latest_datetime
                                > sorted(datetime_objects, reverse=True)[0]
                            ):
                                g.TF_OLD_CHUNKS += [path for path in tf_stat_chunks]
                                for path in sly.fs.list_files(savedir, [".npy"]):
                                    if identifier in path:
                                        os.remove(path)

                        np.save(
                            f"{savedir}/{identifier}_{g.BATCH_SIZE}_{latest_datetime.isoformat()}.npy",
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
                total=sly.fs.get_directory_size(
                    f"{g.STORAGE_DIR}/{stat.basename_stem}"
                ),
                unit="B",
                unit_scale=True,
            ) as pbar:
                g.api.file.upload_bulk(g.TEAM_ID, npy_paths, dst_npy_paths, pbar)

            sly.logger.info(
                f"{stat.basename_stem} chunks: {len(npy_paths)} chunks succesfully uploaded"
            )

        json_paths = sly.fs.list_files(f"{g.STORAGE_DIR}", valid_extensions=[".json"])
        dst_json_paths = [
            f"{g.TF_STATS_DIR}/{get_file_name_with_ext(path)}" for path in json_paths
        ]

        if len(g.TF_OLD_CHUNKS) > 0:
            with tqdm(
                desc=f"Deleting old chunks",
                total=len(g.TF_OLD_CHUNKS),
                unit="B",
                unit_scale=True,
            ) as pbar:
                g.api.file.remove_batch(g.TEAM_ID, g.TF_OLD_CHUNKS, progress_cb=pbar)

            sly.logger.info(f"{len(g.TF_OLD_CHUNKS)} old chunks succesfully deleted")
            g.TF_OLD_CHUNKS = []

        with tqdm(
            desc=f"Uploading .json stats",
            total=sum([sly.fs.get_file_size(path) for path in json_paths]),
            unit="B",
            unit_scale=True,
        ) as pbar:
            g.api.file.upload_bulk(g.TEAM_ID, json_paths, dst_json_paths, pbar)

        sly.logger.info(
            f"{len(json_paths)} updated .json stats succesfully updated and uploaded"
        )
        # sly.fs.remove_dir(g.STORAGE_DIR)
