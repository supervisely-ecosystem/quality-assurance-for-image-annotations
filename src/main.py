from datetime import datetime
import json, os
import numpy as np
import math
import shutil
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


def main():
    project_id = g.PROJECT_ID

    json_project_meta = g.api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(json_project_meta)
    project = g.api.project.get_info_by_id(project_id)
    project_stats = g.api.project.get_stats(project_id)
    datasets = g.api.dataset.get_list(project_id)

    sly.logger.info(f"Processing for the '{project.name}' with PROJECT_ID={project_id}")
    sly.logger.info(f"with the CHUNK_SIZE={g.CHUNK_SIZE} (images per batch)")
    sly.logger.info(
        f"The project consists of {project.items_count} images and has {len(datasets)} datasets"
    )

    updated_images = u.get_updated_images(project, project_meta)

    if len(updated_images) == 0:
        sly.logger.warn("Nothing to update. Skipping stats calculation...")
        return

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

    curr_projectfs_dir = f"{g.STORAGE_DIR}/{project_id}_{project.name}"
    if sly.fs.dir_exists(curr_projectfs_dir):
        sly.fs.clean_dir(curr_projectfs_dir)
    os.makedirs(curr_projectfs_dir, exist_ok=True)

    curr_tf_project_dir = f"{g.TF_STATS_DIR}/{project_id}_{project.name}"

    if g.api.file.dir_exists(g.TEAM_ID, curr_tf_project_dir) is True:
        total_size = sum(
            [
                g.api.file.get_directory_size(
                    g.TEAM_ID, f"{curr_tf_project_dir}/{stat.basename_stem}"
                )
                for stat in stats
            ]
        )
        with tqdm(
            desc=f"Downloading chunks to buffer",
            total=total_size,
            unit="B",
            unit_scale=True,
        ) as pbar:
            for stat in stats:
                g.api.file.download_directory(
                    g.TEAM_ID,
                    f"{curr_tf_project_dir}/{stat.basename_stem}",
                    f"{curr_projectfs_dir}/{stat.basename_stem}",
                    pbar,
                )

    files_fs = sly.fs.list_files_recursively(
        curr_projectfs_dir, valid_extensions=[".npy"]
    )
    u.check_datasets_consistency(datasets, files_fs)

    for dataset in datasets:
        for path in files_fs:
            if f"_{dataset.id}_{project_id}_{g.CHUNK_SIZE}_" not in path:
                sly.logger.warn(
                    f"The old or junk chunk file detected and removed: '{path}'"
                )
                os.remove(path)

    idx_to_infos, infos_to_idx = u.get_indexes_dct(project_id)

    if sly.fs.dir_empty(curr_projectfs_dir):
        sly.logger.warn("The buffer is empty. Calculate full stats")
        if len(updated_images) != project.items_count:
            raise ValueError(
                f"The number of updated images ({len(updated_images)}) should equal to the number of images ({project.items_count}) in the project."
            )
    else:
        for stat in stats:
            files = sly.fs.list_files(
                f"{curr_projectfs_dir}/{stat.basename_stem}",
                [".npy"],
            )

            if len(files) != len(idx_to_infos.keys()):
                sly.logger.warn(
                    f"The number of images in the project has changed. Check chunks in Team Files: {curr_projectfs_dir}/{stat.basename_stem}",
                )

    tf_all_paths = [
        info.path
        for info in g.api.file.list2(g.TEAM_ID, curr_tf_project_dir, recursive=True)
    ]

    unique_batch_sizes = set(
        [
            sly.fs.get_file_name(path).split("_")[-2]
            for path in tf_all_paths
            if path.endswith(".npy")
        ]
    )

    if (len(unique_batch_sizes) > 1) or (str(g.CHUNK_SIZE) not in unique_batch_sizes):
        g.TF_OLD_CHUNKS += [path for path in tf_all_paths if path.endswith(".npy")]
        sly.logger.info(
            "Chunk batch sizes in team files are non-unique. All chunks will be removed."
        )

    sly.logger.info(f"Start calculating stats for {len(updated_images)} images")
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
                    savedir = f"{curr_projectfs_dir}/{stat.basename_stem}"
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
                        if latest_datetime > sorted(datetime_objects, reverse=True)[0]:
                            g.TF_OLD_CHUNKS += tf_stat_chunks
                            for path in sly.fs.list_files(savedir, [".npy"]):
                                if identifier in path:
                                    os.remove(path)

                    np.save(
                        f"{savedir}/{identifier}_{g.CHUNK_SIZE}_{latest_datetime.isoformat()}.npy",
                        stat.to_numpy_raw(),
                    )
                    stat.clean()

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

    for stat in stats:
        stat.sew_chunks(chunks_dir=f"{curr_projectfs_dir}/{stat.basename_stem}/")
        with open(
            f"{curr_projectfs_dir}/{stat.basename_stem}.json",
            "w",
        ) as f:
            json.dump(stat.to_json(), f)

        npy_paths = sly.fs.list_files(
            f"{curr_projectfs_dir}/{stat.basename_stem}",
            valid_extensions=[".npy"],
        )
        dst_npy_paths = [
            f"{curr_tf_project_dir}/{stat.basename_stem}/{get_file_name_with_ext(path)}"
            for path in npy_paths
        ]

        u.check_datasets_consistency(datasets, npy_paths)

        with tqdm(
            desc=f"Uploading {stat.basename_stem} chunks",
            total=sly.fs.get_directory_size(
                f"{curr_projectfs_dir}/{stat.basename_stem}"
            ),
            unit="B",
            unit_scale=True,
        ) as pbar:
            g.api.file.upload_bulk(g.TEAM_ID, npy_paths, dst_npy_paths, pbar)

        sly.logger.info(
            f"{stat.basename_stem} chunks: {len(npy_paths)} chunks succesfully uploaded"
        )

    json_paths = sly.fs.list_files(curr_projectfs_dir, valid_extensions=[".json"])
    dst_json_paths = [
        f"{curr_tf_project_dir}/{get_file_name_with_ext(path)}" for path in json_paths
    ]

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


if __name__ == "__main__":
    tf_cache_dir = f"{g.TF_STATS_DIR}/_cache"
    u.pull_cache(tf_cache_dir)
    main()
    u.push_cache(tf_cache_dir)
