import json
import os
import math
from typing import List, Literal, Optional

import supervisely as sly
import src.globals as g


def get_project_images_all(project_info):
    images_flat = []
    for dataset in g.api.dataset.get_list(project_info.id):
        images_flat += g.api.image.get_list(dataset.id)
    return images_flat


def get_updated_images(project: sly.ImageInfo, project_meta: sly.ProjectMeta):
    updated_images = []
    images_flat = get_project_images_all(project)

    if g.META_CACHE.get(project.id) is not None:
        if len(g.META_CACHE[project.id].obj_classes) != len(project_meta.obj_classes):
            sly.logger.warn(
                "Changes in the number of classes detected. Recalculate full stats... "  # TODO
            )
            g.META_CACHE[project.id] = project_meta.to_json()
            return images_flat

    g.META_CACHE[project.id] = project_meta.to_json()

    for image in images_flat:
        try:
            image: sly.ImageInfo
            cached_updated_at = g.IMAGES_CACHE[image.id]
            if image.updated_at != cached_updated_at:
                updated_images.append(image)
                g.IMAGES_CACHE[image.id] = image.updated_at
        except KeyError:
            updated_images.append(image)
            g.IMAGES_CACHE[image.id] = image.updated_at

    set_A, set_B = set(g.IMAGES_CACHE), set([image.id for image in images_flat])
    if set_A != set_B:
        sly.logger.warn(
            f"The add/delete operation was detected in the images with the following ids: {set_A.symmetric_difference(set_B)}"
        )
        sly.logger.info("Recalculate full statistics")
        return images_flat

    if len(updated_images) == project.items_count:
        sly.logger.info(f"Full dataset statistics will be calculated.")
    elif len(updated_images) > 0:
        sly.logger.info(f"The changes in {len(updated_images)} images detected")
    return updated_images


def get_indexes_dct(project_id):
    idx_to_infos, infos_to_idx = {}, {}

    for dataset in g.api.dataset.get_list(project_id):
        images_all = g.api.image.get_list(dataset.id)
        images_all = sorted(images_all, key=lambda x: x.id)

        for idx, image_batch in enumerate(sly.batched(images_all, g.CHUNK_SIZE)):
            identifier = f"chunk_{idx}_{dataset.id}_{project_id}"
            for image in image_batch:
                infos_to_idx[image.id] = identifier
            idx_to_infos[identifier] = image_batch

    return idx_to_infos, infos_to_idx


def pull_cache(tf_cache_dir: str):
    if not g.api.file.dir_exists(g.TEAM_ID, tf_cache_dir):
        return

    local_dir = f"{g.STORAGE_DIR}/_cache"

    if sly.fs.dir_exists(local_dir):
        sly.fs.clean_dir(local_dir)
    g.api.file.download_directory(g.TEAM_ID, tf_cache_dir, local_dir)
    files = sly.fs.list_files(local_dir, [".json"])

    for file in files:
        if "meta_cache.json" in file:
            with open(file, "r") as f:
                g.META_CACHE = {
                    int(k): sly.ProjectMeta().from_json(v)
                    for k, v in json.load(f).items()
                }
        if "images_cache.json" in file:
            with open(file, "r") as f:
                g.IMAGES_CACHE = {
                    int(k): v
                    for k, v in json.load(f)
                    .get(str(g.PROJECT_ID), g.IMAGES_CACHE)
                    .items()
                }

    sly.logger.info("The cache was pulled from team files")


def push_cache(tf_cache_dir: str):
    local_cache_dir = f"{g.STORAGE_DIR}/_cache"
    os.makedirs(local_cache_dir, exist_ok=True)

    with open(f"{local_cache_dir}/meta_cache.json", "w") as f:
        json.dump(g.META_CACHE, f)
    with open(f"{local_cache_dir}/images_cache.json", "w") as f:
        json.dump({g.PROJECT_ID: g.IMAGES_CACHE}, f)

    g.api.file.upload_directory(
        g.TEAM_ID,
        local_cache_dir,
        tf_cache_dir,
        change_name_if_conflict=False,
        replace_if_conflict=True,
    )

    sly.logger.info("The cache was pushed to team files")


def check_datasets_consistency(project_info, datasets, npy_paths, num_stats):
    for dataset in datasets:
        actual_ceil = math.ceil(dataset.items_count / g.CHUNK_SIZE)
        max_chunks = math.ceil(
            len(
                [
                    path
                    for path in npy_paths
                    if f"_{dataset.id}_" in sly.fs.get_file_name(path)
                ]
            )
            / num_stats
        )
        if actual_ceil < max_chunks:
            raise ValueError(
                f"The number of chunks per stat ({len(npy_paths)}) not match with the total items count of the project ({project_info.items_count}) using following batch size: {g.CHUNK_SIZE}. Details: DATASET_ID={dataset.id}; actual num of chunks: {actual_ceil}; max num of chunks: {max_chunks}"
            )
    sly.logger.info("The consistency of data is OK")


def remove_junk(project, datasets, files_fs):
    ds_ids, rm_cnt = [str(dataset.id) for dataset in datasets], 0
    for path in files_fs:
        if (path.split("_")[-4] not in ds_ids) or (
            f"_{project.id}_{g.CHUNK_SIZE}_" not in path
        ):
            os.remove(path)
            rm_cnt += 1

    if rm_cnt > 0:
        sly.logger.warn(
            f"The {rm_cnt} old or junk chunk files were detected and removed from the buffer"
        )


def check_idxs_integrity(
    project, stats, curr_projectfs_dir, idx_to_infos, updated_images
):
    if sly.fs.dir_empty(curr_projectfs_dir):
        sly.logger.warn("The buffer is empty. Calculate full stats")
        if len(updated_images) != project.items_count:
            sly.logger.warn(
                f"The number of updated images ({len(updated_images)}) should equal to the number of images ({project.items_count}) in the project. Possibly the problem with cached files. Forcing recalculation..."
            )
            updated_images = get_project_images_all(project)
    else:
        for stat in stats:
            files = sly.fs.list_files(
                f"{curr_projectfs_dir}/{stat.basename_stem}",
                [".npy"],
            )

            if len(files) != len(idx_to_infos.keys()):
                msg = f"The number of images in the project has changed. Check chunks in Team Files: {curr_projectfs_dir}/{stat.basename_stem}"
                sly.logger.error(msg)
                raise RuntimeError(msg)
