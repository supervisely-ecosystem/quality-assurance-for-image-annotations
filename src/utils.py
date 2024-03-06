import json
import os
import math
from typing import List, Literal, Optional, Dict, Tuple
from datetime import datetime

from supervisely import ImageInfo, ProjectMeta, ProjectInfo, DatasetInfo

from tqdm import tqdm
import supervisely as sly
import src.globals as g
import numpy as np

from supervisely.io.fs import (
    get_file_name_with_ext,
    get_file_name,
    list_files,
    get_file_size,
    list_files_recursively,
)


def pull_cache(
    team_id: int, project_id: int, tf_cache_dir: str, curr_tf_project_dir: str
) -> bool:
    force_stats_recalc = False

    local_cache_dir = f"{g.STORAGE_DIR}/_cache"
    if sly.fs.dir_exists(local_cache_dir):
        sly.fs.clean_dir(local_cache_dir)

    if not g.api.file.dir_exists(team_id, tf_cache_dir):
        sly.logger.warning("The cache directory not exists in team files. ")
        return False

    if not g.api.file.dir_exists(team_id, curr_tf_project_dir):
        sly.logger.warning("The project directory not exists in team files.")
        return False

    g.api.file.download_directory(team_id, tf_cache_dir, local_cache_dir)

    spath = f"{local_cache_dir}/project_statistics_meta.json"
    if os.path.exists(spath):
        with open(spath, "r", encoding="utf-8") as f:
            stats_meta = json.load(f)
        smeta = stats_meta.get(str(project_id))
        if smeta is not None:
            if smeta["chunk_size"] != g.CHUNK_SIZE:
                sly.logger.warning(
                    "The chunk size has changed. Recalculating full stats..."
                )
                return True

    path = os.path.join(local_cache_dir, "meta_cache.json")
    if os.path.exists(os.path.join(local_cache_dir, "meta_cache.json")):
        with open(path, "r", encoding="utf-8") as f:
            g.META_CACHE = {
                int(k): sly.ProjectMeta().from_json(v) for k, v in json.load(f).items()
            }
    else:
        sly.logger.info(
            "The 'meta_cache.json' file not exists. Stats will be recalculated."
        )
        force_stats_recalc = True

    path = os.path.join(local_cache_dir, "images_cache.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            g.IMAGES_CACHE = json.load(f)
            g.PROJ_IMAGES_CACHE = {
                int(k): v for k, v in g.IMAGES_CACHE.get(str(project_id), {}).items()
            }
    else:
        sly.logger.info(
            "The 'images_cache.json' file not exists. Stats will be recalculated."
        )
        force_stats_recalc = True

    sly.logger.info("The cache was pulled from team files")
    return force_stats_recalc


def get_iso_timestamp():
    now = datetime.now()
    ts = datetime.timestamp(now)
    dt = datetime.utcfromtimestamp(ts)
    return str(dt.isoformat()) + "Z"


def push_cache(team_id: int, project_id: int, tf_cache_dir: str):
    local_cache_dir = f"{g.STORAGE_DIR}/_cache"
    os.makedirs(local_cache_dir, exist_ok=True)

    ts = get_iso_timestamp()

    stats_meta = {}
    spath = f"{local_cache_dir}/project_statistics_meta.json"
    if os.path.exists(spath):
        with open(spath, "r", encoding="utf-8") as f:
            stats_meta = json.load(f)

        smeta = stats_meta.get(str(project_id))
        if smeta is not None:
            smeta["updated_at"] = ts
            smeta["chunk_size"] = g.CHUNK_SIZE
            stats_meta[str(project_id)] = smeta
        else:
            stats_meta[str(project_id)] = {
                "updated_at": ts,
                "created_at": ts,
                "chunk_size": g.CHUNK_SIZE,
            }

        with open(spath, "w", encoding="utf-8") as f:
            json.dump(stats_meta, f)
    else:
        stats_meta = {
            str(project_id): {
                "updated_at": ts,
                "created_at": ts,
                "chunk_size": g.CHUNK_SIZE,
            }
        }
        with open(spath, "w", encoding="utf-8") as f:
            json.dump(stats_meta, f)

    jcache = {k: v.to_json() for k, v in g.META_CACHE.items()}
    with open(f"{local_cache_dir}/meta_cache.json", "w", encoding="utf-8") as f:
        json.dump(jcache, f)

    with open(f"{local_cache_dir}/images_cache.json", "w", encoding="utf-8") as f:
        tmp = {str(k): v for k, v in g.PROJ_IMAGES_CACHE.items()}
        g.IMAGES_CACHE.update({str(project_id): tmp})
        json.dump(g.IMAGES_CACHE, f)

    g.api.file.upload_directory(
        team_id,
        local_cache_dir,
        tf_cache_dir,
        change_name_if_conflict=False,
        replace_if_conflict=True,
    )

    sly.logger.info("The cache was pushed to team files")


def get_project_images_all(project_info: ProjectInfo) -> List[ImageInfo]:
    images_flat = []
    for dataset in g.api.dataset.get_list(project_info.id):
        images_flat += g.api.image.get_list(dataset.id)
    return images_flat


def get_updated_images_and_classes(
    project: ProjectInfo,
    project_meta: ProjectMeta,
    force_stats_recalc: bool,
) -> List[ImageInfo]:
    updated_images, updated_classes = [], []
    if len(project_meta.obj_classes.items()) == 0:
        sly.logger.info("The project is fully unlabeled")
        return updated_images

    images_flat = get_project_images_all(project)

    if force_stats_recalc is True:
        return images_flat, []

    if g.META_CACHE.get(project.id) is not None:
        cached_classes = g.META_CACHE[project.id].obj_classes
        if len(cached_classes) != len(project_meta.obj_classes):
            updated_classes = list(
                set(cached_classes.keys()).symmetric_difference(
                    set(project_meta.obj_classes.keys())
                )
            )
            sly.logger.info(
                f"Changes in the number of classes detected: {updated_classes}"
            )
            # sly.logger.warning(
            #     "Changes in the number of classes detected. Recalculate full stats... "  # TODO
            # )
            # g.META_CACHE[project.id] = project_meta
            # return images_flat

    g.META_CACHE[project.id] = project_meta

    for image in images_flat:
        try:
            image: ImageInfo
            cached_updated_at = g.PROJ_IMAGES_CACHE[image.id]
            if image.updated_at != cached_updated_at:
                updated_images.append(image)
                g.PROJ_IMAGES_CACHE[image.id] = image.updated_at
        except KeyError:
            updated_images.append(image)
            g.PROJ_IMAGES_CACHE[image.id] = image.updated_at

    set_A, set_B = set(g.PROJ_IMAGES_CACHE), set([image.id for image in images_flat])
    if set_A != set_B:
        sly.logger.warning(
            f"The add/delete operation was detected in the images with the following ids: {set_A.symmetric_difference(set_B)}"
        )
        sly.logger.info("Recalculate full statistics")
        return images_flat, []

    if len(updated_images) == project.items_count:
        sly.logger.info(f"Full dataset statistics will be calculated.")
    elif len(updated_images) > 0:
        sly.logger.info(f"The changes in {len(updated_images)} images detected")
    return updated_images, updated_classes


def get_indexes_dct(project_id: id, datasets: List[DatasetInfo]) -> Tuple[dict, dict]:
    chunk_to_images, image_to_chunk = {}, {}

    for dataset in datasets:
        images_all = g.api.image.get_list(dataset.id)
        images_all = sorted(images_all, key=lambda x: x.id)

        for idx, image_batch in enumerate(sly.batched(images_all, g.CHUNK_SIZE)):
            identifier = f"chunk_{idx}_{dataset.id}_{project_id}"
            for image in image_batch:
                image_to_chunk[image.id] = identifier
            chunk_to_images[identifier] = image_batch

    return chunk_to_images, image_to_chunk


def check_idxs_integrity(
    project, stats, curr_projectfs_dir, idx_to_infos, updated_images
) -> list:
    if sly.fs.dir_empty(curr_projectfs_dir):
        sly.logger.warning("The buffer is empty. Calculate full stats")
        if len(updated_images) != project.items_count:
            sly.logger.warning(
                f"The number of updated images ({len(updated_images)}) should equal to the number of images ({project.items_count}) in the project. Possibly the problem with cached files. Forcing recalculation..."
            )
            return get_project_images_all(project)
    else:
        for stat in stats:
            files = sly.fs.list_files(
                f"{curr_projectfs_dir}/{stat.basename_stem}",
                [".npy"],
            )

            if len(files) != len(idx_to_infos.keys()):
                msg = f"The number of images in the project has changed. Check chunks in Team Files: {curr_projectfs_dir}/{stat.basename_stem}. Forcing recalculation..."
                sly.logger.warning(msg)
                # raise RuntimeError(msg)
                return get_project_images_all(project)

    return updated_images


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
        sly.logger.warning(
            f"The {rm_cnt} old or junk chunk files were detected and removed from the buffer"
        )


def download_stats_chunks_to_buffer(team_id, tf_project_dir, project_fs_dir, stats):
    total_size = sum(
        [
            g.api.file.get_directory_size(
                team_id, f"{tf_project_dir}/{stat.basename_stem}/"
            )
            for stat in stats
        ]
    )
    with tqdm(
        desc=f"Downloading stats chunks to buffer",
        total=total_size,
        unit="B",
        unit_scale=True,
    ) as pbar:
        for stat in stats:
            g.api.file.download_directory(
                team_id,
                f"{tf_project_dir}/{stat.basename_stem}",
                f"{project_fs_dir}/{stat.basename_stem}",
                pbar,
            )


def calculate_and_save_stats(
    datasets,
    project_meta,
    updated_images,
    stats,
    tf_all_paths,
    curr_projectfs_dir,
    chunk_to_images,
    image_to_chunk,
):
    with tqdm(desc="Calculating stats", total=len(updated_images)) as pbar:
        for dataset in datasets:
            ds_updated_images = [
                image for image in updated_images if image.dataset_id == dataset.id
            ]
            updated_chunks = list(
                set([image_to_chunk[image.id] for image in ds_updated_images])
            )

            for chunk in updated_chunks:
                images_batch = chunk_to_images[chunk]
                image_ids = [image.id for image in images_batch]
                datetime_objects = [
                    datetime.fromisoformat(timestamp[:-1])
                    for timestamp in [image.updated_at for image in images_batch]
                ]
                latest_datetime = sorted(datetime_objects, reverse=True)[0]

                janns = g.api.annotation.download_json_batch(dataset.id, image_ids)
                anns = [
                    sly.Annotation.from_json(ann_json, project_meta)
                    for ann_json in janns
                ]

                for img, ann in zip(images_batch, anns):
                    for stat in stats:
                        stat.update(img, ann)
                    pbar.update(1)

                for stat in stats:
                    savedir = f"{curr_projectfs_dir}/{stat.basename_stem}"
                    os.makedirs(savedir, exist_ok=True)

                    tf_stat_chunks = [
                        path
                        for path in tf_all_paths
                        if (stat.basename_stem in path) and (chunk in path)
                    ]

                    if len(tf_stat_chunks) > 0:
                        timestamps = [
                            get_file_name(path).split("_")[-1]
                            for path in tf_stat_chunks
                        ]
                        datetime_objects = [
                            datetime.fromisoformat(timestamp)
                            for timestamp in timestamps
                        ]
                        if latest_datetime > sorted(datetime_objects, reverse=True)[0]:
                            g.TF_OLD_CHUNKS += tf_stat_chunks
                            for path in list_files(savedir, [".npy"]):
                                if chunk in path:
                                    os.remove(path)

                    np.save(
                        f"{savedir}/{chunk}_{g.CHUNK_SIZE}_{latest_datetime.isoformat()}.npy",
                        stat.to_numpy_raw(),
                    )
                    stat.clean()


def delete_old_chunks(team_id):
    if len(g.TF_OLD_CHUNKS) > 0:
        with tqdm(
            desc=f"Deleting old chunks in team files",
            total=len(g.TF_OLD_CHUNKS),
            unit="B",
            unit_scale=True,
        ) as pbar:
            g.api.file.remove_batch(team_id, g.TF_OLD_CHUNKS, progress_cb=pbar)

        sly.logger.info(f"{len(g.TF_OLD_CHUNKS)} old chunks succesfully deleted")
        g.TF_OLD_CHUNKS = []


def sew_chunks_to_json_and_upload_chunks(
    team_id, stats, curr_projectfs_dir, curr_tf_project_dir, updated_classes
):
    for stat in stats:
        stat.sew_chunks(
            chunks_dir=f"{curr_projectfs_dir}/{stat.basename_stem}/",
            updated_classes=updated_classes,
        )
        if stat.to_json() is not None:
            with open(
                f"{curr_projectfs_dir}/{stat.basename_stem}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(stat.to_json(), f)

        stat.to_image(f"{curr_projectfs_dir}/{stat.basename_stem}.png")

        npy_paths = list_files(
            f"{curr_projectfs_dir}/{stat.basename_stem}", valid_extensions=[".npy"]
        )
        dst_npy_paths = [
            f"{curr_tf_project_dir}/{stat.basename_stem}/{get_file_name_with_ext(path)}"
            for path in npy_paths
        ]

        with tqdm(
            desc=f"Uploading {stat.basename_stem} chunks",
            total=sly.fs.get_directory_size(
                f"{curr_projectfs_dir}/{stat.basename_stem}"
            ),
            unit="B",
            unit_scale=True,
        ) as pbar:
            g.api.file.upload_bulk(team_id, npy_paths, dst_npy_paths, pbar)

        sly.logger.info(
            f"{stat.basename_stem}: {len(npy_paths)} chunks succesfully uploaded"
        )


def upload_sewed_stats(team_id, curr_projectfs_dir, curr_tf_project_dir):
    remove_files_with_null(curr_projectfs_dir)
    json_paths = list_files(curr_projectfs_dir, valid_extensions=[".json"])
    dst_json_paths = [
        f"{curr_tf_project_dir}/{get_file_name_with_ext(path)}" for path in json_paths
    ]

    with tqdm(
        desc="Uploading .json stats",
        total=sum([get_file_size(path) for path in json_paths]),
        unit="B",
        unit_scale=True,
    ) as pbar:
        g.api.file.upload_bulk(team_id, json_paths, dst_json_paths, pbar)

    sly.logger.info(
        f"{len(json_paths)} updated .json stats succesfully updated and uploaded"
    )


def remove_files_with_null(directory_path: str):
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            with open(file_path, "r") as file:
                try:
                    json_data = json.load(file)
                    if json_data is None:
                        os.remove(file_path)
                        print(f"Removed {filename} as it contains null values.")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {filename}.")
