import json, time
import tarfile
import os
import math
from typing import List, Literal, Optional, Dict, Tuple
import dataset_tools as dtools
from dataset_tools.image.stats.basestats import BaseStats
from datetime import datetime
import humanize
from supervisely import ImageInfo, ProjectMeta, ProjectInfo, DatasetInfo
from itertools import groupby
from tqdm import tqdm
import supervisely as sly
import src.globals as g
import numpy as np
import ujson
from collections import defaultdict

from supervisely.io.fs import (
    get_file_name_with_ext,
    get_file_name,
    list_files,
    get_file_size,
    list_files_recursively,
)


def _load_json_cache(path_img, path_meta):
    if os.path.exists(path_img):
        with open(path_img, "r", encoding="utf-8") as f:
            g.IMAGES_CACHE = json.load(f)
    if os.path.exists(path_meta):
        with open(path_meta, "r", encoding="utf-8") as f:
            g.META_CACHE = {int(k): sly.ProjectMeta().from_json(v) for k, v in json.load(f).items()}


def pull_cache(team_id: int, project_id: int, tf_cache_dir: str, tf_project_dir: str) -> bool:
    force_stats_recalc = False

    local_cache_dir = f"{g.STORAGE_DIR}/_cache"
    if sly.fs.dir_exists(local_cache_dir):
        sly.fs.clean_dir(local_cache_dir)
    if g.api.file.dir_exists(team_id, tf_project_dir):
        g.api.file.download_directory(team_id, tf_cache_dir, local_cache_dir)

    path_img = os.path.join(local_cache_dir, "images_cache.json")
    path_meta = os.path.join(local_cache_dir, "meta_cache.json")
    _load_json_cache(path_img, path_meta)

    if not g.api.file.dir_exists(team_id, tf_cache_dir):
        sly.logger.warning("The cache directory not exists in team files. ")
        return True

    if not g.api.file.dir_exists(team_id, tf_project_dir):
        sly.logger.warning("The project directory not exists in team files.")
        return True

    spath = f"{local_cache_dir}/project_statistics_meta.json"
    if os.path.exists(spath):
        with open(spath, "r", encoding="utf-8") as f:
            stats_meta = json.load(f)
        smeta = stats_meta.get(str(project_id))
        if smeta is not None:
            if smeta["chunk_size"] != g.CHUNK_SIZE:
                sly.logger.warning("The chunk size has changed. Recalculating full stats...")
                return True
            ts = smeta["chunks_dt"][:-1]
            g.CHUNKS_LATEST_DATETIME = datetime.fromisoformat(ts)

    if os.path.exists(path_meta):
        if g.META_CACHE.get(project_id) is None:
            # TODO bug whwn delete clsas false recalc
            # (not see project id in cache)
            g.CHUNKS_LATEST_DATETIME
            sly.logger.info(
                f"The key with project ID={project_id} was not found in 'meta_cache.json'. Stats will be fully recalculated."
            )
            force_stats_recalc = True
    else:
        sly.logger.warning("The 'meta_cache.json' file not exists. Stats will be recalculated.")
        force_stats_recalc = True

    if os.path.exists(path_img):
        if g.IMAGES_CACHE.get(str(project_id)) is not None:
            g.PROJ_IMAGES_CACHE = {int(k): v for k, v in g.IMAGES_CACHE[str(project_id)].items()}
        else:
            sly.logger.info(
                f"The key with project ID={project_id} was not found in 'images_cache.json'. Stats will be fully recalculated."
            )
            force_stats_recalc = True
    else:
        sly.logger.info("The 'images_cache.json' file not exists. Stats will be recalculated.")
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
    chunks_dt = str(g.CHUNKS_LATEST_DATETIME.isoformat()) + "Z"

    stats_meta = {}
    spath = f"{local_cache_dir}/project_statistics_meta.json"
    if os.path.exists(spath):
        with open(spath, "r", encoding="utf-8") as f:
            stats_meta = json.load(f)

        smeta = stats_meta.get(str(project_id))
        if smeta is not None:
            smeta["updated_at"] = ts
            smeta["chunk_size"] = g.CHUNK_SIZE
            smeta["chunks_dt"] = chunks_dt
            stats_meta[str(project_id)] = smeta
        else:
            stats_meta[str(project_id)] = {
                "updated_at": ts,
                "created_at": ts,
                "chunk_size": g.CHUNK_SIZE,
                "chunks_dt": chunks_dt,
            }

        with open(spath, "w", encoding="utf-8") as f:
            json.dump(stats_meta, f)
    else:
        stats_meta = {
            str(project_id): {
                "updated_at": ts,
                "created_at": ts,
                "chunk_size": g.CHUNK_SIZE,
                "chunks_dt": chunks_dt,
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


def get_project_images_all(datasets: List[DatasetInfo]) -> List[ImageInfo]:
    return {d.id: g.api.image.get_list(d.id) for d in datasets}


def get_updated_images_and_classes(
    project: ProjectInfo,
    project_meta: ProjectMeta,
    datasets: List[DatasetInfo],
    force_stats_recalc: bool,
) -> Tuple[List[ImageInfo], List[str]]:
    updated_images, updated_classes = {d.id: [] for d in datasets}, {}
    if len(project_meta.obj_classes.items()) == 0:
        sly.logger.info("The project is fully unlabeled")
        return {}, {}

    images_all_dct = get_project_images_all(datasets)
    images_all_flat = []
    for value in images_all_dct.values():
        images_all_flat.extend(value)

    if force_stats_recalc is True:
        for image in images_all_flat:
            g.PROJ_IMAGES_CACHE[image.id] = image.updated_at
        g.META_CACHE[project.id] = project_meta
        return images_all_dct, {}

    if g.META_CACHE.get(project.id) is not None:
        cached_classes = g.META_CACHE[project.id].obj_classes
        if len(cached_classes) != len(project_meta.obj_classes):
            cached = {x.sly_id: x.name for x in cached_classes}
            actual = {x.sly_id: x.name for x in project_meta.obj_classes}
            cached_ids = set(cached.keys())
            actual_ids = set(actual.keys())

            updated_ids = actual_ids.symmetric_difference(cached_ids)

            def func(pair):
                id, name = pair
                return True if id in updated_ids else False

            for dct in [cached, actual]:
                updated_classes.update(dict(filter(func, dct.items())))

            sly.logger.info(
                f"Changes in the number of classes detected: {list(updated_classes.values())}"
            )

    g.META_CACHE[project.id] = project_meta

    set_A, set_B = set(g.PROJ_IMAGES_CACHE), set([i.id for i in images_all_flat])

    for image in images_all_flat:
        try:
            image: ImageInfo
            cached_updated_at = g.PROJ_IMAGES_CACHE[image.id]
            if image.updated_at != cached_updated_at:
                updated_images[image.dataset_id].append(image)
                g.PROJ_IMAGES_CACHE[image.id] = image.updated_at
        except KeyError:
            updated_images[image.dataset_id].append(image)
            g.PROJ_IMAGES_CACHE[image.id] = image.updated_at

    if set_A != set_B:
        if set_A.issubset(set_B):
            sly.logger.warning(f"The images with the following ids were added: {set_B - set_A}")
        elif set_B.issubset(set_A):
            sly.logger.warning(f"The images with the following ids were deleted: {set_A - set_B}")
            g.PROJ_IMAGES_CACHE = {
                k: v for k, v in g.PROJ_IMAGES_CACHE.items() if k not in (set_A - set_B)
            }

        sly.logger.info("Recalculate full statistics")
        return images_all_dct, {}

    num_updated = sum(len(lst) for lst in updated_images.values())
    if num_updated == project.items_count:
        sly.logger.info(f"Full dataset statistics will be calculated.")
    elif num_updated > 0:
        sly.logger.info(f"The changes in {num_updated} images detected")
    return updated_images, updated_classes


def get_indexes_dct(project_id: id, datasets: List[DatasetInfo]) -> Tuple[dict, dict]:
    chunk_to_images, image_to_chunk = {}, {}

    for dataset in datasets:
        images_all = g.api.image.get_list(dataset.id)  # TODO optimie speed
        images_all = sorted(images_all, key=lambda x: x.id)

        for idx, image_batch in enumerate(sly.batched(images_all, g.CHUNK_SIZE)):
            identifier = f"chunk_{idx}_{dataset.id}_{project_id}"
            for image in image_batch:
                image_to_chunk[image.id] = identifier
            chunk_to_images[identifier] = image_batch

    return chunk_to_images, image_to_chunk


def check_idxs_integrity(
    project, datasets, stats, projectfs_dir, idx_to_infos, updated_images, force_stats_recalc
) -> list:
    if force_stats_recalc is True:
        return get_project_images_all(datasets)

    if sly.fs.dir_empty(projectfs_dir):
        sly.logger.warning("The buffer is empty. Calculate full stats")
        if any(len(x) != d.items_count for x, d in zip(updated_images.values(), datasets)):
            total_updated = sum(len(lst) for lst in updated_images.values())
            sly.logger.warning(
                f"The number of updated images ({total_updated}) should equal to the number of images ({project.items_count}) in the project. Possibly the problem with cached files. Forcing recalculation..."
            )  # TODO
            return get_project_images_all(datasets)
    else:
        for stat in stats:
            files = sly.fs.list_files(
                f"{projectfs_dir}/{stat.basename_stem}",
                [".npy"],
            )

            if len(files) != len(idx_to_infos.keys()):
                msg = f"The number of images in the project has changed. Check chunks in Team Files: {projectfs_dir}/{stat.basename_stem}. Forcing recalculation..."
                sly.logger.warning(msg)
                # raise RuntimeError(msg)
                return get_project_images_all(datasets)

    return updated_images


def check_datasets_consistency(project_info, datasets, npy_paths, num_stats):
    for dataset in datasets:
        actual_ceil = math.ceil(dataset.items_count / g.CHUNK_SIZE)
        max_chunks = math.ceil(
            len([path for path in npy_paths if f"_{dataset.id}_" in sly.fs.get_file_name(path)])
            / num_stats
        )
        if actual_ceil < max_chunks:
            raise ValueError(
                f"The number of chunks per stat ({len(npy_paths)}) not match with the total items count of the project ({project_info.items_count}) using following batch size: {g.CHUNK_SIZE}. Details: DATASET_ID={dataset.id}; actual num of chunks: {actual_ceil}; max num of chunks: {max_chunks}"
            )
    sly.logger.info("The consistency of data is OK")


def remove_junk(team_id, tf_project_dir, project, datasets, project_fs_dir):
    files_fs = list_files_recursively(project_fs_dir, valid_extensions=[".npy"])
    ds_ids, rm_cnt = [str(dataset.id) for dataset in datasets], 0

    grouped_paths = defaultdict(list)
    old_paths = []

    for path in files_fs:
        constant_part = path.split("_")[:-1]
        constant_part = "_".join(constant_part)
        grouped_paths[constant_part].append(path)

    for constant_part, paths_list in grouped_paths.items():
        if len(paths_list) > 1:
            newest_path = max(paths_list, key=lambda x: x.split("_")[-1])
            old_paths += [p for p in paths_list if newest_path != p]
            grouped_paths[constant_part] = [newest_path]

    for path in old_paths:
        os.remove(path)
        rm_cnt += 1

    for path in files_fs:
        if (path.split("_")[-4] not in ds_ids) or (f"_{project.id}_{g.CHUNK_SIZE}_" not in path):
            os.remove(path)
            rm_cnt += 1

    if rm_cnt > 0:
        sly.logger.info(
            f"The {rm_cnt} old or junk chunk files were detected and removed from the buffer"
        )

    chunks_archive = [
        f for f in g.api.file.listdir(team_id, tf_project_dir) if f.endswith(".tar.gz")
    ]
    if len(chunks_archive) > 1:
        for chunks in chunks_archive:
            tf_chunks_dt = ".".join(sly.fs.get_file_name(chunks).split(".")[:-1]).split("_")[-1]
            if tf_chunks_dt != g.CHUNKS_LATEST_DATETIME.isoformat():
                g.api.file.remove_file(team_id, chunks)
                sly.logger.info(
                    f"The {chunks} old or junk chunks archive was detected and removed from the team files."
                )


@sly.timeit
def download_stats_chunks_to_buffer(
    team_id,
    project: ProjectInfo,
    tf_project_dir,
    project_fs_dir,
    force_stats_recalc,
) -> bool:
    if force_stats_recalc:
        return True

    if g.CHUNKS_LATEST_DATETIME is None:
        sly.logger.warning(
            "The chunks identifier of latest datetime is not existed.  Recalculating full stats."
        )
        return True
    cached_chunks_dt = g.CHUNKS_LATEST_DATETIME.isoformat()
    archive_name = f"{project.id}_{project.name}_chunks_{cached_chunks_dt}.tar.gz"
    src_path = f"{tf_project_dir}/{archive_name}"
    dst_path = f"{project_fs_dir}/{archive_name}"

    file = g.api.file.get_info_by_path(team_id, src_path)
    if file is None:
        sly.logger.warning(
            f"The chunks archive file is not existed: '{archive_name}'.  Recalculating full stats."
        )
        return True
    tf_chunks_dt = ".".join(sly.fs.get_file_name(file.path).split(".")[:-1]).split("_")[-1]
    if cached_chunks_dt != tf_chunks_dt:
        sly.logger.warning(
            f"The chunks datetime '{tf_chunks_dt}' differs from the cached one: '{cached_chunks_dt}'.  Recalculating full stats."
        )
        return True

    with tqdm(
        desc="Downloading stats chunks to buffer",
        total=file.sizeb,
        unit="B",
        unit_scale=True,
    ) as pbar:
        try:
            g.api.file.download(team_id, src_path, dst_path, progress_cb=pbar)
        except:
            sly.logger.warning(
                "The integrity of the team files is broken. Recalculating full stats."
            )
            return True

    with tarfile.open(dst_path, "r:gz") as tar:
        tar.extractall(project_fs_dir)

    return False


@sly.timeit
def calculate_and_save_stats(
    updated_images,
    stats,
    tf_all_paths,
    project_fs_dir,
    chunk_to_images,
    image_to_chunk,
):
    total_updated = sum(len(lst) for lst in updated_images.values())
    sly.logger.info(f"Start calculating stats for {total_updated} images.")
    with tqdm(desc="Calculating stats", total=total_updated) as pbar:

        for dataset_id, images in updated_images.items():
            updated_chunks = list(set([image_to_chunk[image.id] for image in images]))

            for chunk in updated_chunks:
                images_chunk = chunk_to_images[chunk]

                for batch_infos in sly.batched(images_chunk, 100):
                    batch_ids = [x.id for x in batch_infos]
                    figures = g.api.image.figure.download(dataset_id, batch_ids, skip_geometry=True)
                    for image in batch_infos:
                        for stat in stats:
                            stat.update2(image, figures.get(image.id, []))
                    pbar.update(len(batch_infos))

                latest_datetime = get_latest_datetime(images_chunk)
                if g.CHUNKS_LATEST_DATETIME is None or g.CHUNKS_LATEST_DATETIME < latest_datetime:
                    g.CHUNKS_LATEST_DATETIME = latest_datetime
                for stat in stats:
                    save_chunks(stat, chunk, project_fs_dir, tf_all_paths, latest_datetime)
                    stat.clean()

        if pbar.last_print_n < pbar.total:  # unlabeled images
            pbar.update(pbar.total - pbar.n)


# @sly.timeit
def get_latest_datetime(images_chunk):
    datetime_objects = [
        datetime.fromisoformat(timestamp[:-1])
        for timestamp in [image.updated_at for image in images_chunk]
    ]
    return sorted(datetime_objects, reverse=True)[0]


# @sly.timeit
def save_chunks(stat, chunk, project_fs_dir, tf_all_paths, latest_datetime):
    savedir = f"{project_fs_dir}/{stat.basename_stem}"
    os.makedirs(savedir, exist_ok=True)

    tf_stat_chunks = [
        path for path in tf_all_paths if (stat.basename_stem in path) and (chunk in path)
    ]

    if len(tf_stat_chunks) > 0:
        timestamps = [get_file_name(path).split("_")[-1] for path in tf_stat_chunks]
        datetime_objects = [datetime.fromisoformat(timestamp) for timestamp in timestamps]
        if latest_datetime > sorted(datetime_objects, reverse=True)[0]:
            for path in list_files(savedir, [".npy"]):
                if chunk in path:
                    os.remove(path)

    np.save(
        f"{savedir}/{chunk}_{g.CHUNK_SIZE}_{latest_datetime.isoformat()}.npy",
        stat.to_numpy_raw(),
    )


@sly.timeit
def sew_chunks_to_json(stats: List[BaseStats], project_fs_dir, updated_classes):
    # @sly.timeit
    def _save_to_json(res, dst_path):
        json_data = ujson.dumps(res)
        json_bytes = json_data.encode("utf-8")
        with open(dst_path, "wb") as f:  # Use binary mode
            f.write(json_bytes)

    for stat in stats:
        # sly.logger.info(f"### {stat.basename_stem}")
        # tm = sly.TinyTimer()
        stat.sew_chunks(
            chunks_dir=f"{project_fs_dir}/{stat.basename_stem}/",
            updated_classes=updated_classes,
        )
        # sly.logger.info(f"chunks_sewed: {tm.get_sec()}")
        if sly.is_development():
            stat.to_image(f"{project_fs_dir}/{stat.basename_stem}.png", version2=True)

        # tm = sly.TinyTimer()
        res = stat.to_json2()
        # sly.logger.info(f"json_received: {tm.get_sec()}")
        if res is not None:
            # tm = sly.TinyTimer()
            _save_to_json(res, f"{project_fs_dir}/{stat.basename_stem}.json")
            # sly.logger.info(f"json_saved: {tm.get_sec()}")


@sly.timeit
def archive_chunks_and_upload(
    team_id,
    project: ProjectInfo,
    stats: List[BaseStats],
    tf_project_dir,
    project_fs_dir,
):
    def _compress_folders(folders, archive_path) -> int:
        with tarfile.open(archive_path, "w:gz") as tar:
            for folder in folders:
                tar.add(folder, arcname=os.path.basename(folder))
        return sly.fs.get_file_size(archive_path)

    folders_to_compress = [f"{project_fs_dir}/{stat.basename_stem}" for stat in stats]

    dt_identifier = g.CHUNKS_LATEST_DATETIME
    archive_name = f"{project.id}_{project.name}_chunks_{dt_identifier.isoformat()}.tar.gz"
    src_path = f"{project_fs_dir}/{archive_name}"
    archive_sizeb = _compress_folders(folders_to_compress, src_path)

    dst_path = f"{tf_project_dir}/{archive_name}"
    with tqdm(
        desc=f"Uploading '{archive_name}'",
        total=archive_sizeb,
        unit="B",
        unit_scale=True,
    ) as pbar:
        g.api.file.upload(team_id, src_path, dst_path, progress_cb=pbar)

    sly.logger.info(f"The '{archive_name}' file was succesfully uploaded.")


@sly.timeit
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

    sly.logger.info(f"{len(json_paths)} updated .json stats succesfully updated and uploaded")


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
