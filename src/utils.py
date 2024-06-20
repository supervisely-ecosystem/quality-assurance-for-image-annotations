from pathlib import Path

import json, time
from packaging.version import Version
import tarfile
import os
import math
from typing import List, Literal, Optional, Dict, Tuple, Union, Set
import dataset_tools as dtools
from dataset_tools.image.stats.basestats import BaseStats
from datetime import datetime
import humanize
from supervisely import ImageInfo, ProjectMeta, ProjectInfo, DatasetInfo, FigureInfo, TeamInfo
from itertools import groupby
from tqdm import tqdm
import supervisely as sly
import src.globals as g
import numpy as np
import ujson
from collections import defaultdict
import random
from supervisely.io.fs import (
    get_file_name_with_ext,
    get_file_name,
    list_files,
    get_file_size,
    list_files_recursively,
)
from supervisely.imaging.color import _validate_hex_color, hex2rgb, random_rgb, rgb2hex


def pull_cache(
    team_id: int, project_id: int, tf_project_dir: str, project_fs_dir: str
) -> Tuple[bool, dict]:
    _cache = {}

    if not g.api.file.dir_exists(team_id, tf_project_dir):
        sly.logger.warning("The project directory not exists in team files.")
        return True, _cache

    filename = f"{project_id}_cache.json"
    tf_cache_path = f"{tf_project_dir}/_cache/{filename}"

    local_cache_path = f"{project_fs_dir}/_cache/{filename}"

    if g.api.file.exists(team_id, tf_cache_path):
        g.api.file.download(team_id, tf_cache_path, local_cache_path)
    else:
        sly.logger.warning(f"The {filename!r} not exists in team files.")
        return True, _cache

    if os.path.exists(local_cache_path):
        with open(local_cache_path, "r", encoding="utf-8") as f:
            _cache = json.load(f)

    images = _cache.get("images")
    meta = _cache.get("meta")
    smeta = _cache.get("stats_meta")

    if images is None:
        sly.logger.info(
            f"The key with project ID={project_id} was not found in 'images_cache.json'. Stats will be fully recalculated."
        )
        return True, _cache

    if meta is None:
        sly.logger.info(
            f"The key with project ID={project_id} was not found in 'meta_cache.json'. Stats will be fully recalculated."
        )
        return True, _cache

    if smeta.get("chunk_size", -1) != g.CHUNK_SIZE:
        sly.logger.warning("The chunk size has changed. Recalculating full stats...")
        return True, _cache

    chunks_dt = smeta.get("chunks_dt")
    if chunks_dt is None:
        sly.logger.warning(
            "The cache has no chunks datetime to verify. Recalculating full stats..."
        )
        return True, _cache
    else:
        g.CHUNKS_LATEST_DATETIME = datetime.fromisoformat(chunks_dt[:-1])

    dtools_version = smeta.get("dataset-tools")
    if dtools_version is None:
        sly.logger.warning(
            "The cache has no 'dataset-tools' version to verify. Recalculating full stats..."
        )
        return True, _cache
    else:
        if Version(dtools_version) < Version(g.MINIMUM_DTOOLS_VERSION):
            sly.logger.warning(
                f"The cached version ({dtools_version}) of 'dataset-tools' package is less than the required one ({g.MINIMUM_DTOOLS_VERSION}). Force statistics recalculation."
            )
            return True, _cache

    sly.logger.info(f"The cache file {filename!r} was pulled from team files")

    _cache["stats_meta"] = smeta
    _cache["meta"] = meta
    _cache["images"] = {int(k): v for k, v in images.items()}
    return False, _cache


def get_iso_timestamp():
    now = datetime.now()
    ts = datetime.timestamp(now)
    dt = datetime.utcfromtimestamp(ts)
    return str(dt.isoformat()) + "Z"


def push_cache(
    team_id: int, project_id: int, tf_project_dir: str, project_fs_dir: str, _cache: dict
) -> dict:
    filename = f"{project_id}_cache.json"
    tf_cache_path = f"{tf_project_dir}/_cache/{filename}"

    local_cache_dir = f"{project_fs_dir}/_cache"
    local_cache_path = f"{local_cache_dir}/{filename}"

    ts_utc = get_iso_timestamp()
    chunks_dt = str(g.CHUNKS_LATEST_DATETIME.isoformat()) + "Z"

    try:
        actual_version = dtools.__version__
    except:
        actual_version = None

    smeta = _cache.get("stats_meta")
    if smeta is None:
        _cache["stats_meta"] = {
            "updated_at": ts_utc,
            "created_at": ts_utc,
            "chunk_size": g.CHUNK_SIZE,
            "chunks_dt": chunks_dt,
            "dataset-tools": actual_version,
        }
    else:
        _cache["stats_meta"]["updated_at"] = ts_utc
        _cache["stats_meta"]["chunk_size"] = g.CHUNK_SIZE
        _cache["stats_meta"]["chunks_dt"] = chunks_dt
        _cache["stats_meta"]["dataset-tools"] = actual_version

    os.makedirs(local_cache_dir, exist_ok=True)
    with open(local_cache_path, "w", encoding="utf-8") as f:
        json.dump(_cache, f)

    g.api.file.upload(team_id, local_cache_path, tf_cache_path)
    sly.logger.info(f"The cache file {filename!r} was pushed to team files")

    # remove old junk
    g.api.file.remove_dir(team_id, f"{os.path.dirname(tf_project_dir)}/_cache/", silent=True)

    return _cache


@sly.timeit
def get_project_images_all(datasets: List[DatasetInfo]) -> Dict[int, ImageInfo]:
    return {d.id: g.api.image.get_list(d.id) for d in datasets}


@sly.timeit
def get_updated_images_and_classes(
    project: ProjectInfo,
    project_meta: ProjectMeta,
    datasets: List[DatasetInfo],
    images_all_dct,
    force_stats_recalc: bool,
    _cache: dict,
) -> Tuple[List[ImageInfo], List[str]]:
    _images_cached = _cache.get("images", {})
    _meta_cached_json = _cache.get("meta")
    _project_meta_cached = ProjectMeta.from_json(_meta_cached_json) if _meta_cached_json else None
    is_meta_changed = compare_metas(project_meta, _project_meta_cached)

    updated_images, updated_classes = {d.id: [] for d in datasets}, {}
    if len(project_meta.obj_classes.items()) == 0:
        sly.logger.info("The project is fully unlabeled")
        return {}, {}, {}, is_meta_changed

    images_all_flat = []
    for value in images_all_dct.values():
        images_all_flat.extend(value)

    images_updated_at = {}
    for image in images_all_flat:
        images_updated_at[image.id] = image.updated_at

    _cache["images"] = images_updated_at
    _cache["meta"] = project_meta.to_json()

    if force_stats_recalc is True:
        return images_all_dct, {}, _cache, is_meta_changed

    if _project_meta_cached is not None:
        cached_classes = _project_meta_cached.obj_classes
        if len(cached_classes) != len(project_meta.obj_classes):
            cached = {x.sly_id: x.name for x in cached_classes}
            actual = {x.sly_id: x.name for x in project_meta.obj_classes}
            cached_ids = set(cached.keys())
            actual_ids = set(actual.keys())

            updated_ids = actual_ids.symmetric_difference(cached_ids)

            def _func(pair):
                id, name = pair
                return True if id in updated_ids else False

            for dct in [cached, actual]:
                updated_classes.update(dict(filter(_func, dct.items())))

            sly.logger.info(
                f"Changes in the number of classes detected: {list(updated_classes.values())}"
            )

    set_A, set_B = set(_images_cached), set([i.id for i in images_all_flat])

    for image in images_all_flat:
        try:
            image: ImageInfo
            cached_updated_at = _images_cached[image.id]
            if image.updated_at != cached_updated_at:
                updated_images[image.dataset_id].append(image)
        except KeyError:
            updated_images[image.dataset_id].append(image)

    if set_A != set_B:
        if set_A.issubset(set_B):
            sly.logger.warning(f"The images with the following ids were added: {set_B - set_A}")
        elif set_B.issubset(set_A):
            sly.logger.warning(f"The images with the following ids were deleted: {set_A - set_B}")

        sly.logger.info("Recalculate full statistics")
        return images_all_dct, {}, _cache, is_meta_changed

    num_updated = sum(len(lst) for lst in updated_images.values())
    if num_updated == project.items_count:
        sly.logger.info(f"Full dataset statistics will be calculated.")
    elif num_updated > 0:
        sly.logger.info(f"The changes in {num_updated} images detected")

    return updated_images, updated_classes, _cache, is_meta_changed


@sly.timeit
def get_indexes_dct(
    project_id: id, datasets: List[DatasetInfo], images_all_dct
) -> Tuple[dict, dict]:
    chunk_to_images, image_to_chunk = {}, {}

    for dataset in datasets:
        images_all = images_all_dct[dataset.id]
        images_all = sorted(images_all, key=lambda x: x.id)

        for idx, image_batch in enumerate(sly.batched(images_all, g.CHUNK_SIZE)):
            identifier = f"chunk_{idx}_{dataset.id}_{project_id}"
            for image in image_batch:
                image_to_chunk[image.id] = identifier
            chunk_to_images[identifier] = image_batch

    return chunk_to_images, image_to_chunk


@sly.timeit
def check_idxs_integrity(
    project,
    datasets,
    stats,
    projectfs_dir,
    idx_to_infos,
    updated_images,
    images_all_dct,
    force_stats_recalc,
) -> list:
    if force_stats_recalc is True:
        return images_all_dct

    if sly.fs.dir_empty(projectfs_dir):
        sly.logger.warning("The buffer is empty. Calculate full stats")
        if any(len(x) != d.items_count for x, d in zip(updated_images.values(), datasets)):
            total_updated = sum(len(lst) for lst in updated_images.values())
            sly.logger.warning(
                f"The number of updated images ({total_updated}) should equal to the number of images ({project.items_count}) in the project. Possibly the problem with cached files. Forcing recalculation..."
            )
            return images_all_dct
    else:
        try:
            for stat in stats:
                files = sly.fs.list_files(
                    f"{projectfs_dir}/{stat.basename_stem}",
                    [".npy"],
                )

                if len(files) != len(idx_to_infos.keys()):
                    msg = f"The number of images in the project has changed. Check chunks in Team Files: {projectfs_dir}/{stat.basename_stem}. Forcing recalculation..."
                    sly.logger.warning(msg)
                    return images_all_dct
        except:
            sly.logger.warning("Error while integrity checking. Recalc full stats.")
            return images_all_dct

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


@sly.timeit
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
def calculate_stats_and_save_chunks(
    updated_images,
    stats,
    tf_all_paths,
    project_fs_dir,
    chunk_to_images,
    image_to_chunk,
    project_stats: dict,
    project,
) -> Dict[int, Set[ImageInfo]]:
    heatmaps_image_ids = defaultdict(set)
    heatmaps_figure_ids = defaultdict(set)
    total_updated = sum(len(lst) for lst in updated_images.values())
    total_updated_figures = sum(x.labels_count for lst in updated_images.values() for x in lst)
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
                        figs = figures.get(image.id, [])
                        for stat in stats:
                            stat.update2(image, figs)
                        _update_heatmaps_sample(
                            heatmaps_figure_ids,
                            heatmaps_image_ids,
                            figs,
                            total_updated_figures,
                            project_stats["objects"]["total"]["objectsInDataset"],
                            project.size,
                        )

                    pbar.update(len(batch_infos))

                latest_datetime = get_latest_datetime(images_chunk)
                if g.CHUNKS_LATEST_DATETIME is None or g.CHUNKS_LATEST_DATETIME < latest_datetime:
                    g.CHUNKS_LATEST_DATETIME = latest_datetime
                for stat in stats:
                    save_chunks(stat, chunk, project_fs_dir, tf_all_paths, latest_datetime)
                    stat.clean()

        if pbar.last_print_n < pbar.total:  # unlabeled images
            pbar.update(pbar.total - pbar.n)

    return heatmaps_image_ids, heatmaps_figure_ids


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
def sew_chunks_to_json(
    stats: List[BaseStats], project_fs_dir, updated_classes, is_meta_changed: bool
):
    # @sly.timeit
    def _save_to_json(res, dst_path):
        json_data = ujson.dumps(res)
        json_bytes = json_data.encode("utf-8")
        with open(dst_path, "wb") as f:  # Use binary mode
            f.write(json_bytes)

    for stat in stats:
        stat.sew_chunks(chunks_dir=f"{project_fs_dir}/{stat.basename_stem}/")
        if sly.is_development():
            stat.to_image(f"{project_fs_dir}/{stat.basename_stem}.png", version2=True)

        res = stat.to_json2()
        if res is not None:
            _save_to_json(res, f"{project_fs_dir}/{stat.basename_stem}.json")


def _update_heatmaps_sample(
    heatmaps_figure_ids,
    heatmaps_image_ids,
    figs: List[FigureInfo],
    total_updated_figures: int,
    total_project_figures: int,
    project_size: str,
):
    if total_project_figures == 0:
        return
    threshold = 1
    if total_updated_figures / total_project_figures > 0.3 and int(project_size) > 10e9:
        threshold = 60 / total_project_figures

    for fig in figs:
        if random.random() < threshold:
            heatmaps_figure_ids[fig.class_id].add(fig.id)
            heatmaps_image_ids[fig.dataset_id].add(fig.entity_id)


def calculate_and_upload_heatmaps(
    team: TeamInfo,
    tf_project_dir: str,
    project_fs_dir: str,
    heatmaps: dtools.ClassesHeatmaps,
    heatmaps_image_ids: Dict[int, Set[int]],
    heatmaps_figure_ids: Dict[int, Set[int]],
):
    if len(heatmaps_image_ids) == 0:
        return

    sample_total = sum(len(lst) for lst in heatmaps_image_ids.values())
    with tqdm(desc="Calculating heatmaps from sample", total=sample_total) as pbar:

        for dataset_id, image_ids in heatmaps_image_ids.items():
            image_infos = g.api.image.get_info_by_id_batch(list(image_ids))

            for batch_infos in sly.batched(image_infos, 100):
                batch_ids = [x.id for x in batch_infos]
                figures = g.api.image.figure.download(dataset_id, batch_ids)

                for image in batch_infos:
                    figs = figures.get(image.id, [])
                    filtered = [x for x in figs if x.id in heatmaps_figure_ids[x.class_id]]
                    heatmaps.update2(image, filtered, skip_broken_geometry=True)
                    pbar.update(1)

    heatmaps_name = f"{heatmaps.basename_stem}.png"
    fs_heatmap_path = f"{project_fs_dir}/{heatmaps_name}"
    tf_heatmap_path = f"{tf_project_dir}/{heatmaps_name}"
    heatmaps.to_image(fs_heatmap_path)

    g.api.file.upload(team.id, fs_heatmap_path, tf_heatmap_path)
    sly.logger.info(f"The {heatmaps_name!r} file was succesfully uploaded.")
    add_heatmaps_status_ok(team, tf_project_dir, project_fs_dir)


def add_heatmaps_status_ok(team, tf_project_dir, project_fs_dir):
    status_path = f"{project_fs_dir}/_cache/heatmaps/status_ok"
    tf_status_path = f"{tf_project_dir}/_cache/heatmaps/status_ok"
    os.makedirs(f"{project_fs_dir}/_cache/heatmaps", exist_ok=True)
    Path(status_path).touch()
    g.api.file.upload(team.id, status_path, tf_status_path)


@sly.timeit
def archive_chunks_and_upload(
    team: TeamInfo,
    project: ProjectInfo,
    stats: List[BaseStats],
    tf_project_dir,
    project_fs_dir,
    datasets,
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
        g.api.file.upload(team.id, src_path, dst_path, progress_cb=pbar)

    remove_junk(team.id, tf_project_dir, project, datasets, project_fs_dir)
    sly.logger.info(f"The '{archive_name}' file was succesfully uploaded.")


@sly.timeit
def upload_sewed_stats(team_id, curr_projectfs_dir, curr_tf_project_dir):
    remove_files_with_null(curr_projectfs_dir)
    stats_paths = list_files(curr_projectfs_dir, valid_extensions=[".json"])
    dst_json_paths = [
        f"{curr_tf_project_dir}/{get_file_name_with_ext(path)}" for path in stats_paths
    ]

    with tqdm(
        desc="Uploading .json stats",
        total=sum([get_file_size(path) for path in stats_paths]),
        unit="B",
        unit_scale=True,
    ) as pbar:
        try:
            g.api.file.upload_bulk(team_id, stats_paths, dst_json_paths, pbar)
        except:
            pass

    sly.logger.info(
        f"{len(stats_paths)} updated .json and .png stats succesfully updated and uploaded"
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


def applicability_test(stat):
    if len(stat._tag_ids) == 0:
        return False
    return True


def handle_broken_project_meta(json_project_meta: dict) -> dict:
    for idx, cls in enumerate(json_project_meta["classes"]):
        # if _validate_hex_color(cls["color"]) is False:
        #     new_color = rgb2hex(random_rgb())
        #     sly.logger.warning(
        #         f"'{cls['color']}' is not validated as hex. Trying to convert it to: {new_color}"
        #     )
        #     json_project_meta["classes"][idx]["color"] = new_color

        for node, data in cls["geometry_config"]["nodes"].items():
            curr_color = data.get("color")
            new_color = rgb2hex(random_rgb())
            if curr_color is not None:
                if _validate_hex_color("#" + curr_color) is True:
                    data["color"] = "#" + data["color"]

    return json_project_meta


def compare_metas(
    project_meta: ProjectMeta, _project_meta_cached: Union[ProjectMeta, dict]
) -> bool:
    if _project_meta_cached is None:
        return False
    for tag_meta in project_meta.tag_metas:
        if tag_meta not in _project_meta_cached.tag_metas:
            return True
    for tag_meta in _project_meta_cached.tag_metas:
        if tag_meta not in project_meta.tag_metas:
            return True
    for class_meta in project_meta.obj_classes:
        if class_meta not in _project_meta_cached.obj_classes:
            return True
    for class_meta in _project_meta_cached.obj_classes:
        if class_meta not in project_meta.obj_classes:
            return True
    return False
