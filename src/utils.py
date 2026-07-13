import json
from packaging.version import Version
import tarfile
import os
import math
import shutil
from dataclasses import dataclass
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple, Union, Set, Optional
import dataset_tools as dtools
from dataset_tools.repo.heatmap_status import HeatmapStatusReporter
from dataset_tools.image.stats.basestats import BaseStats
from datetime import datetime, timezone
from supervisely import ImageInfo, ProjectMeta, ProjectInfo, DatasetInfo, FigureInfo, TeamInfo
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
from supervisely.imaging.color import _validate_hex_color, random_rgb, rgb2hex


@dataclass
class StatsRunState:
    chunks_latest_datetime: Optional[datetime] = None


class ActiveRequestOwnershipError(RuntimeError):
    pass


def pull_cache(
    team_id: int,
    project_id: int,
    tf_project_dir: str,
    project_fs_dir: str,
    run_state: StatsRunState,
) -> Tuple[bool, dict]:
    _cache = {}
    run_state.chunks_latest_datetime = None

    if not g.api.file.dir_exists(team_id, tf_project_dir):
        sly.logger.log(g._WARNING, "The project directory not exists in team files.")
        return True, _cache

    filename = f"{project_id}_cache.json"
    tf_cache_path = f"{tf_project_dir}/_cache/{filename}"

    local_cache_path = f"{project_fs_dir}/_cache/{filename}"

    if g.api.file.exists(team_id, tf_cache_path):
        g.api.file.download(team_id, tf_cache_path, local_cache_path)
    else:
        sly.logger.log(g._WARNING, f"The {filename!r} not exists in team files.")
        return True, _cache

    try:
        if os.path.exists(local_cache_path):
            with open(local_cache_path, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        if not isinstance(_cache, dict):
            raise TypeError("Cache root must be a JSON object")
    except (OSError, ValueError, TypeError) as e:
        sly.logger.log(
            g._WARNING,
            f"Failed to read cache file {filename!r}: {e!r}. Recalculating full stats...",
        )
        return True, {}

    images = _cache.get("images")
    meta = _cache.get("meta")
    smeta = _cache.get("stats_meta")

    if images is None:
        sly.logger.log(
            g._INFO,
            f"The key with project ID={project_id} was not found in 'images_cache.json'. "
            "Stats will be fully recalculated.",
        )
        return True, {}

    if meta is None:
        sly.logger.log(
            g._INFO,
            f"The key with project ID={project_id} was not found in 'meta_cache.json'. "
            "Stats will be fully recalculated.",
        )
        return True, {}

    try:
        ProjectMeta.from_json(meta)
    except Exception as e:
        sly.logger.log(
            g._WARNING,
            f"Invalid project meta in cache: {e!r}. Recalculating full stats...",
        )
        return True, {}

    if not isinstance(smeta, dict):
        sly.logger.log(
            g._WARNING,
            "The cache has no valid 'stats_meta'. Recalculating full stats...",
        )
        return True, {}

    if smeta.get("chunk_size", -1) != g.CHUNK_SIZE:
        sly.logger.log(g._WARNING, "The chunk size has changed. Recalculating full stats...")
        return True, {}

    chunks_dt = smeta.get("chunks_dt")
    if chunks_dt is None:
        sly.logger.log(
            g._WARNING, "The cache has no chunks datetime to verify. Recalculating full stats..."
        )
        return True, {}

    try:
        if not isinstance(chunks_dt, str) or not chunks_dt.endswith("Z"):
            raise ValueError("Chunks datetime must use the UTC 'Z' format")
        chunks_latest_datetime = datetime.fromisoformat(chunks_dt[:-1])
        if chunks_latest_datetime.tzinfo is not None:
            raise ValueError("Chunks datetime must use the UTC 'Z' format")
    except (AttributeError, TypeError, ValueError) as e:
        sly.logger.log(
            g._WARNING,
            f"Invalid chunks datetime in cache: {e!r}. Recalculating full stats...",
        )
        return True, {}

    dtools_version = smeta.get("dataset-tools")
    if dtools_version is None:
        sly.logger.log(
            g._WARNING,
            "The cache has no 'dataset-tools' version to verify. Recalculating full stats...",
        )
        return True, {}
    else:
        try:
            is_outdated = Version(dtools_version) < Version(g.MINIMUM_DTOOLS_VERSION)
        except (TypeError, ValueError) as e:
            sly.logger.log(
                g._WARNING,
                f"Invalid 'dataset-tools' version in cache: {e!r}. Recalculating full stats...",
            )
            return True, {}
        if is_outdated:
            sly.logger.log(
                g._WARNING,
                f"The cached version ({dtools_version}) of 'dataset-tools' package is less "
                f"than the required one ({g.MINIMUM_DTOOLS_VERSION}). "
                "Force statistics recalculation.",
            )
            return True, {}

    sly.logger.log(g._INFO, f"The cache file {filename!r} was pulled from team files")

    try:
        images = {int(k): v for k, v in images.items()}
    except (AttributeError, TypeError, ValueError) as e:
        sly.logger.log(
            g._WARNING,
            f"Invalid images mapping in cache: {e!r}. Recalculating full stats...",
        )
        return True, {}

    run_state.chunks_latest_datetime = chunks_latest_datetime
    _cache["stats_meta"] = smeta
    _cache["meta"] = meta
    _cache["images"] = images
    return False, _cache


def get_iso_timestamp():
    now = datetime.now()
    ts = datetime.timestamp(now)
    dt = datetime.utcfromtimestamp(ts)
    return str(dt.isoformat()) + "Z"


def push_cache(
    team_id: int,
    project_id: int,
    tf_project_dir: str,
    project_fs_dir: str,
    _cache: dict,
    run_state: StatsRunState,
) -> dict:
    filename = f"{project_id}_cache.json"
    tf_cache_path = f"{tf_project_dir}/_cache/{filename}"

    local_cache_dir = f"{project_fs_dir}/_cache"
    local_cache_path = f"{local_cache_dir}/{filename}"

    ts_utc = get_iso_timestamp()
    if run_state.chunks_latest_datetime is None:
        raise ValueError("The latest chunks datetime is not initialized")
    chunks_dt = str(run_state.chunks_latest_datetime.isoformat()) + "Z"

    try:
        actual_version = dtools.__version__
    except Exception:
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
    sly.logger.log(g._INFO, f"The cache file {filename!r} was pushed to team files")

    # remove old junk
    g.api.file.remove_dir(team_id, f"{os.path.dirname(tf_project_dir)}/_cache/", silent=True)

    return _cache


@sly.timeit
def get_project_images_all(datasets: List[DatasetInfo]) -> Dict[int, ImageInfo]:
    return {d.id: g.api.image.get_list(d.id) for d in datasets}


def get_changed_object_class_ids(
    project_meta: ProjectMeta, _project_meta_cached: Union[ProjectMeta, dict]
) -> Set[int]:
    if _project_meta_cached is None:
        return set()

    cached = {x.sly_id: x.name for x in _project_meta_cached.obj_classes}
    actual = {x.sly_id: x.name for x in project_meta.obj_classes}

    removed_ids = set(cached) - set(actual)
    renamed_ids = {
        class_id for class_id in set(cached) & set(actual) if cached[class_id] != actual[class_id]
    }
    return removed_ids | renamed_ids


@sly.timeit
def get_updated_images_and_classes(
    project: ProjectInfo,
    project_meta: ProjectMeta,
    datasets: List[DatasetInfo],
    images_all_dct,
    force_stats_recalc: bool,
    _cache: dict,
) -> Tuple[Dict[int, List[ImageInfo]], Set[int], dict, bool]:
    _images_cached = _cache.get("images", {})
    _meta_cached_json = _cache.get("meta")
    _project_meta_cached = ProjectMeta.from_json(_meta_cached_json) if _meta_cached_json else None
    is_meta_changed = compare_metas(project_meta, _project_meta_cached)

    images_all_flat = []
    for value in images_all_dct.values():
        images_all_flat.extend(value)

    updated_images, changed_object_class_ids = {d.id: [] for d in datasets}, set()
    images_updated_at = {}
    for image in images_all_flat:
        images_updated_at[image.id] = image.updated_at

    _cache["images"] = images_updated_at
    _cache["meta"] = project_meta.to_json()

    if force_stats_recalc is True:
        return images_all_dct, set(), _cache, is_meta_changed

    if _project_meta_cached is not None:
        changed_object_class_ids = get_changed_object_class_ids(project_meta, _project_meta_cached)
        if len(changed_object_class_ids) > 0:
            sly.logger.log(
                g._INFO,
                f"Object class metadata changes detected for class IDs: {sorted(changed_object_class_ids)}",
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
            sly.logger.log(
                g._INFO, f"The images with the following ids were added: {set_B - set_A}"
            )
        elif set_B.issubset(set_A):
            sly.logger.log(
                g._INFO, f"The images with the following ids were deleted: {set_A - set_B}"
            )

        sly.logger.log(g._INFO, "Recalculate full statistics")
        return images_all_dct, set(), _cache, is_meta_changed

    num_updated = sum(len(lst) for lst in updated_images.values())
    if num_updated == getattr(project, "items_count", 0):
        sly.logger.log(g._INFO, "Full dataset statistics will be calculated.")
    elif num_updated > 0:
        sly.logger.log(g._INFO, f"The changes in {num_updated} images detected")

    return updated_images, changed_object_class_ids, _cache, is_meta_changed


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
    project: ProjectInfo,
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
        sly.logger.log(g._WARNING, "The buffer is empty. Calculate full stats")
        if any(len(x) != d.items_count for x, d in zip(updated_images.values(), datasets)):
            total_updated = sum(len(lst) for lst in updated_images.values())
            sly.logger.log(
                g._WARNING,
                f"The number of updated images ({total_updated}) should equal to the number "
                f"of images ({project.items_count}) in the project. Possibly the problem "
                "with cached files. Forcing recalculation...",
            )
            return images_all_dct
    else:
        try:
            expected_chunks = set(idx_to_infos)
            for stat in stats:
                files = sly.fs.list_files(
                    f"{projectfs_dir}/{stat.basename_stem}",
                    [".npy"],
                )
                parsed_chunks = [_parse_chunk_filename(path) for path in files]
                actual_chunks = [
                    parsed[0] for parsed in parsed_chunks if parsed is not None
                ]

                if (
                    any(parsed is None for parsed in parsed_chunks)
                    or any(parsed[1] != g.CHUNK_SIZE for parsed in parsed_chunks)
                    or len(actual_chunks) != len(expected_chunks)
                    or set(actual_chunks) != expected_chunks
                ):
                    msg = (
                        "The number of images in the project has changed. Check chunks in "
                        f"Team Files: {projectfs_dir}/{stat.basename_stem}. "
                        "Forcing recalculation..."
                    )
                    sly.logger.log(g._WARNING, msg)
                    return images_all_dct

                for path in files:
                    np.load(path, allow_pickle=True)
        except Exception:
            sly.logger.log(g._WARNING, "Error while integrity checking. Recalc full stats.")
            return images_all_dct

    return updated_images


def add_changed_class_chunks_to_updated_images(
    updated_images: Dict[int, List[ImageInfo]],
    changed_object_class_ids: Set[int],
    project_fs_dir: str,
    chunk_to_images: Dict[str, List[ImageInfo]],
    image_to_chunk: Dict[int, str],
    images_all_dct: Dict[int, List[ImageInfo]],
) -> Tuple[Dict[int, List[ImageInfo]], int, bool]:
    if len(changed_object_class_ids) == 0:
        return updated_images, 0, False

    class_balance_dir = f"{project_fs_dir}/class_balance"

    try:
        if not sly.fs.dir_exists(class_balance_dir):
            raise FileNotFoundError(
                f"Class balance chunks directory not found: {class_balance_dir}"
            )

        files = list_files(class_balance_dir, [".npy"])
        if len(files) == 0:
            raise FileNotFoundError(f"Class balance chunk files not found: {class_balance_dir}")

        affected_chunks = set()
        for file in files:
            loaded_data = np.load(file, allow_pickle=True).tolist()
            if loaded_data is None:
                continue
            if (
                not isinstance(loaded_data, (list, tuple))
                or len(loaded_data) == 0
                or not isinstance(loaded_data[0], dict)
            ):
                raise ValueError(f"Unexpected class balance chunk format: {file}")

            images_by_class = loaded_data[0]
            for class_id in changed_object_class_ids:
                for image_id in images_by_class.get(class_id, set()):
                    chunk = image_to_chunk.get(image_id)
                    if chunk is None:
                        raise KeyError(
                            f"Image ID={image_id} from class balance chunk is missing in current indexes"
                        )
                    affected_chunks.add(chunk)

        if len(affected_chunks) == 0:
            sly.logger.log(
                g._INFO,
                "Object class metadata changed, but affected classes were not found in cached chunks.",
            )
            return updated_images, 0, False

        existing_image_ids = {image.id for images in updated_images.values() for image in images}
        added_images = 0
        for chunk in affected_chunks:
            images_chunk = chunk_to_images.get(chunk)
            if images_chunk is None:
                raise KeyError(f"Chunk {chunk!r} is missing in current indexes")
            for image in images_chunk:
                if image.id in existing_image_ids:
                    continue
                updated_images.setdefault(image.dataset_id, []).append(image)
                existing_image_ids.add(image.id)
                added_images += 1

        sly.logger.log(
            g._INFO,
            f"Object class metadata changed. Recalculating {len(affected_chunks)} affected "
            f"chunks ({added_images} additional images).",
        )
        return updated_images, len(affected_chunks), False

    except Exception as e:
        sly.logger.log(
            g._WARNING,
            "Failed to resolve chunks affected by object class metadata changes: "
            f"{repr(e)}. Recalculating full statistics.",
        )
        return images_all_dct, len(chunk_to_images), True


def check_datasets_consistency(project_info, datasets, npy_paths, num_stats):
    for dataset in datasets:
        actual_ceil = math.ceil(dataset.items_count / g.CHUNK_SIZE)
        max_chunks = math.ceil(
            len([path for path in npy_paths if f"_{dataset.id}_" in sly.fs.get_file_name(path)])
            / num_stats
        )
        if actual_ceil < max_chunks:
            raise ValueError(
                f"The number of chunks per stat ({len(npy_paths)}) not match with the total "
                f"items count of the project ({project_info.items_count}) using following "
                f"batch size: {g.CHUNK_SIZE}. Details: DATASET_ID={dataset.id}; actual num "
                f"of chunks: {actual_ceil}; max num of chunks: {max_chunks}"
            )
    sly.logger.log(g._INFO, "The consistency of data is OK")


@sly.timeit
def remove_junk(
    team_id,
    tf_project_dir,
    project,
    datasets,
    project_fs_dir,
    run_state: StatsRunState,
):
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
        if path in old_paths:
            continue
        parsed = _parse_chunk_filename(path)
        if parsed is None:
            os.remove(path)
            rm_cnt += 1
            continue

        identifier, chunk_size, _ = parsed
        _, _, dataset_id, project_id = identifier.rsplit("_", 3)
        if (
            dataset_id not in ds_ids
            or project_id != str(project.id)
            or chunk_size != g.CHUNK_SIZE
        ):
            os.remove(path)
            rm_cnt += 1

    if rm_cnt > 0:
        sly.logger.log(
            g._INFO,
            f"The {rm_cnt} old or junk chunk files were detected and removed from the buffer",
        )

    chunks_archive = [
        f for f in g.api.file.listdir(team_id, tf_project_dir) if f.endswith(".tar.gz")
    ]
    if len(chunks_archive) > 1:
        if run_state.chunks_latest_datetime is None:
            raise ValueError("The latest chunks datetime is not initialized")
        for chunks in chunks_archive:
            tf_chunks_dt = ".".join(sly.fs.get_file_name(chunks).split(".")[:-1]).split("_")[-1]
            if tf_chunks_dt != run_state.chunks_latest_datetime.isoformat():
                g.api.file.remove_file(team_id, chunks)
                sly.logger.log(
                    g._INFO,
                    f"The {chunks} old or junk chunks archive was detected and removed from the team files.",
                )


@sly.timeit
def download_stats_chunks_to_buffer(
    team_id,
    project: ProjectInfo,
    tf_project_dir,
    project_fs_dir,
    force_stats_recalc,
    run_state: StatsRunState,
    stats: List[BaseStats],
) -> bool:
    if force_stats_recalc:
        return True

    if run_state.chunks_latest_datetime is None:
        sly.logger.log(
            g._WARNING,
            "The chunks identifier of latest datetime is not existed.  Recalculating full stats.",
        )
        return True
    cached_chunks_dt = run_state.chunks_latest_datetime.isoformat()
    archive_name = f"{project.id}_{project.name}_chunks_{cached_chunks_dt}.tar.gz"
    src_path = f"{tf_project_dir}/{archive_name}"
    dst_path = f"{project_fs_dir}/{archive_name}"

    file = g.api.file.get_info_by_path(team_id, src_path)
    if file is None:
        sly.logger.log(
            g._WARNING,
            f"The chunks archive file is not existed: '{archive_name}'.  Recalculating full stats.",
        )
        return True
    tf_chunks_dt = ".".join(sly.fs.get_file_name(file.path).split(".")[:-1]).split("_")[-1]
    if cached_chunks_dt != tf_chunks_dt:
        sly.logger.log(
            g._WARNING,
            f"The chunks datetime '{tf_chunks_dt}' differs from the cached one: "
            f"'{cached_chunks_dt}'.  Recalculating full stats.",
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
        except Exception:
            sly.logger.log(
                g._WARNING, "The integrity of the team files is broken. Recalculating full stats."
            )
            return True

    try:
        with TemporaryDirectory(
            prefix=f".chunks-{project.id}-", dir=os.path.dirname(project_fs_dir)
        ) as staging_dir:
            with tarfile.open(dst_path, "r:gz") as tar:
                members = tar.getmembers()
                _validate_chunks_archive_members(
                    members, {stat.basename_stem for stat in stats}
                )
                tar.extractall(staging_dir, members=members, filter="data")
            shutil.copytree(staging_dir, project_fs_dir, dirs_exist_ok=True)
    except (OSError, tarfile.TarError, ValueError) as e:
        sly.logger.log(
            g._WARNING,
            f"Failed to safely extract chunks archive: {e!r}. Recalculating full stats.",
        )
        sly.fs.clean_dir(project_fs_dir)
        return True

    return False


def _validate_chunks_archive_members(members, allowed_stat_dirs: Set[str]):
    for member in members:
        parts = tuple(
            part for part in PurePosixPath(member.name).parts if part not in {"", "."}
        )
        if ".." in parts:
            raise ValueError(f"Unsafe path in chunks archive: {member.name!r}")
        if member.isdir() and len(parts) == 1 and parts[0] in allowed_stat_dirs:
            continue
        if (
            member.isreg()
            and len(parts) == 2
            and parts[0] in allowed_stat_dirs
            and parts[-1].endswith(".npy")
            and _parse_chunk_filename(parts[-1]) is not None
        ):
            continue
        raise ValueError(f"Unexpected member in chunks archive: {member.name!r}")


@sly.timeit
def calculate_stats_and_save_chunks(
    updated_images,
    stats,
    tf_all_paths,
    project_fs_dir,
    chunk_to_images,
    image_to_chunk,
    run_state: StatsRunState,
) -> None:
    total_updated = sum(len(lst) for lst in updated_images.values())
    if total_updated == 0 and run_state.chunks_latest_datetime is None:
        run_state.chunks_latest_datetime = datetime.now(timezone.utc).replace(tzinfo=None)
        for stat in stats:
            os.makedirs(f"{project_fs_dir}/{stat.basename_stem}", exist_ok=True)

    sly.logger.log(g._INFO, f"Start calculating stats for {total_updated} images.")
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

                    pbar.update(len(batch_infos))

                latest_datetime = get_latest_datetime(images_chunk)
                if (
                    run_state.chunks_latest_datetime is None
                    or run_state.chunks_latest_datetime < latest_datetime
                ):
                    run_state.chunks_latest_datetime = latest_datetime
                for stat in stats:
                    save_chunks(stat, chunk, project_fs_dir, tf_all_paths, latest_datetime)
                    stat.clean()

        # if pbar.last_print_n < pbar.total:  # unlabeled images
        #     pbar.update(pbar.total - pbar.n)


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
        path
        for path in tf_all_paths
        if os.path.basename(os.path.dirname(path)) == stat.basename_stem
        and _get_chunk_identifier(path) == chunk
    ]

    if len(tf_stat_chunks) > 0:
        timestamps = [get_file_name(path).split("_")[-1] for path in tf_stat_chunks]
        datetime_objects = [datetime.fromisoformat(timestamp) for timestamp in timestamps]
        if latest_datetime > sorted(datetime_objects, reverse=True)[0]:
            for path in list_files(savedir, [".npy"]):
                if _get_chunk_identifier(path) == chunk:
                    os.remove(path)

    np.save(
        f"{savedir}/{chunk}_{g.CHUNK_SIZE}_{latest_datetime.isoformat()}.npy",
        stat.to_numpy_raw(),
    )


def _get_chunk_identifier(path: str) -> Optional[str]:
    parsed = _parse_chunk_filename(path)
    return parsed[0] if parsed is not None else None


def _parse_chunk_filename(path: str) -> Optional[Tuple[str, int, datetime]]:
    parts = get_file_name(path).rsplit("_", 2)
    if len(parts) != 3:
        return None
    identifier_parts = parts[0].rsplit("_", 3)
    if len(identifier_parts) != 4 or identifier_parts[0] != "chunk":
        return None
    try:
        for identifier_part in identifier_parts[1:]:
            int(identifier_part)
        return parts[0], int(parts[1]), datetime.fromisoformat(parts[2])
    except (TypeError, ValueError):
        return None


@sly.timeit
def sew_chunks_to_json(
    stats: List[BaseStats], project_fs_dir, _changed_object_class_ids, is_meta_changed: bool
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
            if isinstance(res, dict) and isinstance(res.get("columnsOptions"), list):
                res["columnsOptions"] = [
                    ({} if opt is None else opt) for opt in res["columnsOptions"]
                ]
            _save_to_json(res, f"{project_fs_dir}/{stat.basename_stem}.json")


def collect_heatmap_sample(images_all_dct, project_stats: dict, project):
    heatmaps_image_ids = defaultdict(set)
    heatmaps_figure_ids = defaultdict(set)
    total_figures = sum(image.labels_count for images in images_all_dct.values() for image in images)
    total_images = sum(len(images) for images in images_all_dct.values())

    with tqdm(desc="Collecting heatmap sample", total=total_images) as pbar:
        for dataset_id, images in images_all_dct.items():
            for batch_infos in sly.batched(images, 100):
                batch_ids = [image.id for image in batch_infos]
                figures = g.api.image.figure.download(dataset_id, batch_ids, skip_geometry=True)
                for image in batch_infos:
                    _update_heatmaps_sample(
                        heatmaps_figure_ids,
                        heatmaps_image_ids,
                        figures.get(image.id, []),
                        total_figures,
                        project_stats["objects"]["total"]["objectsInDataset"],
                        project.size,
                    )
                pbar.update(len(batch_infos))

    return heatmaps_image_ids, heatmaps_figure_ids


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
    storage_dir: str,
    project_id: int,
    heatmaps: dtools.ClassesHeatmaps,
    heatmaps_image_ids: Dict[int, Set[int]],
    heatmaps_figure_ids: Dict[int, Set[int]],
    before_publish=None,
):
    heatmaps_name = f"{heatmaps.basename_stem}.png"
    tf_heatmap_path = f"{tf_project_dir}/{heatmaps_name}"
    heatmaps_status = HeatmapStatusReporter(g.api, project_id, logger=sly.logger)

    def ensure_publish_allowed():
        if before_publish is not None:
            before_publish()

    stage = "preparing"
    try:
        ensure_publish_allowed()
        if len(heatmaps_image_ids) == 0:
            if g.api.file.exists(team.id, tf_heatmap_path):
                g.api.file.remove(team.id, tf_heatmap_path)
            ensure_publish_allowed()
            heatmaps_status.skipped(
                "No sampled object annotations for heatmap generation.",
                output_path=tf_heatmap_path,
            )
            return

        stage = "collecting_heatmap_sample"
        sample_total = sum(len(lst) for lst in heatmaps_image_ids.values())
        ensure_publish_allowed()
        heatmaps_status.running(
            stage,
            "Collecting annotation data for heatmap generation.",
            progress=0.1,
            output_path=tf_heatmap_path,
        )
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

        stage = "rendering"
        ensure_publish_allowed()
        heatmaps_status.running(
            stage,
            "Rendering heatmap image.",
            progress=0.8,
            output_path=tf_heatmap_path,
        )
        with TemporaryDirectory(
            prefix=f".heatmaps-{project_id}-", dir=storage_dir
        ) as render_dir:
            fs_heatmap_path = os.path.join(render_dir, heatmaps_name)
            heatmaps.to_image(fs_heatmap_path)

            stage = "uploading"
            ensure_publish_allowed()
            heatmaps_status.running(
                stage,
                "Uploading heatmap image.",
                progress=0.95,
                output_path=tf_heatmap_path,
            )
            g.api.file.upload(team.id, fs_heatmap_path, tf_heatmap_path)
        sly.logger.log(g._INFO, f"The {heatmaps_name!r} file was succesfully uploaded.")
        ensure_publish_allowed()
        heatmaps_status.success(
            "Heatmap generation completed successfully.",
            output_path=tf_heatmap_path,
        )

    except ActiveRequestOwnershipError as e:
        sly.logger.warning(f"Heatmap publication cancelled after lock ownership changed: {e}")
        return
    except Exception as e:
        sly.logger.error(f"Error calculating heatmaps: {repr(e)}")
        try:
            ensure_publish_allowed()
        except ActiveRequestOwnershipError as ownership_error:
            sly.logger.warning(
                "Heatmap failure status was not published after lock ownership changed: "
                f"{ownership_error}"
            )
            return
        except Exception as ownership_check_error:
            sly.logger.warning(
                "Heatmap failure status was not published because lock ownership could "
                f"not be verified: {ownership_check_error}"
            )
            raise e from ownership_check_error
        heatmaps_status.failed(e, stage=stage, output_path=tf_heatmap_path)
        raise


@sly.timeit
def archive_chunks_and_upload(
    team: TeamInfo,
    project: ProjectInfo,
    stats: List[BaseStats],
    tf_project_dir,
    project_fs_dir,
    datasets,
    run_state: StatsRunState,
):
    def _compress_folders(folders, archive_path) -> int:
        with tarfile.open(archive_path, "w:gz") as tar:
            for folder in folders:
                tar.add(folder, arcname=os.path.basename(folder))
        return sly.fs.get_file_size(archive_path)

    folders_to_compress = [f"{project_fs_dir}/{stat.basename_stem}" for stat in stats]

    dt_identifier = run_state.chunks_latest_datetime
    if dt_identifier is None:
        raise ValueError("The latest chunks datetime is not initialized")
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

    remove_junk(team.id, tf_project_dir, project, datasets, project_fs_dir, run_state)
    sly.logger.log(g._INFO, f"The '{archive_name}' file was succesfully uploaded.")


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
        g.api.file.upload_bulk(team_id, stats_paths, dst_json_paths, pbar)

    sly.logger.log(
        g._INFO, f"{len(stats_paths)} updated .json and .png stats succesfully updated and uploaded"
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
            rgb2hex(random_rgb())
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
