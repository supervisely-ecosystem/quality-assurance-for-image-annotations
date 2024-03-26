from datetime import datetime
import json, os
import numpy as np
import src.globals as g
import src.utils as u
import supervisely as sly
import logging

from tqdm import tqdm
import dataset_tools as dtools
from supervisely.io.fs import (
    get_file_name_with_ext,
    get_file_name,
    list_files,
    get_file_size,
    list_files_recursively,
)

from pathlib import Path
import src.globals as g
import src.utils as u
import supervisely as sly
from fastapi import Response, HTTPException, status, Request
from supervisely.app.widgets import Container
from src.ui.input import card_1

layout = Container(widgets=[card_1], direction="vertical")

static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)
server = app.get_server()


@server.get("/get-stats", response_class=Response)
async def stats_endpoint(request: Request, response: Response, project_id: int):

    project = g.api.project.get_info_by_id(project_id, raise_error=True)
    team = g.api.team.get_info_by_id(project.team_id)

    tf_cache_dir = f"{g.TF_STATS_DIR}/_cache"
    tf_project_dir = f"{g.TF_STATS_DIR}/{project_id}_{project.name}"

    g.initialize_global_cache()
    force_stats_recalc = u.pull_cache(team.id, project_id, tf_cache_dir, tf_project_dir)

    json_project_meta = g.api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(json_project_meta)

    sly.logger.info(f"Processing for the '{project.name}' project")
    sly.logger.info(f"with the PROJECT_ID={project_id}")
    sly.logger.info(f"with the CHUNK_SIZE={g.CHUNK_SIZE} (images per batch)")
    sly.logger.info(
        f"The project consists of {project.items_count} images and has {project.datasets_count} datasets"
    )

    datasets = g.api.dataset.get_list(project_id)
    project_stats = g.api.project.get_stats(project_id)

    cache = {}
    stats = [
        dtools.ClassBalance(project_meta, project_stats, stat_cache=cache),
        dtools.ClassCooccurrence(project_meta),
        dtools.ClassesPerImage(project_meta, project_stats, datasets, stat_cache=cache),
        dtools.ObjectsDistribution(project_meta),
        dtools.ObjectSizes(project_meta, project_stats),
        dtools.ClassSizes(project_meta),
        dtools.ClassesTreemap(project_meta),
    ]

    project_fs_dir = f"{g.STORAGE_DIR}/{project_id}_{project.name}"
    if sly.fs.dir_exists(project_fs_dir):
        sly.fs.clean_dir(project_fs_dir)
    os.makedirs(project_fs_dir, exist_ok=True)

    updated_images, updated_classes = u.get_updated_images_and_classes(
        project, project_meta, datasets, force_stats_recalc
    )
    total_updated = sum(len(lst) for lst in updated_images.values())
    if total_updated == 0:
        sly.logger.info("Nothing to update. Skipping stats calculation...")
        response.status_code = status.HTTP_200_OK
        response.body = b"Nothing to update. Skipping stats calculation..."
        return response

    # updated_images = u.get_project_images_all(project, datasets)  # !tmp

    if (
        g.api.file.dir_exists(team.id, tf_project_dir) is True
        and total_updated < project.items_count
    ):
        force_stats_recalc = u.download_stats_chunks_to_buffer(
            team.id, tf_project_dir, project_fs_dir, stats, force_stats_recalc
        )
    files_fs = list_files_recursively(project_fs_dir, valid_extensions=[".npy"])
    u.check_datasets_consistency(project, datasets, files_fs, len(stats))
    u.remove_junk(project, datasets, files_fs)

    idx_to_infos, infos_to_idx = u.get_indexes_dct(project_id, datasets)
    updated_images = u.check_idxs_integrity(
        project, datasets, stats, project_fs_dir, idx_to_infos, updated_images
    )

    tf_all_paths = [
        info.path for info in g.api.file.list2(team.id, tf_project_dir, recursive=True)
    ]

    sly.logger.info(f"Start calculating stats for {total_updated} images")

    u.calculate_and_save_stats(
        project,
        datasets,
        project_meta,
        updated_images,
        total_updated,
        stats,
        tf_all_paths,
        project_fs_dir,
        idx_to_infos,
        infos_to_idx,
    )

    # u.delete_old_chunks(team.id)
    u.sew_chunks_to_json_and_upload_chunks(
        team.id, stats, project_fs_dir, tf_project_dir, updated_classes
    )
    u.upload_sewed_stats(team.id, project_fs_dir, tf_project_dir)

    u.push_cache(team.id, project_id, tf_cache_dir)

    response.body = f"The stats for {total_updated} images were calculated."
    response.status_code = status.HTTP_200_OK
    return response
