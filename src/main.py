import os

import src.globals as g
import src.utils as u
import supervisely as sly
import logging
from fastapi.responses import JSONResponse
from fastapi import Depends, BackgroundTasks
import asyncio
import dataset_tools as dtools
from supervisely.io.fs import (
    get_file_name_with_ext,
    get_file_name,
    list_files,
    get_file_size,
    list_files_recursively,
)

from io import StringIO
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

# Global variable for asynchronous interruption
interrupt_get_stats = asyncio.Event()

lock = asyncio.Lock()


@server.get("/get-stats", response_class=Response)
async def stats_endpoint(
    response: Response, project_id: int, background_tasks: BackgroundTasks
):
    background_tasks.add_task(write_log_message, "Interrupting get-stats endpoint")

    project = g.api.project.get_info_by_id(project_id, raise_error=True)
    team = g.api.team.get_info_by_id(project.team_id)

    tf_cache_dir = f"{g.TF_STATS_DIR}/_cache"
    tf_project_dir = f"{g.TF_STATS_DIR}/{project_id}_{project.name}"

    g.initialize_global_cache()
    async with lock:
        force_stats_recalc = await u.pull_cache(
            team.id, project_id, tf_cache_dir, tf_project_dir
        )

    json_project_meta = g.api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(json_project_meta)

    sly.logger.info(f"Processing for the '{project.name}' project")
    sly.logger.info(f"with the PROJECT_ID={project_id}")
    sly.logger.info(f"with the CHUNK_SIZE={g.CHUNK_SIZE} (images per batch)")
    sly.logger.info(
        f"The project consists of {project.items_count} images and has {project.datasets_count} datasets"
    )

    async with lock:
        updated_images, updated_classes = await u.get_updated_images_and_classes(
            project, project_meta, force_stats_recalc
        )
    if len(updated_images) == 0:
        sly.logger.info("Nothing to update. Skipping stats calculation...")
        response.status_code = status.HTTP_200_OK
        response.body = b"Nothing to update. Skipping stats calculation..."
        return response

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

    if g.api.file.dir_exists(team.id, tf_project_dir) is True:
        async with lock:
            await u.download_stats_chunks_to_buffer(
                team.id, tf_project_dir, project_fs_dir, stats
            )

    # Check for interruption signal
    if interrupt_get_stats.is_set():
        response.body = b"Stats calculation interrupted."
        response.status_code = status.HTTP_200_OK
        return response

    files_fs = list_files_recursively(project_fs_dir, valid_extensions=[".npy"])
    async with lock:
        await u.check_datasets_consistency(project, datasets, files_fs, len(stats))
    async with lock:
        await u.remove_junk(project, datasets, files_fs)

    idx_to_infos, infos_to_idx = await u.get_indexes_dct(project_id, datasets)
    async with lock:
        updated_images = await u.check_idxs_integrity(
            project, stats, project_fs_dir, idx_to_infos, updated_images
        )

    tf_all_paths = [
        info.path for info in g.api.file.list2(team.id, tf_project_dir, recursive=True)
    ]

    sly.logger.info(f"Start calculating stats for {len(updated_images)} images")
    async with lock:
        await u.calculate_and_save_stats(
            datasets,
            project_meta,
            updated_images,
            stats,
            tf_all_paths,
            project_fs_dir,
            idx_to_infos,
            infos_to_idx,
        )

    async with lock:
        await u.delete_old_chunks(team.id)
    async with lock:
        await u.sew_chunks_to_json_and_upload_chunks(
            team.id, stats, project_fs_dir, tf_project_dir, updated_classes
        )
    async with lock:
        await u.upload_sewed_stats(team.id, project_fs_dir, tf_project_dir)

    async with lock:
        await u.push_cache(team.id, project_id, tf_cache_dir)

    response.body = f"The stats for {len(updated_images)} images were calculated."
    response.status_code = status.HTTP_200_OK
    if sly.is_production():
        file = g.api.file.get_info_by_path(
            team.id, tf_project_dir + "/class_balance.json"
        )
        g.api.task.set_output_directory(g.api.task_id, file.id, tf_cache_dir)
        # sly.logger.info(f"task id: {g.api.task_id}, file id: {file.id}")
        #  makes no sence because the app is not stopped.
    return response


log_stream = StringIO()
sly.logger.addHandler(logging.StreamHandler(log_stream))


@server.post("/log-message")
async def log_message(background_tasks: BackgroundTasks, message: str):
    background_tasks.add_task(write_log_message, message)
    return {"message": "Log message received"}


async def write_log_message(message: str):
    sly.logger.info(message)


@server.get("/long_process")
async def long_process():
    print("Long process started")
    for i in range(10):
        async with lock:
            await asyncio.sleep(1)
            print(f"Long process: {i+1} second(s) passed")
    print("Long process finished")
    return {"message": "Long process finished"}


@server.get("/interrupt")
async def interrupt():
    print("Interrupt started")
    async with lock:
        await asyncio.sleep(2)
        print("Interrupt finished")
    return {"message": "Interrupt finished"}
