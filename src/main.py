import os
import src.globals as g
import src.utils as u
import supervisely as sly
import asyncio
import concurrent.futures
import dataset_tools as dtools
from supervisely.io.fs import (
    get_file_name_with_ext,
    get_file_name,
    list_files,
    get_file_size,
    list_files_recursively,
)
import time
import threading
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi_utils.tasks import repeat_every
from supervisely.app.widgets import Container
from src.ui.input import card_1
from concurrent.futures import ThreadPoolExecutor

layout = Container(widgets=[card_1], direction="vertical")
static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)
# app = sly.Application()
server = app.get_server()


@server.get("/ping")
def test_ping():
    """test asynchronous behaviour"""
    return JSONResponse("ping")


@server.on_event("startup")
@repeat_every(seconds=60 * 60)  # 1 hour
def clean_active_requests() -> None:
    sly.fs.clean_dir(g.ACTIVE_REQUESTS_DIR)
    sly.logger.debug(f"The '{g.ACTIVE_REQUESTS_DIR}' has been cleaned with a scheduler.")


@server.get("/clean-active-requests")
def clean_set():
    sly.fs.clean_dir(g.ACTIVE_REQUESTS_DIR)
    sly.logger.debug(f"The '{g.ACTIVE_REQUESTS_DIR}' has been cleaned manually.")


@server.get("/get-stats")
def stats_endpoint(project_id: int):
    try:
        result = main_func(project_id)
    except Exception as e:
        msg = e.__class__.__name__ + ": " + str(e)
        sly.logger.error(msg)
        active_project_path = f"{g.ACTIVE_REQUESTS_DIR}/{project_id}"
        sly.fs.silent_remove(active_project_path)
        raise HTTPException(
            status_code=500,
            detail={
                "title": "The app has got the following error:",
                "message": msg,
            },
        ) from e

    return result


def main_func(project_id: int):

    active_project_path = f"{g.ACTIVE_REQUESTS_DIR}/{project_id}"
    if os.path.isfile(active_project_path):
        msg = f"Request for the project with ID={project_id} is busy. Wait until the previous one will be finished..."
        sly.logger.info(msg)
        while True:
            if os.path.isfile(active_project_path):
                time.sleep(5)
            else:
                break
        return JSONResponse({"message": msg})

    Path(active_project_path).touch()

    sly.logger.info("Start Quality Assurance.")

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
        sly.fs.silent_remove(active_project_path)
        return JSONResponse({"message": "Nothing to update. Skipping stats calculation..."})

    # updated_images = u.get_project_images_all(project, datasets)  # !tmp

    if (
        g.api.file.dir_exists(team.id, tf_project_dir) is True
        and total_updated < project.items_count
    ):
        force_stats_recalc = u.download_stats_chunks_to_buffer(
            team.id, project, tf_project_dir, project_fs_dir, force_stats_recalc
        )

    # u.remove_junk(team.id, tf_project_dir, project, datasets, project_fs_dir)

    idx_to_infos, infos_to_idx = u.get_indexes_dct(project_id, datasets)
    updated_images = u.check_idxs_integrity(
        project, datasets, stats, project_fs_dir, idx_to_infos, updated_images, force_stats_recalc
    )

    tf_all_paths = [info.path for info in g.api.file.list2(team.id, tf_project_dir, recursive=True)]

    u.calculate_and_save_stats(
        updated_images,
        stats,
        tf_all_paths,
        project_fs_dir,
        idx_to_infos,
        infos_to_idx,
    )
    sly.logger.info("Stats calculation finished.")
    u.remove_junk(team.id, tf_project_dir, project, datasets, project_fs_dir)

    u.sew_chunks_to_json(stats, project_fs_dir, updated_classes)
    # u.archive_chunks_and_upload(team.id, project, stats, tf_project_dir, project_fs_dir)
    sly.logger.debug("Start threading")
    thread = threading.Thread(
        target=u.archive_chunks_and_upload,
        args=(team.id, project, stats, tf_project_dir, project_fs_dir),
    )
    thread.start()

    u.upload_sewed_stats(team.id, project_fs_dir, tf_project_dir)
    u.push_cache(team.id, project_id, tf_cache_dir)
    sly.fs.silent_remove(active_project_path)
    return JSONResponse({"message": f"The stats for {total_updated} images were calculated."})
