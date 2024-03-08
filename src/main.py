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


@server.get("/get-stats", response_class=Response)
async def stats_endpoint(response: Response, project_id: int):

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

    updated_images, updated_classes = u.get_updated_images_and_classes(
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
        u.download_stats_chunks_to_buffer(
            team.id, tf_project_dir, project_fs_dir, stats
        )

    files_fs = list_files_recursively(project_fs_dir, valid_extensions=[".npy"])
    u.check_datasets_consistency(project, datasets, files_fs, len(stats))
    u.remove_junk(project, datasets, files_fs)

    idx_to_infos, infos_to_idx = u.get_indexes_dct(project_id, datasets)
    updated_images = u.check_idxs_integrity(
        project, stats, project_fs_dir, idx_to_infos, updated_images
    )

    tf_all_paths = [
        info.path for info in g.api.file.list2(team.id, tf_project_dir, recursive=True)
    ]

    # unique_batch_sizes = set(
    #     [
    #         get_file_name(path).split("_")[-2]
    #         for path in tf_all_paths
    #         if path.endswith(".npy")
    #     ]
    # )

    # if (len(unique_batch_sizes) > 1) or (str(g.CHUNK_SIZE) not in unique_batch_sizes):
    #     g.TF_OLD_CHUNKS += [path for path in tf_all_paths if path.endswith(".npy")]
    #     sly.logger.info(
    #         "Chunk batch sizes in team files are non-unique. All chunks will be removed."
    #     )

    sly.logger.info(f"Start calculating stats for {len(updated_images)} images")
    u.calculate_and_save_stats(
        datasets,
        project_meta,
        updated_images,
        stats,
        tf_all_paths,
        project_fs_dir,
        idx_to_infos,
        infos_to_idx,
    )

    u.delete_old_chunks(team.id)
    u.sew_chunks_to_json_and_upload_chunks(
        team.id, stats, project_fs_dir, tf_project_dir, updated_classes
    )
    u.upload_sewed_stats(team.id, project_fs_dir, tf_project_dir)

    u.push_cache(team.id, project_id, tf_cache_dir)

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


# @server.get("/log-message")
async def log_message():
    sly.logger.info("This is a debug log message.")

    log_msg = get_last_log_message()
    return JSONResponse(content={"log_message": log_msg}, status_code=200)

    # try:
    #     sly.logger.info("This is a debug log message.")

    #     log_msg = get_last_log_message()
    #     return JSONResponse(content={"log_message": log_msg}, status_code=200)
    # except Exception as e:
    #     return HTTPException(detail=str(e), status_code=500)

    # sly.logger.info("This is a debug log message.")

    # background_tasks.add_task(check_logger_messages)

    # return {"message": "Log message processed in the background"}


# @server.get("/log-message")


async def check_logger_messages():
    while True:
        # Get the last log message
        log_messages = log_stream.getvalue()

        # Process or log the messages as needed (replace this with your logic)
        print("Background Task: Log Messages -", log_messages)

        # Sleep for a while before checking again
        await asyncio.sleep(2)  # Adjust the sleep interval as needed


# def get_last_log_message():
#     log_messages = log_stream.getvalue()
#     log_stream.truncate(0)
#     log_stream.seek(0)

#     return log_messages.strip()


# Start the background task
@server.on_event("startup")
async def startup_event():
    asyncio.create_task(check_logger_messages())


# from fastapi import FastAPI

# # app = FastAPI()
# import asyncio
# from time import sleep


# async def fake_async_sleep(seconds):
#     await asyncio.sleep(seconds)


# @server.get("/endpoint1")
# async def read_endpoint1():
#     await fake_async_sleep(100)
#     return {"message": "Hello from Endpoint 1"}


# @server.get("/endpoint2")
# async def read_endpoint2():
#     return {"message": "Hello from Endpoint 2"}
