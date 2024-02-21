from datetime import datetime
import json, os
import numpy as np
import src.globals as g
import src.utils as u
import supervisely as sly
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
from fastapi import Response, HTTPException, status
from supervisely.app.widgets import Container
from src.ui.input import card_1


layout = Container(widgets=[card_1], direction="vertical")

static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)
server = app.get_server()


@server.get("/get-stats", response_class=Response)
async def stats_endpoint(response: Response, project_id: int):
    tf_cache_dir = f"{g.TF_STATS_DIR}/_cache"

    g.initialize_global_cache()
    u.pull_cache(project_id, tf_cache_dir)

    json_project_meta = g.api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(json_project_meta)
    project = g.api.project.get_info_by_id(project_id)
    project_stats = g.api.project.get_stats(project_id)
    datasets = g.api.dataset.get_list(project_id)

    sly.logger.info(f"Processing for the '{project.name}' project")
    sly.logger.info(f"with the PROJECT_ID={project_id}")
    sly.logger.info(f"with the CHUNK_SIZE={g.CHUNK_SIZE} (images per batch)")
    sly.logger.info(
        f"The project consists of {project.items_count} images and has {len(datasets)} datasets"
    )

    updated_images = u.get_updated_images(project, project_meta)

    if len(updated_images) == 0:
        sly.logger.warn("Nothing to update. Skipping stats calculation...")
        response.status_code = status.HTTP_200_OK
        response.body = b"Nothing to update. Skipping stats calculation..."
        return response

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

    curr_projectfs_dir = f"{g.STORAGE_DIR}/{project_id}_{project.name}"
    if sly.fs.dir_exists(curr_projectfs_dir):
        sly.fs.clean_dir(curr_projectfs_dir)
    os.makedirs(curr_projectfs_dir, exist_ok=True)

    curr_tf_project_dir = f"{g.TF_STATS_DIR}/{project_id}_{project.name}"

    u.download_stats_chunks_to_buffer(curr_tf_project_dir, curr_projectfs_dir, stats)

    files_fs = list_files_recursively(curr_projectfs_dir, valid_extensions=[".npy"])
    u.check_datasets_consistency(project, datasets, files_fs, len(stats))
    u.remove_junk(project, datasets, files_fs)

    idx_to_infos, infos_to_idx = u.get_indexes_dct(project_id, datasets)
    u.check_idxs_integrity(
        project, stats, curr_projectfs_dir, idx_to_infos, updated_images
    )

    tf_all_paths = [
        info.path
        for info in g.api.file.list2(g.TEAM_ID, curr_tf_project_dir, recursive=True)
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
        curr_projectfs_dir,
        idx_to_infos,
        infos_to_idx,
    )

    u.delete_old_chunks()

    u.sew_chunks_to_stats_and_upload_chunks(
        stats, curr_projectfs_dir, curr_tf_project_dir
    )
    u.upload_sewed_stats(curr_projectfs_dir, curr_tf_project_dir)

    u.push_cache(project_id, tf_cache_dir)

    response.body = f"The stats for {len(updated_images)} images were calculated."
    response.status_code = status.HTTP_200_OK
    return response


# TODO посмотреть для измененной аннотации почему update_at не появляется
# if __name__ == "__main__":
#     tf_cache_dir = f"{g.TF_STATS_DIR}/_cache"
#     u.pull_cache(tf_cache_dir)
#     main()
#     u.push_cache(tf_cache_dir)
