import os
import json

from collections import defaultdict
import src.globals as g
import src.utils as u
import supervisely as sly
from supervisely import ProjectInfo, TeamInfo
import dataset_tools as dtools
from supervisely.io.fs import (
    get_file_name_with_ext,
    get_file_name,
    list_files,
    get_file_size,
    list_files_recursively,
)
from datetime import datetime, timezone
import time
import threading
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from supervisely.app.widgets import Container
from src.ui.input import card_1


layout = Container(widgets=[card_1], direction="vertical")
static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)
# app = sly.Application()
server = app.get_server()


TIMELOCK_LIMIT = 100  # seconds


def _get_extra(user_id, team, workspace, project) -> dict:
    if project is None or team is None or workspace is None:
        if user_id is not None:
            return {"USER_ID": user_id}
    else:
        if user_id is not None:
            return {
                "USER_ID": user_id,
                "TEAM_ID": team.id,
                "WORKSPACE_ID": workspace.id,
                "PROJECT_ID": project.id,
            }
    return None


@server.get("/get-stats")
def stats_endpoint(project_id: int, user_id: int = None):

    project = None
    team = None
    workspace = None

    try:
        project = g.api.project.get_info_by_id(project_id, raise_error=True)
        team = g.api.team.get_info_by_id(project.team_id, raise_error=True)
        workspace = g.api.workspace.get_info_by_id(project.workspace_id, raise_error=True)

        result = main_func(team, project)

    except Exception as e:
        msg = e.__class__.__name__ + ": " + str(e)
        xtr = _get_extra(user_id, team, workspace, project)
        sly.logger.error(msg, extra=xtr)

        active_project_path = f"{g.ACTIVE_REQUESTS_DIR}/{project_id}"
        active_project_path_tf = f"{g.TF_ACTIVE_REQUESTS_DIR}/{project_id}"
        sly.fs.silent_remove(active_project_path)
        tf_project_dir = f"{g.TF_STATS_DIR}/{project.id}_{project.name}"
        project_fs_dir = f"{g.STORAGE_DIR}/{project.id}_{project.name}"
        if team is not None:
            g.api.file.remove(team.id, active_project_path_tf)
            u.add_heatmaps_status_ok(team, tf_project_dir, project_fs_dir)

        raise HTTPException(
            status_code=500,
            detail={
                "title": "The app has got the following error:",
                "message": msg,
            },
        ) from e

    return result


def _remove_old_active_project_request(now, team, file):
    if sly.is_development():
        g.api.file.remove(team.id, file.path)
    dt = datetime.fromisoformat(file.updated_at[:-1]).replace(tzinfo=timezone.utc)
    if (now - dt).seconds > TIMELOCK_LIMIT:
        g.api.file.remove(team.id, file.path)
        sly.logger.debug(
            f"The temporary file {file.path!r} has been removed from tf because of time limit ({TIMELOCK_LIMIT} secs). TEAM_ID={team.id}"
        )


def main_func(team: TeamInfo, project: ProjectInfo):

    sly.logger.debug("Checking requests...")

    active_project_path_local = f"{g.ACTIVE_REQUESTS_DIR}/{project.id}"
    active_project_path_tf = f"{g.TF_ACTIVE_REQUESTS_DIR}/{project.id}"

    file = g.api.file.get_info_by_path(team.id, active_project_path_tf)
    if file is not None:
        now = datetime.now(timezone.utc)
        _remove_old_active_project_request(now, team, file)
        if g.api.file.exists(team.id, file.path) is True:
            msg = f"Request for the project with ID={project.id} is busy. Wait until the previous one will be finished..."
            sly.logger.info(msg)
            while True:
                if g.api.file.exists(team.id, active_project_path_tf):
                    now = datetime.now(timezone.utc)
                    _remove_old_active_project_request(now, team, file)
                    time.sleep(5)
                else:
                    break
            return JSONResponse({"message": msg})

    Path(active_project_path_local).touch()
    g.api.file.upload(team.id, active_project_path_local, active_project_path_tf)

    sly.logger.info("Start Quality Assurance.")

    tf_project_dir = f"{g.TF_STATS_DIR}/{project.id}_{project.name}"
    project_fs_dir = f"{g.STORAGE_DIR}/{project.id}_{project.name}"

    force_stats_recalc = False
    force_stats_recalc, _cache = u.pull_cache(team.id, project.id, tf_project_dir, project_fs_dir)

    json_project_meta = g.api.project.get_meta(project.id)
    project_meta = sly.ProjectMeta.from_json(json_project_meta)
    datasets = g.api.dataset.get_list(project.id)
    project_stats = g.api.project.get_stats(project.id)

    sly.logger.info(f"Processing for the '{project.name}' project")
    sly.logger.info(f"with the PROJECT_ID={project.id}")
    sly.logger.info(f"with the CHUNK_SIZE={g.CHUNK_SIZE} (images per batch)")
    sly.logger.info(
        f"The project consists of {project.items_count} images and has {project.datasets_count} datasets"
    )

    stats = [
        dtools.ClassBalance(project_meta, project_stats),
        dtools.ClassCooccurrence(project_meta),
        dtools.ClassesPerImage(project_meta, project_stats, datasets),
        dtools.ObjectsDistribution(project_meta),
        dtools.ObjectSizes(project_meta, project_stats),
        dtools.ClassSizes(project_meta),
        dtools.ClassesTreemap(project_meta),
        dtools.TagsImagesCooccurrence(project_meta),
        dtools.TagsObjectsCooccurrence(project_meta),
        dtools.ClassToTagCooccurrence(project_meta),
        dtools.TagsImagesOneOfDistribution(project_meta),
        dtools.TagsObjectsOneOfDistribution(project_meta),
    ]

    heatmaps = dtools.ClassesHeatmaps(project_meta, project_stats)

    if sly.fs.dir_exists(project_fs_dir):
        sly.fs.clean_dir(project_fs_dir)
    os.makedirs(project_fs_dir, exist_ok=True)

    if g.api.file.dir_exists(team.id, tf_project_dir):
        for stat in stats:
            path = f"{tf_project_dir}/{stat.basename_stem}.json"
            if not g.api.file.exists(team.id, path):
                force_stats_recalc = True
                sly.logger.warning(
                    f"The calcuated stat {stat.basename_stem!r} not exists. Forcing full stats recalculation..."
                )
        if not g.api.file.exists(team.id, f"{tf_project_dir}/{heatmaps.basename_stem}.png"):
            force_stats_recalc = True
            sly.logger.warning(
                f"The calcuated stat {heatmaps.basename_stem!r} not exists. Forcing full stats recalculation..."
            )

    images_all_dct = u.get_project_images_all(datasets)
    updated_images, updated_classes, _cache = u.get_updated_images_and_classes(
        project, project_meta, datasets, images_all_dct, force_stats_recalc, _cache
    )
    total_updated = sum(len(lst) for lst in updated_images.values())
    if total_updated == 0:
        sly.logger.info("Nothing to update. Skipping stats calculation...")
        g.api.file.remove(team.id, active_project_path_tf)
        u.add_heatmaps_status_ok(team, tf_project_dir, project_fs_dir)
        return JSONResponse({"message": "Nothing to update. Skipping stats calculation..."})

    if (
        g.api.file.dir_exists(team.id, tf_project_dir) is True
        and total_updated < project.items_count
    ):
        force_stats_recalc = u.download_stats_chunks_to_buffer(
            team.id, project, tf_project_dir, project_fs_dir, force_stats_recalc
        )
        tf_status_path = f"{tf_project_dir}/_cache/heatmaps/status_ok"
        g.api.file.remove(team.id, tf_status_path)

    idx_to_infos, infos_to_idx = u.get_indexes_dct(project.id, datasets, images_all_dct)
    updated_images = u.check_idxs_integrity(
        project,
        datasets,
        stats,
        project_fs_dir,
        idx_to_infos,
        updated_images,
        images_all_dct,
        force_stats_recalc,
    )

    tf_all_paths = [info.path for info in g.api.file.list2(team.id, tf_project_dir, recursive=True)]

    heatmaps_image_ids, heatmaps_figure_ids = u.calculate_stats_and_save_chunks(
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

    sly.logger.debug("Start threading of 'calculate_and_save_heatmaps'")
    thread1 = threading.Thread(
        target=u.calculate_and_upload_heatmaps,
        args=(
            team,
            tf_project_dir,
            project_fs_dir,
            heatmaps,
            heatmaps_image_ids,
            heatmaps_figure_ids,
        ),
    )
    thread1.start()

    sly.logger.debug("Start threading of 'archive_chunks_and_upload'")
    thread2 = threading.Thread(
        target=u.archive_chunks_and_upload,
        args=(team, project, stats, tf_project_dir, project_fs_dir),
    )
    thread2.start()

    u.upload_sewed_stats(team.id, project_fs_dir, tf_project_dir)
    u.push_cache(team.id, project.id, tf_project_dir, project_fs_dir, _cache)
    # sly.fs.silent_remove(active_project_path)
    g.api.file.remove(team.id, active_project_path_tf)
    return JSONResponse({"message": f"The stats for {total_updated} images were calculated."})
