import os
import src.globals as g
import src.utils as u
import supervisely as sly
from supervisely import ProjectInfo, TeamInfo, WorkspaceInfo
import dataset_tools as dtools
from datetime import datetime, timezone
import time
import threading
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from supervisely.app.widgets import Container
from src.ui.input import card_1
import fcntl


layout = Container(widgets=[card_1], direction="vertical")
static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)
server = app.get_server()


TIMELOCK_LIMIT = 60  # seconds


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

        result = main_func(user_id, team, workspace, project)

    except Exception as e:
        msg = e.__class__.__name__ + ": " + str(e)
        xtr = _get_extra(user_id, team, workspace, project)
        sly.logger.error(msg, extra=xtr)

        active_project_path = f"{g.ACTIVE_REQUESTS_DIR}/{project_id}"
        active_project_path_tf = f"{g.TF_ACTIVE_REQUESTS_DIR}/{project_id}"
        sly.fs.silent_remove(active_project_path)

        # Clean up resources if project info is available
        if project is not None:
            tf_project_dir = f"{g.TF_STATS_DIR}/{project.id}_{project.name}"
            project_fs_dir = f"{g.STORAGE_DIR}/{project.id}_{project.name}"

            # Clean up lock file on error to prevent deadlock
            lock_file_path = f"{project_fs_dir}/.processing.lock"
            sly.fs.silent_remove(lock_file_path)

            # Clean up team files if team info is available
            if team is not None:
                try:
                    g.api.file.remove(team.id, active_project_path_tf)
                except Exception as cleanup_error:
                    sly.logger.warning(f"Failed to remove active project file: {cleanup_error}")

                try:
                    u.add_heatmaps_status_ok(team, tf_project_dir, project_fs_dir)
                except Exception as cleanup_error:
                    sly.logger.warning(f"Failed to add heatmaps status: {cleanup_error}")
        elif team is not None:
            # Only remove active project file if we have team but not project info
            try:
                g.api.file.remove(team.id, active_project_path_tf)
            except Exception as cleanup_error:
                sly.logger.warning(f"Failed to remove active project file: {cleanup_error}")

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


def check_if_QA_tab_is_active(team: TeamInfo, project: ProjectInfo) -> str:
    """
    Checks if the QA tab is active for the project.
    Uses Team Files as the primary lock mechanism to prevent race conditions across multiple app instances.
    Local file lock is used only for optimization within the same app instance.

    The locking strategy:
    1. First check Team Files (global lock across all app instances)
    2. If no lock exists, use local file lock to ensure atomicity within this instance
    3. Double-check Team Files again before creating the lock (prevents race between instances)
    4. Create lock file in Team Files

    Args:
        team: TeamInfo object containing team information
        project: ProjectInfo object containing project information

    Returns:
        str: Path to the active project file in team files, or JSONResponse if project is busy
    """
    sly.logger.log(g._INFO, "Checking requests...")

    active_project_path_local = f"{g.ACTIVE_REQUESTS_DIR}/{project.id}"
    active_project_path_tf = f"{g.TF_ACTIVE_REQUESTS_DIR}/{project.id}"
    lock_file_path = f"{g.ACTIVE_REQUESTS_DIR}/{project.id}.lock"

    # First check: Check Team Files for existing lock (global check across all app instances)
    file = g.api.file.get_info_by_path(team.id, active_project_path_tf)
    if file is not None:
        now = datetime.now(timezone.utc)
        _remove_old_active_project_request(now, team, file)
        if g.api.file.exists(team.id, file.path) is True:
            msg = f"Request for the project with ID={project.id} is busy. Wait until the previous one will be finished..."
            sly.logger.log(g._INFO, msg)
            return JSONResponse({"message": msg})

    # Use local file lock for atomic operations within this app instance
    try:
        with open(lock_file_path, "w") as lock_file:
            # Try to acquire exclusive lock (non-blocking) - prevents race within same instance
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except IOError:
                # Lock is already held by another request in this app instance
                msg = f"Request for the project with ID={project.id} is busy in this app instance. Wait until the previous one will be finished..."
                sly.logger.log(g._INFO, msg)
                return JSONResponse({"message": msg})

            # Double-check Team Files before creating lock (prevents race between different instances)
            file = g.api.file.get_info_by_path(team.id, active_project_path_tf)
            if file is not None:
                now = datetime.now(timezone.utc)
                _remove_old_active_project_request(now, team, file)
                if g.api.file.exists(team.id, file.path) is True:
                    msg = f"Request for the project with ID={project.id} is busy. Wait until the previous one will be finished..."
                    sly.logger.log(g._INFO, msg)
                    return JSONResponse({"message": msg})

            # Create the lock file locally
            with open(active_project_path_local, "w") as request_file:
                request_file.write(f"Started at: {datetime.now(timezone.utc).isoformat()}")

            # Upload to Team Files - this becomes the global lock
            try:
                g.api.file.upload(team.id, active_project_path_local, active_project_path_tf)
                sly.logger.log(g._INFO, f"Created active request file for project {project.id}")
            except Exception as e:
                sly.logger.warning(f"Failed to upload active request file: {e}")
                # Clean up local file if upload failed
                sly.fs.silent_remove(active_project_path_local)
                raise

    except Exception as e:
        sly.logger.error(f"Error in check_if_QA_tab_is_active: {e}")
        raise
    finally:
        # Lock is automatically released when the file is closed
        sly.fs.silent_remove(lock_file_path)

    sly.logger.log(g._INFO, "Finish checking if 'QA & Stats' tab is active.")
    return active_project_path_tf


def main_func(user_id: int, team: TeamInfo, workspace: WorkspaceInfo, project: ProjectInfo):

    g.initialize_log_levels(project.id)

    active_project_path_tf = check_if_QA_tab_is_active(team, project)

    sly.logger.log(g._INFO, "Start Quality Assurance.")

    tf_project_dir = f"{g.TF_STATS_DIR}/{project.id}_{project.name}"
    project_fs_dir = f"{g.STORAGE_DIR}/{project.id}_{project.name}"

    force_stats_recalc = False
    force_stats_recalc, _cache = u.pull_cache(team.id, project.id, tf_project_dir, project_fs_dir)

    json_project_meta = g.api.project.get_meta(project.id)
    try:
        project_meta = sly.ProjectMeta.from_json(json_project_meta)
    except Exception:
        json_project_meta = u.handle_broken_project_meta(json_project_meta)
        project_meta = sly.ProjectMeta.from_json(json_project_meta)
    datasets = g.api.dataset.get_list(project.id, recursive=True)
    project_stats = g.api.project.get_stats(project.id)

    sly.logger.log(g._INFO, f"Processing for the '{project.name}' project")
    sly.logger.log(
        g._INFO,
        f"with the USER_ID={user_id} TEAM_ID={team.id} WORKSPACE_ID={workspace.id} PROJECT_ID={project.id}",
    )
    sly.logger.log(g._INFO, f"with the CHUNK_SIZE={g.CHUNK_SIZE} (images per batch)")
    sly.logger.log(
        g._INFO,
        f"The project consists of {project.items_count} images and has {project.datasets_count} datasets",
    )

    stats = [
        dtools.OverviewPie(project_meta, project_stats),
        dtools.OverviewDonut(project_meta, project_stats),
        dtools.ClassBalance(project_meta, project_stats),
        dtools.ClassCooccurrence(project_meta),
        dtools.ClassesPerImage(project_meta, project_stats, datasets),
        dtools.DatasetsAnnotations(project_meta, project_stats, datasets),
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
        # Additional check before cleaning to prevent conflicts with concurrent processes
        lock_file_path = f"{project_fs_dir}/.processing.lock"
        try:
            if os.path.exists(lock_file_path):
                sly.logger.log(
                    g._WARNING, f"Another process is working with {project_fs_dir}. Waiting..."
                )
                # Wait for directory to be released
                max_wait = 300  # 5 minutes maximum
                wait_time = 0
                while os.path.exists(lock_file_path) and wait_time < max_wait:
                    time.sleep(5)
                    wait_time += 5
                if os.path.exists(lock_file_path):
                    sly.logger.log(
                        g._WARNING, f"Timeout waiting for directory lock. Proceeding anyway."
                    )

            # Create lock file to indicate this process is working with the directory
            with open(lock_file_path, "w") as f:
                f.write(f"Locked by process at {datetime.now(timezone.utc).isoformat()}")

            sly.fs.clean_dir(project_fs_dir)
        except Exception as e:
            sly.logger.log(g._WARNING, f"Error handling directory lock: {e}")
            # Continue even if lock failed to avoid blocking the entire process
    os.makedirs(project_fs_dir, exist_ok=True)

    if g.api.file.dir_exists(team.id, tf_project_dir):
        mandatory_class_stats = (
            dtools.OverviewPie,
            dtools.OverviewDonut,
            dtools.ClassBalance,
            dtools.ClassCooccurrence,
            dtools.ClassesPerImage,
            dtools.DatasetsAnnotations,
            dtools.ObjectsDistribution,
            dtools.ObjectSizes,
            dtools.ClassSizes,
            dtools.ClassesTreemap,
        )
        optional_tag_stats = (
            dtools.TagsImagesCooccurrence,
            dtools.TagsObjectsCooccurrence,
            dtools.ClassToTagCooccurrence,
            dtools.TagsImagesOneOfDistribution,
            dtools.TagsObjectsOneOfDistribution,
        )
        for stat in stats:
            path = f"{tf_project_dir}/{stat.basename_stem}.json"
            if isinstance(stat, mandatory_class_stats):
                if not g.api.file.exists(team.id, path):
                    force_stats_recalc = True
                    sly.logger.log(
                        g._WARNING,
                        f"The calcuated stat {stat.basename_stem!r} does not exist. Forcing full stats recalculation...",
                    )
            if isinstance(stat, optional_tag_stats):
                if g.api.file.exists(team.id, path) and u.applicability_test(stat) is False:
                    g.api.file.remove_file(team.id, path)
                    sly.logger.log(
                        g._INFO,
                        f"The applicability of tag stat {stat.basename_stem!r} has been changed. Deleting the old stat from team files.",
                    )

        if not g.api.file.exists(team.id, f"{tf_project_dir}/{heatmaps.basename_stem}.png"):
            force_stats_recalc = True
            sly.logger.log(
                g._WARNING,
                f"The calcuated stat {heatmaps.basename_stem!r} not exists. Forcing full stats recalculation...",
            )

    images_all_dct = u.get_project_images_all(datasets)
    updated_images, updated_classes, _cache, is_meta_changed = u.get_updated_images_and_classes(
        project, project_meta, datasets, images_all_dct, force_stats_recalc, _cache
    )
    total_updated = sum(len(lst) for lst in updated_images.values())
    if total_updated == 0 and not is_meta_changed:
        sly.logger.log(g._INFO, "Nothing to update. Skipping stats calculation...")

        # Clean up lock file before early exit to prevent deadlock
        lock_file_path = f"{project_fs_dir}/.processing.lock"
        sly.fs.silent_remove(lock_file_path)

        if isinstance(active_project_path_tf, str):
            g.api.file.remove(team.id, active_project_path_tf)
        u.add_heatmaps_status_ok(team, tf_project_dir, project_fs_dir)
        return JSONResponse({"message": "Nothing to update. Skipping stats calculation..."})

    if getattr(project, "items_count", None) is None:
        force_stats_recalc = True
        is_updated_images_count_valid = True
    else:
        is_updated_images_count_valid = total_updated < project.items_count

    if g.api.file.dir_exists(team.id, tf_project_dir) is True and is_updated_images_count_valid:
        force_stats_recalc = u.download_stats_chunks_to_buffer(
            team.id, project, tf_project_dir, project_fs_dir, force_stats_recalc
        )
        tf_status_path = f"{tf_project_dir}/_cache/heatmaps/status_ok"
        g.api.file.remove(team.id, tf_status_path)

    if force_stats_recalc is True:
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
        project_stats,
        project,
    )
    sly.logger.log(g._INFO, "Stats calculation finished.")
    u.remove_junk(team.id, tf_project_dir, project, datasets, project_fs_dir)
    u.sew_chunks_to_json(stats, project_fs_dir, updated_classes, is_meta_changed)

    sly.logger.log(g._INFO, "Start threading of 'calculate_and_save_heatmaps'")
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

    sly.logger.log(g._INFO, "Start threading of 'archive_chunks_and_upload'")
    thread2 = threading.Thread(
        target=u.archive_chunks_and_upload,
        args=(team, project, stats, tf_project_dir, project_fs_dir, datasets),
    )
    thread2.start()

    u.upload_sewed_stats(team.id, project_fs_dir, tf_project_dir)
    u.push_cache(team.id, project.id, tf_project_dir, project_fs_dir, _cache)

    # Clean up lock file after processing is complete
    lock_file_path = f"{project_fs_dir}/.processing.lock"
    sly.fs.silent_remove(lock_file_path)

    # sly.fs.silent_remove(active_project_path)
    if isinstance(active_project_path_tf, str):
        g.api.file.remove(team.id, active_project_path_tf)
    return JSONResponse(
        {"message": f"The statistics were updated: {total_updated} images were calculated"}
    )
