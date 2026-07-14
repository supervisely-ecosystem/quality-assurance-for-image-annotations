import os
import src.globals as g
import src.utils as u
import supervisely as sly
from supervisely import ProjectInfo, TeamInfo, WorkspaceInfo
import dataset_tools as dtools
from dataset_tools.repo.heatmap_status import HeatmapStatusReporter
from datetime import datetime, timezone
import time
import threading
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from uuid import uuid4
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from supervisely.app.widgets import Container
from src.ui.input import card_1
import fcntl


layout = Container(widgets=[card_1], direction="vertical")
app = sly.Application(layout=layout)
server = app.get_server()


TIMELOCK_LIMIT = 180  # seconds
LOCK_HEARTBEAT_INTERVAL = 30  # seconds


@dataclass
class _ActiveRequestLock:
    project_id: int
    local_path: str
    tf_path: str
    content_hash: str


class _ActiveRequestHeartbeat:
    def __init__(self, team_id: int, lock: _ActiveRequestLock):
        self._team_id = team_id
        self._lock = lock
        self._stop_event = threading.Event()
        self._started = False
        self._ownership_error = None
        self._state_lock = threading.RLock()
        self._thread = threading.Thread(
            target=self._run,
            name=f"stats-lock-heartbeat-{lock.project_id}",
            daemon=True,
        )

    def start(self):
        self._thread.start()
        self._started = True

    def stop(self):
        self._stop_event.set()
        if self._started:
            self._thread.join()

    def _refresh_lock(self):
        with self._state_lock:
            self._ensure_owned_unlocked()
            with open(self._lock.local_path, "w", encoding="utf-8") as lock_file:
                lock_file.write(uuid4().hex)
            uploaded = g.api.file.upload(
                self._team_id,
                self._lock.local_path,
                self._lock.tf_path,
            )
            self._lock.content_hash = uploaded.hash
            self._ensure_owned_unlocked()

    def ensure_owned(self):
        with self._state_lock:
            self._ensure_owned_unlocked()

    def _ensure_owned_unlocked(self):
        if self._ownership_error is not None:
            raise self._ownership_error
        current = g.api.file.get_info_by_path(self._team_id, self._lock.tf_path)
        if current is None or current.hash != self._lock.content_hash:
            raise u.ActiveRequestOwnershipError(
                f"Active request lock {self._lock.tf_path!r} is no longer owned by this run"
            )

    def _run(self):
        while not self._stop_event.wait(LOCK_HEARTBEAT_INTERVAL):
            try:
                self._refresh_lock()
            except u.ActiveRequestOwnershipError as e:
                self._ownership_error = e
                self._stop_event.set()
                sly.logger.error(str(e))
                return
            except Exception as e:
                sly.logger.warning(
                    f"Failed to refresh active request lock {self._lock.tf_path!r}: {e}"
                )


class ActiveRequestInProgressError(RuntimeError):
    pass


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

        raise HTTPException(
            status_code=500,
            detail={
                "title": "The app has got the following error:",
                "message": msg,
            },
        ) from e

    return result


@server.get("/heatmaps/status")
def heatmaps_status_endpoint(project_id: int):
    try:
        return dtools.heatmap_status_endpoint(g.api, project_id)
    except Exception as e:
        msg = e.__class__.__name__ + ": " + str(e)
        sly.logger.error(msg)
        raise HTTPException(
            status_code=500,
            detail={
                "title": "The app has got the following error:",
                "message": msg,
            },
        ) from e


def _remove_old_active_project_request(now, team, file):
    updated_at = file.updated_at
    if isinstance(updated_at, str) and updated_at.endswith("Z"):
        updated_at = updated_at[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(updated_at)
    except (TypeError, ValueError):
        sly.logger.warning(
            f"Cannot parse updated_at for active request {file.path!r}; keeping the lock."
        )
        return
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if (now - dt).total_seconds() > TIMELOCK_LIMIT:
        g.api.file.remove(team.id, file.path)
        sly.logger.debug(
            f"The temporary file {file.path!r} has been removed from tf because of "
            f"time limit ({TIMELOCK_LIMIT} secs). TEAM_ID={team.id}"
        )


def _heatmap_generation_is_running(project_id: int) -> bool:
    try:
        status = dtools.heatmap_status_endpoint(g.api, project_id)
        return status.get("status") == "running"
    except Exception as e:
        sly.logger.warning(f"Failed to check heatmap status for project {project_id}: {e}")
        return False


def check_if_QA_tab_is_active(team: TeamInfo, project: ProjectInfo) -> _ActiveRequestLock:
    """
    Checks if the QA tab is active for the project and waits in queue if busy.
    Uses Team Files as the primary lock mechanism to prevent race conditions across multiple app instances.
    Local file lock is used for optimization within the same app instance.

    The locking strategy with queue:
    1. Acquire local file lock FIRST (atomic within instance)
    2. Check Team Files for existing lock (global check across all app instances)
    3. If locked by another process:
       - Release local lock
       - Wait in queue until lock is released
       - Try again from step 1
    4. If not locked, create lock file in Team Files
    5. Triple-check Team Files after upload to ensure our lock was successful

    Args:
        team: TeamInfo object containing team information
        project: ProjectInfo object containing project information

    Returns:
        _ActiveRequestLock: Owned lock metadata for heartbeat and safe release
    """
    sly.logger.log(g._INFO, "Checking requests...")

    active_project_path_local = f"{g.ACTIVE_REQUESTS_DIR}/{project.id}"
    active_project_path_tf = f"{g.TF_ACTIVE_REQUESTS_DIR}/{project.id}"
    lock_file_path = f"{g.ACTIVE_REQUESTS_DIR}/{project.id}.lock"

    # Create directories if they don't exist
    os.makedirs(g.ACTIVE_REQUESTS_DIR, exist_ok=True)

    max_wait_time = 600  # 10 minutes maximum wait time
    start_time = time.time()
    attempt = 0

    while True:
        attempt += 1

        # Check if we've exceeded maximum wait time
        if time.time() - start_time > max_wait_time:
            msg = f"Timeout waiting for project {project.id} to become available after {max_wait_time} seconds"
            sly.logger.error(msg)
            raise Exception(msg)

        # Acquire local file lock FIRST - this ensures only one thread per app instance can proceed
        try:
            with open(lock_file_path, "w") as lock_file:
                # Try to acquire exclusive lock (non-blocking) - prevents race within same app instance
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except IOError:
                    if _heatmap_generation_is_running(project.id):
                        raise ActiveRequestInProgressError(
                            f"Heatmap generation for project {project.id} is already in progress"
                        )
                    # Lock is already held by another request in this app instance
                    # Wait and retry
                    if attempt == 1:
                        sly.logger.log(
                            g._INFO,
                            f"Request for project ID={project.id} is in queue (this app instance is busy). Waiting...",
                        )
                    time.sleep(5)
                    continue

                # Now that we have local lock, check Team Files (global lock across all app instances)
                file = g.api.file.get_info_by_path(team.id, active_project_path_tf)
                if file is not None:
                    now = datetime.now(timezone.utc)
                    _remove_old_active_project_request(now, team, file)

                    # Check if file still exists after cleanup
                    if g.api.file.exists(team.id, file.path) is True:
                        if _heatmap_generation_is_running(project.id):
                            raise ActiveRequestInProgressError(
                                f"Heatmap generation for project {project.id} is already in progress"
                            )
                        # Another app instance is processing - release local lock and wait
                        if attempt == 1:
                            sly.logger.log(
                                g._INFO,
                                f"Request for project ID={project.id} is in queue (app instance is busy). Waiting...",
                            )
                        # Lock will be released when exiting 'with' block
                        time.sleep(5)
                        continue

                # No locks exist - we can proceed!
                # Create the lock file locally
                with open(active_project_path_local, "w") as request_file:
                    request_file.write(uuid4().hex)

                # Upload to Team Files - this becomes the global lock
                try:
                    uploaded = g.api.file.upload(
                        team.id, active_project_path_local, active_project_path_tf
                    )
                    sly.logger.log(g._INFO, f"Created active request file for project {project.id}")
                except Exception as e:
                    sly.logger.warning(f"Failed to upload active request file: {e}")
                    # Clean up local file if upload failed
                    sly.fs.silent_remove(active_project_path_local)
                    raise

                # Triple-check: verify our lock file was successfully created
                file = g.api.file.get_info_by_path(team.id, active_project_path_tf)
                if file is None or file.hash != uploaded.hash:
                    msg = (
                        f"Failed to retain ownership of lock for project {project.id}. "
                        "Retrying..."
                    )
                    sly.logger.warning(msg)
                    sly.fs.silent_remove(active_project_path_local)
                    time.sleep(2)
                    continue

                # Success! We have the lock
                if attempt > 1:
                    wait_time = int(time.time() - start_time)
                    sly.logger.log(
                        g._INFO,
                        f"Project {project.id} lock acquired after waiting {wait_time} seconds",
                    )

                sly.logger.log(g._INFO, "Finish checking if 'QA & Stats' tab is active.")
                return _ActiveRequestLock(
                    project_id=project.id,
                    local_path=active_project_path_local,
                    tf_path=active_project_path_tf,
                    content_hash=uploaded.hash,
                )

        except ActiveRequestInProgressError:
            raise
        except Exception as e:
            sly.logger.error(f"Error in check_if_QA_tab_is_active: {e}")
            raise


def _release_active_project_request(team_id: int, lock: _ActiveRequestLock):
    try:
        current = g.api.file.get_info_by_path(team_id, lock.tf_path)
        if current is None:
            return
        if current.hash != lock.content_hash:
            sly.logger.warning(
                f"Not removing active request lock {lock.tf_path!r}: ownership changed."
            )
            return
        g.api.file.remove(team_id, lock.tf_path)
    finally:
        sly.fs.silent_remove(lock.local_path)


def _get_heatmap_state(team_id: int, project_id: int, tf_project_dir: str, heatmaps):
    if not g.api.file.dir_exists(team_id, tf_project_dir):
        return False, "missing"

    heatmaps_path = f"{tf_project_dir}/{heatmaps.basename_stem}.png"
    status = dtools.heatmap_status_endpoint(g.api, project_id)
    heatmap_exists = g.api.file.exists(team_id, heatmaps_path)
    if status["status"] == "skipped" and heatmap_exists:
        g.api.file.remove(team_id, heatmaps_path)
        heatmap_exists = False
    return heatmap_exists, status["status"]


def _should_rebuild_heatmaps(stats_changed: bool, heatmap_exists: bool, heatmap_status: str):
    if stats_changed:
        return True
    if heatmap_status in {"running", "skipped"}:
        return False
    if heatmap_status in {"failed", "stale", "unknown", "missing"}:
        return True
    return not heatmap_exists


def _report_stats_failure_if_owned(project_id: int, error: Exception, ensure_owned):
    try:
        ensure_owned()
        HeatmapStatusReporter(g.api, project_id, logger=sly.logger).failed(
            error,
            stage="stats",
            message="Statistics calculation failed before heatmap generation.",
        )
    except u.ActiveRequestOwnershipError as ownership_error:
        sly.logger.warning(
            "Statistics failure status was not published after lock ownership changed: "
            f"{ownership_error}"
        )
    except Exception as status_error:
        sly.logger.warning(f"Failed to update heatmaps status: {status_error}")


def _start_heatmap_thread(
    team,
    tf_project_dir,
    project_id,
    heatmaps,
    images_all_dct,
    project_stats,
    project,
    before_publish=None,
    on_complete=None,
):
    if before_publish is not None:
        before_publish()
    HeatmapStatusReporter(g.api, project_id, logger=sly.logger).running(
        "queued",
        "Statistics are ready. Heatmap generation is queued.",
        progress=0,
    )

    def run_heatmaps():
        try:
            try:
                if before_publish is not None:
                    before_publish()
                HeatmapStatusReporter(g.api, project_id, logger=sly.logger).running(
                    "collecting_sample",
                    "Collecting a project-wide heatmap sample.",
                    progress=0.05,
                )
                heatmaps_image_ids, heatmaps_figure_ids = u.collect_heatmap_sample(
                    images_all_dct, project_stats, project
                )
            except u.ActiveRequestOwnershipError as e:
                sly.logger.warning(
                    f"Heatmap sample collection cancelled after lock ownership changed: {e}"
                )
                return
            except Exception as e:
                sly.logger.error(f"Error collecting heatmap sample: {e!r}")
                try:
                    if before_publish is not None:
                        before_publish()
                except Exception as ownership_error:
                    sly.logger.warning(
                        "Heatmap sampling failure status was not published because lock "
                        f"ownership could not be verified: {ownership_error}"
                    )
                    return
                HeatmapStatusReporter(g.api, project_id, logger=sly.logger).failed(
                    e,
                    stage="collecting_sample",
                    message="Failed to collect the project-wide heatmap sample.",
                )
                return

            u.calculate_and_upload_heatmaps(
                team,
                tf_project_dir,
                g.STORAGE_DIR,
                project_id,
                heatmaps,
                heatmaps_image_ids,
                heatmaps_figure_ids,
                before_publish,
            )
        finally:
            if on_complete is not None:
                on_complete()

    thread = threading.Thread(target=run_heatmaps, name=f"heatmaps-{project_id}")
    thread.start()
    return thread


def _publish_stats(
    team,
    project,
    stats,
    tf_project_dir,
    project_fs_dir,
    datasets,
    cache,
    run_state,
    ensure_owned,
):
    ensure_owned()
    u.upload_sewed_stats(team.id, project_fs_dir, tf_project_dir)
    ensure_owned()
    u.archive_chunks_and_upload(
        team,
        project,
        stats,
        tf_project_dir,
        project_fs_dir,
        datasets,
        run_state,
    )
    ensure_owned()
    u.push_cache(
        team.id,
        project.id,
        tf_project_dir,
        project_fs_dir,
        cache,
        run_state,
    )


def _build_stats(project_meta, project_stats, datasets):
    return [
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


def main_func(user_id: int, team: TeamInfo, workspace: WorkspaceInfo, project: ProjectInfo):
    g.initialize_log_levels(project.id)
    active_request_lock = None
    lock_heartbeat = None
    lock_release_transferred = False
    lock_released = False
    lock_release_guard = threading.Lock()
    report_heatmap_failure = False

    def release_active_request():
        nonlocal lock_released
        with lock_release_guard:
            if lock_released:
                return
            lock_released = True

        if lock_heartbeat is not None:
            lock_heartbeat.stop()
        try:
            _release_active_project_request(team.id, active_request_lock)
        except Exception as cleanup_error:
            sly.logger.warning(
                f"Failed to release active project request: {cleanup_error}"
            )

    try:
        try:
            active_request_lock = check_if_QA_tab_is_active(team, project)
        except ActiveRequestInProgressError:
            return JSONResponse(
                {"message": "Heatmap generation is already in progress."},
                status_code=202,
            )
        lock_heartbeat = _ActiveRequestHeartbeat(team.id, active_request_lock)
        lock_heartbeat.start()
        sly.logger.log(g._INFO, "Start Quality Assurance.")

        tf_project_dir = f"{g.TF_STATS_DIR}/{project.id}_{project.name}"
        with TemporaryDirectory(
            prefix=f"stats-{project.id}-", dir=g.STORAGE_DIR
        ) as project_fs_dir:
            run_state = u.StatsRunState()
            force_stats_recalc, _cache = u.pull_cache(
                team.id,
                project.id,
                tf_project_dir,
                project_fs_dir,
                run_state,
            )

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

            stats = _build_stats(project_meta, project_stats, datasets)
            heatmaps = dtools.ClassesHeatmaps(project_meta, project_stats)

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
                    if isinstance(stat, mandatory_class_stats) and not g.api.file.exists(
                        team.id, path
                    ):
                        force_stats_recalc = True
                        sly.logger.log(
                            g._WARNING,
                            f"The calculated stat {stat.basename_stem!r} does not exist. "
                            "Forcing full stats recalculation...",
                        )
                    if (
                        isinstance(stat, optional_tag_stats)
                        and g.api.file.exists(team.id, path)
                        and u.applicability_test(stat) is False
                    ):
                        g.api.file.remove_file(team.id, path)
                        sly.logger.log(
                            g._INFO,
                            f"The applicability of tag stat {stat.basename_stem!r} has been changed. "
                            "Deleting the old stat from team files.",
                        )

            images_all_dct = u.get_project_images_all(datasets)
            cached_image_ids = set(_cache.get("images", {}))
            updated_images, changed_object_class_ids, _cache, is_meta_changed = (
                u.get_updated_images_and_classes(
                    project,
                    project_meta,
                    datasets,
                    images_all_dct,
                    force_stats_recalc,
                    _cache,
                )
            )
            total_updated = sum(len(lst) for lst in updated_images.values())
            changed_image_ids = {
                image.id for images in updated_images.values() for image in images
            }
            current_image_ids = {
                image.id for images in images_all_dct.values() for image in images
            }
            image_set_changed = cached_image_ids != current_image_ids
            figure_signatures = _cache.setdefault("figure_signatures", {})
            for removed_image_id in set(figure_signatures) - current_image_ids:
                del figure_signatures[removed_image_id]

            if getattr(project, "items_count", None) is None:
                force_stats_recalc = True
                is_updated_images_count_valid = True
            else:
                is_updated_images_count_valid = total_updated < project.items_count

            stats_changed = force_stats_recalc or total_updated > 0 or is_meta_changed
            # Image timestamps also change for tag-only updates. Figure signatures below
            # decide whether image changes actually invalidate the heatmap.
            heatmap_inputs_changed = (
                force_stats_recalc
                or image_set_changed
                or len(changed_object_class_ids) > 0
            )
            heatmap_exists, heatmap_status = _get_heatmap_state(
                team.id, project.id, tf_project_dir, heatmaps
            )
            rebuild_heatmaps = _should_rebuild_heatmaps(
                heatmap_inputs_changed, heatmap_exists, heatmap_status
            )
            report_heatmap_failure = rebuild_heatmaps

            if not stats_changed:
                if rebuild_heatmaps:
                    _start_heatmap_thread(
                        team,
                        tf_project_dir,
                        project.id,
                        heatmaps,
                        images_all_dct,
                        project_stats,
                        project,
                        before_publish=lock_heartbeat.ensure_owned,
                        on_complete=release_active_request,
                    )
                    lock_release_transferred = True
                    report_heatmap_failure = False
                elif heatmap_status == "running":
                    sly.logger.log(
                        g._INFO,
                        "Heatmaps are still running. Frontend will check availability.",
                    )

                sly.logger.log(g._INFO, "Nothing to update. Skipping stats calculation...")
                return JSONResponse(
                    {"message": "Nothing to update. Skipping stats calculation..."}
                )

            if g.api.file.dir_exists(team.id, tf_project_dir) and is_updated_images_count_valid:
                force_stats_recalc = u.download_stats_chunks_to_buffer(
                    team.id,
                    project,
                    tf_project_dir,
                    project_fs_dir,
                    force_stats_recalc,
                    run_state,
                    stats,
                )

            idx_to_infos, infos_to_idx = u.get_indexes_dct(
                project.id, datasets, images_all_dct
            )
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

            total_updated = sum(len(lst) for lst in updated_images.values())
            is_full_stats_recalc = force_stats_recalc or (
                getattr(project, "items_count", None) is not None
                and total_updated == project.items_count
            )
            if not is_full_stats_recalc:
                updated_images, _, force_full_from_meta = (
                    u.add_changed_class_chunks_to_updated_images(
                        updated_images,
                        changed_object_class_ids,
                        project_fs_dir,
                        idx_to_infos,
                        infos_to_idx,
                        images_all_dct,
                    )
                )
                if force_full_from_meta:
                    force_stats_recalc = True

            total_updated = sum(len(lst) for lst in updated_images.values())
            is_final_full_recalc = force_stats_recalc or (
                getattr(project, "items_count", None) is not None
                and total_updated == project.items_count
            )
            if is_final_full_recalc:
                run_state.chunks_latest_datetime = None

            tf_all_paths = []
            if g.api.file.dir_exists(team.id, tf_project_dir):
                tf_all_paths = [
                    info.path
                    for info in g.api.file.list2(team.id, tf_project_dir, recursive=True)
                ]
            figures_changed = u.calculate_stats_and_save_chunks(
                updated_images,
                stats,
                tf_all_paths,
                project_fs_dir,
                idx_to_infos,
                infos_to_idx,
                run_state,
                figure_signatures,
                changed_image_ids,
            )
            rebuild_heatmaps = rebuild_heatmaps or figures_changed
            sly.logger.log(g._INFO, "Stats calculation finished.")
            lock_heartbeat.ensure_owned()
            u.remove_junk(
                team.id,
                tf_project_dir,
                project,
                datasets,
                project_fs_dir,
                run_state,
            )
            u.sew_chunks_to_json(
                stats, project_fs_dir, changed_object_class_ids, is_meta_changed
            )
            _publish_stats(
                team,
                project,
                stats,
                tf_project_dir,
                project_fs_dir,
                datasets,
                _cache,
                run_state,
                lock_heartbeat.ensure_owned,
            )

            if rebuild_heatmaps:
                _start_heatmap_thread(
                    team,
                    tf_project_dir,
                    project.id,
                    heatmaps,
                    images_all_dct,
                    project_stats,
                    project,
                    before_publish=lock_heartbeat.ensure_owned,
                    on_complete=release_active_request,
                )
                lock_release_transferred = True
            report_heatmap_failure = False

            if rebuild_heatmaps:
                message = (
                    "Stats calculation completed. Heatmaps are being calculated in background."
                )
            else:
                message = "Stats calculation completed. Existing heatmaps remain valid."
            sly.logger.log(g._INFO, message)
            return JSONResponse(
                {
                    "message": f"The statistics were updated: {total_updated} images were calculated"
                }
            )
    except Exception as e:
        if (
            not isinstance(e, u.ActiveRequestOwnershipError)
            and active_request_lock is not None
            and report_heatmap_failure
        ):
            _report_stats_failure_if_owned(
                project.id, e, lock_heartbeat.ensure_owned
            )
        raise
    finally:
        if active_request_lock is not None and not lock_release_transferred:
            release_active_request()
