import importlib
import os
import tempfile
import threading
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("SERVER_ADDRESS", "http://localhost")
os.environ.setdefault("API_TOKEN", "test-token")
os.environ.setdefault(
    "SLY_APP_DATA_DIR",
    os.path.join(tempfile.gettempdir(), "quality-assurance-tests"),
)

main = importlib.import_module("src.main")


class HeatmapPlanTest(unittest.TestCase):
    def test_noop_rebuilds_missing_failed_heatmap(self):
        self.assertTrue(
            main._should_rebuild_heatmaps(
                stats_changed=False,
                heatmap_exists=False,
                heatmap_status="failed",
            )
        )

    def test_noop_rebuilds_failed_heatmap_even_if_old_png_exists(self):
        self.assertTrue(
            main._should_rebuild_heatmaps(
                stats_changed=False,
                heatmap_exists=True,
                heatmap_status="failed",
            )
        )

    def test_noop_does_not_duplicate_running_heatmap(self):
        self.assertFalse(
            main._should_rebuild_heatmaps(
                stats_changed=False,
                heatmap_exists=False,
                heatmap_status="running",
            )
        )

    def test_stats_update_always_rebuilds_heatmap(self):
        self.assertTrue(
            main._should_rebuild_heatmaps(
                stats_changed=True,
                heatmap_exists=True,
                heatmap_status="success",
            )
        )

    def test_old_stats_run_does_not_overwrite_newer_heatmap_status(self):
        ensure_owned = Mock(
            side_effect=main.u.ActiveRequestOwnershipError("ownership changed")
        )
        reporter = Mock()

        with patch.object(main, "HeatmapStatusReporter", return_value=reporter):
            main._report_stats_failure_if_owned(
                7, RuntimeError("stats failed"), ensure_owned
            )

        reporter.failed.assert_not_called()


class ActiveRequestCleanupTest(unittest.TestCase):
    def setUp(self):
        self.team = SimpleNamespace(id=2)
        self.workspace = SimpleNamespace(id=3)
        self.project = SimpleNamespace(id=1, name="project")
        self.lock = main._ActiveRequestLock(
            project_id=1,
            local_path=f"{main.g.ACTIVE_REQUESTS_DIR}/1",
            tf_path="/stats/_active_requests/1",
            content_hash="owned-hash",
        )

    @patch.object(main, "_release_active_project_request")
    @patch.object(main, "check_if_QA_tab_is_active", side_effect=RuntimeError("busy"))
    def test_does_not_release_lock_when_acquire_failed(self, _acquire, release):
        with self.assertRaisesRegex(RuntimeError, "busy"):
            main.main_func(None, self.team, self.workspace, self.project)

        release.assert_not_called()

    @patch.object(main, "_release_active_project_request")
    @patch.object(main, "TemporaryDirectory", side_effect=RuntimeError("workspace failed"))
    @patch.object(main, "check_if_QA_tab_is_active")
    def test_releases_owned_lock_when_processing_fails(
        self, acquire, _temporary_directory, release
    ):
        acquire.return_value = self.lock
        with self.assertRaisesRegex(RuntimeError, "workspace failed"):
            main.main_func(None, self.team, self.workspace, self.project)

        release.assert_called_once_with(2, self.lock)

    def test_fresh_lock_is_not_removed_in_development(self):
        file_api = SimpleNamespace(remove=Mock())
        now = datetime.now(timezone.utc)
        file = SimpleNamespace(
            path="/stats/_active_requests/1",
            updated_at=(now - timedelta(seconds=1)).isoformat(),
        )

        with patch.object(main.g, "api", SimpleNamespace(file=file_api)), patch.object(
            main.sly, "is_development", return_value=True
        ):
            main._remove_old_active_project_request(now, self.team, file)

        file_api.remove.assert_not_called()

    def test_lock_age_uses_total_seconds(self):
        file_api = SimpleNamespace(remove=Mock())
        now = datetime.now(timezone.utc)
        file = SimpleNamespace(
            path="/stats/_active_requests/1",
            updated_at=(now - timedelta(days=2, seconds=1)).isoformat(),
        )

        with patch.object(main.g, "api", SimpleNamespace(file=file_api)):
            main._remove_old_active_project_request(now, self.team, file)

        file_api.remove.assert_called_once_with(2, file.path)

    def test_heartbeat_refreshes_owned_lock_path(self):
        file_api = SimpleNamespace(
            get_info_by_path=Mock(return_value=SimpleNamespace(hash="owned-hash")),
            upload=Mock(return_value=SimpleNamespace(hash="owned-hash")),
        )
        heartbeat = main._ActiveRequestHeartbeat(self.team.id, self.lock)

        with patch.object(main.g, "api", SimpleNamespace(file=file_api)):
            heartbeat._refresh_lock()

        self.assertEqual(file_api.get_info_by_path.call_count, 2)
        file_api.upload.assert_called_once_with(
            2,
            self.lock.local_path,
            self.lock.tf_path,
        )

    def test_release_does_not_remove_foreign_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, "lock")
            with open(local_path, "w", encoding="utf-8") as lock_file:
                lock_file.write("owned")
            lock = main._ActiveRequestLock(
                project_id=1,
                local_path=local_path,
                tf_path=self.lock.tf_path,
                content_hash="owned-hash",
            )
            file_api = SimpleNamespace(
                get_info_by_path=Mock(return_value=SimpleNamespace(hash="foreign-hash")),
                remove=Mock(),
            )

            with patch.object(main.g, "api", SimpleNamespace(file=file_api)):
                main._release_active_project_request(self.team.id, lock)

            file_api.remove.assert_not_called()
            self.assertFalse(os.path.exists(local_path))

    def test_stop_waits_for_inflight_heartbeat_upload(self):
        upload_started = threading.Event()
        allow_upload_to_finish = threading.Event()

        def upload(*_args):
            upload_started.set()
            allow_upload_to_finish.wait()
            return SimpleNamespace(hash="owned-hash")

        file_api = SimpleNamespace(
            get_info_by_path=Mock(return_value=SimpleNamespace(hash="owned-hash")),
            upload=Mock(side_effect=upload),
        )
        heartbeat = main._ActiveRequestHeartbeat(self.team.id, self.lock)

        with patch.object(main.g, "api", SimpleNamespace(file=file_api)), patch.object(
            main, "LOCK_HEARTBEAT_INTERVAL", 0.01
        ):
            heartbeat.start()
            self.assertTrue(upload_started.wait(timeout=1))
            stopper = threading.Thread(target=heartbeat.stop)
            stopper.start()
            stopper.join(timeout=0.05)
            self.assertTrue(stopper.is_alive())
            allow_upload_to_finish.set()
            stopper.join(timeout=1)

        self.assertFalse(stopper.is_alive())
        self.assertEqual(file_api.upload.call_count, 1)


class HeatmapStateTest(unittest.TestCase):
    def test_skipped_status_removes_stale_png(self):
        file_api = SimpleNamespace(
            dir_exists=Mock(return_value=True),
            exists=Mock(return_value=True),
            remove=Mock(),
        )
        heatmaps = SimpleNamespace(basename_stem="classes_heatmaps")

        with patch.object(main.g, "api", SimpleNamespace(file=file_api)), patch.object(
            main.dtools,
            "heatmap_status_endpoint",
            return_value={"status": "skipped"},
        ):
            exists, status = main._get_heatmap_state(
                2, 1, "/stats/1_project", heatmaps
            )

        self.assertFalse(exists)
        self.assertEqual(status, "skipped")
        file_api.remove.assert_called_once_with(
            2, "/stats/1_project/classes_heatmaps.png"
        )


class StatsPublishingTest(unittest.TestCase):
    def test_upload_failure_prevents_archive_and_cache_commit(self):
        team = SimpleNamespace(id=2)
        project = SimpleNamespace(id=1)
        ensure_owned = Mock()

        with patch.object(
            main.u,
            "upload_sewed_stats",
            side_effect=RuntimeError("upload failed"),
        ), patch.object(main.u, "archive_chunks_and_upload") as archive, patch.object(
            main.u, "push_cache"
        ) as push_cache:
            with self.assertRaisesRegex(RuntimeError, "upload failed"):
                main._publish_stats(
                    team,
                    project,
                    [],
                    "/stats/project",
                    "/tmp/project",
                    [],
                    {},
                    main.u.StatsRunState(),
                    ensure_owned,
                )

        ensure_owned.assert_called_once_with()
        archive.assert_not_called()
        push_cache.assert_not_called()


class MainNoopTest(unittest.TestCase):
    def _run_noop(self, heatmap_state):
        team = SimpleNamespace(id=2)
        workspace = SimpleNamespace(id=3)
        project = SimpleNamespace(
            id=1,
            name="project",
            items_count=10,
            datasets_count=0,
        )
        lock = main._ActiveRequestLock(
            project_id=1,
            local_path=f"{main.g.ACTIVE_REQUESTS_DIR}/1",
            tf_path="/stats/_active_requests/1",
            content_hash="owned-hash",
        )
        file_api = SimpleNamespace(dir_exists=Mock(return_value=False))
        api = SimpleNamespace(
            project=SimpleNamespace(
                get_meta=Mock(return_value={}),
                get_stats=Mock(return_value={}),
            ),
            dataset=SimpleNamespace(get_list=Mock(return_value=[])),
            file=file_api,
        )
        heartbeat = Mock()
        download_chunks = Mock()
        collect_heatmap_sample = Mock(return_value=({10: {1}}, {20: {2}}))
        start_heatmap_thread = Mock()

        with ExitStack() as stack:
            stack.enter_context(patch.object(main.g, "api", api))
            stack.enter_context(patch.object(main.g, "initialize_log_levels"))
            stack.enter_context(
                patch.object(main, "check_if_QA_tab_is_active", return_value=lock)
            )
            stack.enter_context(
                patch.object(main, "_ActiveRequestHeartbeat", return_value=heartbeat)
            )
            stack.enter_context(patch.object(main, "_release_active_project_request"))
            stack.enter_context(patch.object(main.sly.ProjectMeta, "from_json", return_value=Mock()))
            stack.enter_context(patch.object(main, "_build_stats", return_value=[]))
            stack.enter_context(
                patch.object(
                    main.dtools,
                    "ClassesHeatmaps",
                    return_value=SimpleNamespace(basename_stem="classes_heatmaps"),
                )
            )
            stack.enter_context(
                patch.object(main.u, "pull_cache", return_value=(False, {}))
            )
            stack.enter_context(
                patch.object(main.u, "get_project_images_all", return_value={})
            )
            stack.enter_context(
                patch.object(
                    main.u,
                    "get_updated_images_and_classes",
                    return_value=({}, set(), {}, False),
                )
            )
            stack.enter_context(
                patch.object(main, "_get_heatmap_state", return_value=heatmap_state)
            )
            stack.enter_context(
                patch.object(
                    main.u,
                    "collect_heatmap_sample",
                    collect_heatmap_sample,
                )
            )
            stack.enter_context(
                patch.object(main, "_start_heatmap_thread", start_heatmap_thread)
            )
            stack.enter_context(
                patch.object(
                    main.u,
                    "download_stats_chunks_to_buffer",
                    download_chunks,
                )
            )

            response = main.main_func(None, team, workspace, project)

        return response, download_chunks, collect_heatmap_sample, start_heatmap_thread

    def test_unchanged_project_does_not_download_chunks_archive(self):
        response, download_chunks, collect_heatmap_sample, start_heatmap_thread = (
            self._run_noop((True, "success"))
        )

        self.assertIn(b"Nothing to update", response.body)
        download_chunks.assert_not_called()
        collect_heatmap_sample.assert_not_called()
        start_heatmap_thread.assert_not_called()

    def test_unchanged_project_rebuilds_missing_failed_heatmap(self):
        response, download_chunks, collect_heatmap_sample, start_heatmap_thread = (
            self._run_noop((False, "failed"))
        )

        self.assertIn(b"Nothing to update", response.body)
        download_chunks.assert_not_called()
        collect_heatmap_sample.assert_called_once()
        start_heatmap_thread.assert_called_once()


if __name__ == "__main__":
    unittest.main()
