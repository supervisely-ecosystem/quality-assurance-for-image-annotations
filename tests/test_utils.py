import importlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("SERVER_ADDRESS", "http://localhost")
os.environ.setdefault("API_TOKEN", "test-token")
os.environ.setdefault(
    "SLY_APP_DATA_DIR",
    os.path.join(tempfile.gettempdir(), "quality-assurance-tests"),
)

utils = importlib.import_module("src.utils")


class CacheFileApi:
    def __init__(self, payload):
        self.payload = payload

    def dir_exists(self, team_id, path):
        return True

    def exists(self, team_id, path):
        return True

    def download(self, team_id, src_path, dst_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if isinstance(self.payload, str):
            Path(dst_path).write_text(self.payload, encoding="utf-8")
        else:
            Path(dst_path).write_text(json.dumps(self.payload), encoding="utf-8")


class ArchiveFileApi:
    def __init__(self, archive_path, remote_path):
        self.archive_path = archive_path
        self.remote_path = remote_path

    def get_info_by_path(self, team_id, path):
        return SimpleNamespace(path=self.remote_path, sizeb=os.path.getsize(self.archive_path))

    def download(self, team_id, src_path, dst_path, progress_cb=None):
        shutil.copyfile(self.archive_path, dst_path)


class DummyStat:
    basename_stem = "class_balance"

    def to_numpy_raw(self):
        return np.array([1])


class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.log_levels = (
            patch.object(utils.g, "_INFO", 20, create=True),
            patch.object(utils.g, "_WARNING", 30, create=True),
            patch.object(utils.sly.logger, "log"),
        )
        for patcher in self.log_levels:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.log_levels):
            patcher.stop()

    @staticmethod
    def _valid_cache(chunks_dt):
        return {
            "images": {"1": "2026-01-01T00:00:00Z"},
            "meta": {"classes": [], "tags": []},
            "stats_meta": {
                "chunk_size": utils.g.CHUNK_SIZE,
                "chunks_dt": chunks_dt,
                "dataset-tools": "0.2.3",
            },
        }

    def test_pull_cache_keeps_chunk_state_isolated_per_run(self):
        state_a = utils.StatsRunState()
        state_b = utils.StatsRunState()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                utils.g,
                "api",
                SimpleNamespace(file=CacheFileApi(self._valid_cache("2026-01-01T00:00:00Z"))),
            ):
                force_a, _ = utils.pull_cache(1, 10, "/stats/a", f"{temp_dir}/a", state_a)

            with patch.object(
                utils.g,
                "api",
                SimpleNamespace(file=CacheFileApi(self._valid_cache("2026-02-02T00:00:00Z"))),
            ):
                force_b, _ = utils.pull_cache(1, 20, "/stats/b", f"{temp_dir}/b", state_b)

        self.assertFalse(force_a)
        self.assertFalse(force_b)
        self.assertEqual(state_a.chunks_latest_datetime, datetime(2026, 1, 1))
        self.assertEqual(state_b.chunks_latest_datetime, datetime(2026, 2, 2))

    def test_pull_cache_treats_corruption_as_cache_miss(self):
        cases = (
            ("invalid JSON", "{"),
            (
                "missing stats_meta",
                {
                    "images": {"1": "2026-01-01T00:00:00Z"},
                    "meta": {"classes": [], "tags": []},
                },
            ),
            (
                "invalid project meta",
                {
                    **self._valid_cache("2026-01-01T00:00:00Z"),
                    "meta": {},
                },
            ),
            ("invalid chunks datetime", self._valid_cache("not-a-date")),
            ("chunks datetime without UTC marker", self._valid_cache("2026-01-01T00:00:00")),
            (
                "invalid dataset-tools version",
                {
                    **self._valid_cache("2026-01-01T00:00:00Z"),
                    "stats_meta": {
                        "chunk_size": utils.g.CHUNK_SIZE,
                        "chunks_dt": "2026-01-01T00:00:00Z",
                        "dataset-tools": "not a version",
                    },
                },
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            for index, (name, payload) in enumerate(cases):
                with self.subTest(name=name):
                    state = utils.StatsRunState(datetime(2000, 1, 1))
                    with patch.object(
                        utils.g,
                        "api",
                        SimpleNamespace(file=CacheFileApi(payload)),
                    ):
                        force, cache = utils.pull_cache(
                            1,
                            index,
                            f"/stats/{index}",
                            f"{temp_dir}/{index}",
                            state,
                        )
                    self.assertTrue(force)
                    self.assertEqual(cache, {})
                    self.assertIsNone(state.chunks_latest_datetime)

    def test_upload_sewed_stats_propagates_upload_error(self):
        file_api = SimpleNamespace(upload_bulk=Mock(side_effect=RuntimeError("upload failed")))
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "class_balance.json").write_text("{}", encoding="utf-8")
            with patch.object(utils.g, "api", SimpleNamespace(file=file_api)):
                with self.assertRaisesRegex(RuntimeError, "upload failed"):
                    utils.upload_sewed_stats(1, temp_dir, "/stats/project")

    def test_download_chunks_rejects_tar_path_traversal(self):
        state = utils.StatsRunState(datetime(2026, 1, 1))
        project = SimpleNamespace(id=7, name="project")
        archive_name = "7_project_chunks_2026-01-01T00:00:00.tar.gz"
        remote_path = f"/stats/7_project/{archive_name}"

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, "malicious.tar.gz")
            payload = b"outside"
            with tarfile.open(archive_path, "w:gz") as tar:
                member = tarfile.TarInfo("../sentinel.txt")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))

            project_fs_dir = os.path.join(temp_dir, "buffer", "7_project")
            os.makedirs(project_fs_dir)
            file_api = ArchiveFileApi(archive_path, remote_path)
            with patch.object(utils.g, "api", SimpleNamespace(file=file_api)):
                force = utils.download_stats_chunks_to_buffer(
                    1,
                    project,
                    "/stats/7_project",
                    project_fs_dir,
                    False,
                    state,
                    [DummyStat()],
                )

            self.assertTrue(force)
            self.assertFalse(Path(project_fs_dir).parent.joinpath("sentinel.txt").exists())

    def test_download_chunks_rejects_malformed_chunk_filename(self):
        state = utils.StatsRunState(datetime(2026, 1, 1))
        project = SimpleNamespace(id=7, name="project")
        archive_name = "7_project_chunks_2026-01-01T00:00:00.tar.gz"
        remote_path = f"/stats/7_project/{archive_name}"

        with tempfile.TemporaryDirectory() as temp_dir:
            malformed_path = Path(temp_dir, "bad.npy")
            np.save(malformed_path, [1])
            archive_path = os.path.join(temp_dir, "malformed.tar.gz")
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(malformed_path, arcname="class_balance/bad.npy")

            project_fs_dir = os.path.join(temp_dir, "buffer", "7_project")
            os.makedirs(project_fs_dir)
            file_api = ArchiveFileApi(archive_path, remote_path)
            with patch.object(utils.g, "api", SimpleNamespace(file=file_api)):
                force = utils.download_stats_chunks_to_buffer(
                    1,
                    project,
                    "/stats/7_project",
                    project_fs_dir,
                    False,
                    state,
                    [DummyStat()],
                )

            self.assertTrue(force)
            self.assertFalse(Path(project_fs_dir, "class_balance", "bad.npy").exists())

    def test_download_chunks_extracts_valid_archive(self):
        state = utils.StatsRunState(datetime(2026, 1, 1))
        project = SimpleNamespace(id=7, name="project")
        archive_name = "7_project_chunks_2026-01-01T00:00:00.tar.gz"
        remote_path = f"/stats/7_project/{archive_name}"

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir, "source", "class_balance")
            source_dir.mkdir(parents=True)
            np.save(source_dir / "chunk_0_1_7_1000_2026-01-01T00:00:00.npy", [1])
            archive_path = os.path.join(temp_dir, "valid.tar.gz")
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(source_dir, arcname="class_balance")

            project_fs_dir = os.path.join(temp_dir, "buffer", "7_project")
            os.makedirs(project_fs_dir)
            file_api = ArchiveFileApi(archive_path, remote_path)
            with patch.object(utils.g, "api", SimpleNamespace(file=file_api)):
                force = utils.download_stats_chunks_to_buffer(
                    1,
                    project,
                    "/stats/7_project",
                    project_fs_dir,
                    False,
                    state,
                    [DummyStat()],
                )

            self.assertFalse(force)
            self.assertTrue(
                Path(
                    project_fs_dir,
                    "class_balance",
                    "chunk_0_1_7_1000_2026-01-01T00:00:00.npy",
                ).is_file()
            )

    def test_download_chunks_treats_corrupt_archive_as_cache_miss(self):
        state = utils.StatsRunState(datetime(2026, 1, 1))
        project = SimpleNamespace(id=7, name="project")
        archive_name = "7_project_chunks_2026-01-01T00:00:00.tar.gz"
        remote_path = f"/stats/7_project/{archive_name}"

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, "corrupt.tar.gz")
            Path(archive_path).write_bytes(b"not a tar archive")
            project_fs_dir = os.path.join(temp_dir, "buffer", "7_project")
            os.makedirs(project_fs_dir)
            file_api = ArchiveFileApi(archive_path, remote_path)

            with patch.object(utils.g, "api", SimpleNamespace(file=file_api)):
                force = utils.download_stats_chunks_to_buffer(
                    1,
                    project,
                    "/stats/7_project",
                    project_fs_dir,
                    False,
                    state,
                    [DummyStat()],
                )

            self.assertTrue(force)

    def test_save_chunks_matches_complete_chunk_identifier(self):
        chunk = "chunk_1_2_3"
        latest = datetime(2026, 2, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            savedir = Path(temp_dir, "class_balance")
            savedir.mkdir()
            old_chunk = savedir / "chunk_1_2_3_1000_2026-01-01T00:00:00.npy"
            other_chunk = savedir / "chunk_10_2_3_1000_2026-01-01T00:00:00.npy"
            np.save(old_chunk, [1])
            np.save(other_chunk, [1])
            tf_paths = [
                "/stats/project/class_balance/chunk_1_2_3_1000_2026-01-01T00:00:00.npy"
            ]

            utils.save_chunks(DummyStat(), chunk, temp_dir, tf_paths, latest)

            self.assertFalse(old_chunk.exists())
            self.assertTrue(other_chunk.exists())
            self.assertTrue(
                savedir.joinpath("chunk_1_2_3_1000_2026-02-01T00:00:00.npy").exists()
            )

    def test_integrity_rejects_duplicate_and_missing_chunk_identifiers(self):
        expected_chunks = {
            "chunk_1_2_3": [],
            "chunk_10_2_3": [],
        }
        updated_images = {2: []}
        all_images = {2: [SimpleNamespace(id=1)]}
        project = SimpleNamespace(items_count=1)
        dataset = SimpleNamespace(items_count=1)
        stats = [DummyStat()]

        with tempfile.TemporaryDirectory() as temp_dir:
            stat_dir = Path(temp_dir, DummyStat.basename_stem)
            stat_dir.mkdir()
            np.save(stat_dir / "chunk_1_2_3_1000_2026-01-01T00:00:00.npy", [1])
            np.save(stat_dir / "chunk_1_2_3_1000_2026-01-02T00:00:00.npy", [1])

            result = utils.check_idxs_integrity(
                project,
                [dataset],
                stats,
                temp_dir,
                expected_chunks,
                updated_images,
                all_images,
                False,
            )

        self.assertIs(result, all_images)

    def test_integrity_rejects_wrong_chunk_size_and_corrupt_npy(self):
        expected_chunks = {"chunk_1_2_3": []}
        updated_images = {2: []}
        all_images = {2: [SimpleNamespace(id=1)]}
        project = SimpleNamespace(items_count=1)
        dataset = SimpleNamespace(items_count=1)
        stats = [DummyStat()]

        cases = ("wrong-size", "corrupt")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                stat_dir = Path(temp_dir, DummyStat.basename_stem)
                stat_dir.mkdir()
                if case == "wrong-size":
                    np.save(
                        stat_dir / "chunk_1_2_3_999_2026-01-01T00:00:00.npy",
                        [1],
                    )
                else:
                    path = stat_dir / "chunk_1_2_3_1000_2026-01-01T00:00:00.npy"
                    path.write_bytes(b"not numpy")

                result = utils.check_idxs_integrity(
                    project,
                    [dataset],
                    stats,
                    temp_dir,
                    expected_chunks,
                    updated_images,
                    all_images,
                    False,
                )

            self.assertIs(result, all_images)

    def test_remove_junk_does_not_delete_duplicate_obsolete_chunk_twice(self):
        project = SimpleNamespace(id=3)
        datasets = [SimpleNamespace(id=2)]
        file_api = SimpleNamespace(listdir=Mock(return_value=[]))

        with tempfile.TemporaryDirectory() as temp_dir:
            stat_dir = Path(temp_dir, DummyStat.basename_stem)
            stat_dir.mkdir()
            paths = [
                stat_dir / "chunk_1_999_777_1000_2026-01-01T00:00:00.npy",
                stat_dir / "chunk_1_999_777_1000_2026-01-02T00:00:00.npy",
            ]
            for path in paths:
                np.save(path, [1])

            with patch.object(utils.g, "api", SimpleNamespace(file=file_api)):
                utils.remove_junk(
                    1,
                    "/stats/project",
                    project,
                    datasets,
                    temp_dir,
                    utils.StatsRunState(),
                )

            self.assertFalse(any(path.exists() for path in paths))

    def test_remove_junk_removes_malformed_chunk_without_crashing(self):
        project = SimpleNamespace(id=3)
        datasets = [SimpleNamespace(id=2)]
        file_api = SimpleNamespace(listdir=Mock(return_value=[]))

        with tempfile.TemporaryDirectory() as temp_dir:
            stat_dir = Path(temp_dir, DummyStat.basename_stem)
            stat_dir.mkdir()
            malformed_path = stat_dir / "bad.npy"
            np.save(malformed_path, [1])

            with patch.object(utils.g, "api", SimpleNamespace(file=file_api)):
                utils.remove_junk(
                    1,
                    "/stats/project",
                    project,
                    datasets,
                    temp_dir,
                    utils.StatsRunState(),
                )

            self.assertFalse(malformed_path.exists())

    def test_empty_snapshot_initializes_chunk_state_and_directories(self):
        state = utils.StatsRunState()
        stats = [DummyStat()]

        with tempfile.TemporaryDirectory() as temp_dir:
            utils.calculate_stats_and_save_chunks(
                {},
                stats,
                [],
                temp_dir,
                {},
                {},
                state,
            )

            self.assertIsNotNone(state.chunks_latest_datetime)
            self.assertTrue(Path(temp_dir, DummyStat.basename_stem).is_dir())

    def test_collect_heatmap_sample_scans_all_datasets(self):
        images = {
            10: [SimpleNamespace(id=1, dataset_id=10, labels_count=1)],
            20: [SimpleNamespace(id=2, dataset_id=20, labels_count=1)],
        }
        figures = {
            1: SimpleNamespace(id=101, class_id=1001, dataset_id=10, entity_id=1),
            2: SimpleNamespace(id=202, class_id=2002, dataset_id=20, entity_id=2),
        }

        def download(dataset_id, image_ids, skip_geometry=False):
            return {image_id: [figures[image_id]] for image_id in image_ids}

        api = SimpleNamespace(
            image=SimpleNamespace(figure=SimpleNamespace(download=Mock(side_effect=download)))
        )
        project_stats = {"objects": {"total": {"objectsInDataset": 2}}}
        project = SimpleNamespace(size="1")

        with patch.object(utils.g, "api", api), patch.object(utils.random, "random", return_value=0):
            image_ids, figure_ids = utils.collect_heatmap_sample(
                images, project_stats, project
            )

        self.assertEqual(image_ids, {10: {1}, 20: {2}})
        self.assertEqual(figure_ids, {1001: {101}, 2002: {202}})
        self.assertEqual(api.image.figure.download.call_count, 2)

    def test_empty_heatmap_sample_removes_stale_output(self):
        reporter = Mock()
        file_api = SimpleNamespace(
            exists=Mock(return_value=True),
            remove=Mock(),
        )
        before_publish = Mock()
        heatmaps = SimpleNamespace(basename_stem="classes_heatmaps")
        team = SimpleNamespace(id=1)

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            utils.g, "api", SimpleNamespace(file=file_api)
        ), patch.object(utils, "HeatmapStatusReporter", return_value=reporter):
            utils.calculate_and_upload_heatmaps(
                team,
                "/stats/project",
                temp_dir,
                7,
                heatmaps,
                {},
                {},
                before_publish,
            )

        self.assertEqual(before_publish.call_count, 2)
        file_api.remove.assert_called_once_with(
            1, "/stats/project/classes_heatmaps.png"
        )
        reporter.skipped.assert_called_once()

    def test_lost_lock_does_not_overwrite_newer_heatmap_status(self):
        reporter = Mock()
        file_api = SimpleNamespace(exists=Mock(), remove=Mock(), upload=Mock())
        before_publish = Mock(
            side_effect=utils.ActiveRequestOwnershipError("ownership changed")
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            utils.g, "api", SimpleNamespace(file=file_api)
        ), patch.object(utils, "HeatmapStatusReporter", return_value=reporter):
            utils.calculate_and_upload_heatmaps(
                SimpleNamespace(id=1),
                "/stats/project",
                temp_dir,
                7,
                SimpleNamespace(basename_stem="classes_heatmaps"),
                {},
                {},
                before_publish,
            )

        reporter.failed.assert_not_called()
        reporter.skipped.assert_not_called()
        file_api.remove.assert_not_called()
        file_api.upload.assert_not_called()

    def test_failed_old_heatmap_run_does_not_overwrite_newer_status(self):
        reporter = Mock()
        before_publish = Mock(
            side_effect=[
                None,
                None,
                utils.ActiveRequestOwnershipError("ownership changed"),
            ]
        )
        api = SimpleNamespace(
            file=SimpleNamespace(),
            image=SimpleNamespace(
                get_info_by_id_batch=Mock(side_effect=RuntimeError("download failed"))
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            utils.g, "api", api
        ), patch.object(utils, "HeatmapStatusReporter", return_value=reporter):
            utils.calculate_and_upload_heatmaps(
                SimpleNamespace(id=1),
                "/stats/project",
                temp_dir,
                7,
                SimpleNamespace(basename_stem="classes_heatmaps"),
                {10: {1}},
                {},
                before_publish,
            )

        self.assertEqual(before_publish.call_count, 3)
        reporter.failed.assert_not_called()

    def test_heatmap_keeps_original_error_when_ownership_probe_fails(self):
        reporter = Mock()
        before_publish = Mock(
            side_effect=[None, None, RuntimeError("ownership probe failed")]
        )
        api = SimpleNamespace(
            file=SimpleNamespace(),
            image=SimpleNamespace(
                get_info_by_id_batch=Mock(side_effect=ValueError("download failed"))
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            utils.g, "api", api
        ), patch.object(utils, "HeatmapStatusReporter", return_value=reporter):
            with self.assertRaisesRegex(ValueError, "download failed") as raised:
                utils.calculate_and_upload_heatmaps(
                    SimpleNamespace(id=1),
                    "/stats/project",
                    temp_dir,
                    7,
                    SimpleNamespace(basename_stem="classes_heatmaps"),
                    {10: {1}},
                    {},
                    before_publish,
                )

        self.assertEqual(before_publish.call_count, 3)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(str(raised.exception.__cause__), "ownership probe failed")
        reporter.failed.assert_not_called()

    def test_heatmap_render_uses_isolated_temporary_directories(self):
        render_dirs = []
        upload_sources_exist = []

        class Heatmaps:
            basename_stem = "classes_heatmaps"

            def update2(self, image, figures, skip_broken_geometry=False):
                return None

            def to_image(self, path):
                render_dirs.append(os.path.dirname(path))
                Path(path).write_bytes(b"png")

        image = SimpleNamespace(id=1)
        figure = SimpleNamespace(id=10, class_id=100)
        file_api = SimpleNamespace(
            upload=Mock(
                side_effect=lambda _team_id, src, _dst: upload_sources_exist.append(
                    os.path.exists(src)
                )
            )
        )
        api = SimpleNamespace(
            image=SimpleNamespace(
                get_info_by_id_batch=Mock(return_value=[image]),
                figure=SimpleNamespace(download=Mock(return_value={1: [figure]})),
            ),
            file=file_api,
        )
        reporter = Mock()

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            utils.g, "api", api
        ), patch.object(utils, "HeatmapStatusReporter", return_value=reporter):
            for _ in range(2):
                utils.calculate_and_upload_heatmaps(
                    SimpleNamespace(id=1),
                    "/stats/project",
                    temp_dir,
                    7,
                    Heatmaps(),
                    {1: {1}},
                    {100: {10}},
                )

        self.assertEqual(len(set(render_dirs)), 2)
        self.assertEqual(upload_sources_exist, [True, True])
        self.assertTrue(all(not os.path.exists(path) for path in render_dirs))


if __name__ == "__main__":
    unittest.main()
