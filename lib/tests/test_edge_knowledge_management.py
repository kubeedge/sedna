# Copyright 2026 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for lazy edge knowledge-base directory creation."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from sedna.common.config import Context
from sedna.datasources import BaseDataSource
from sedna.core.lifelong_learning import LifelongLearning
from sedna.core.lifelong_learning.knowledge_management import (
    edge_knowledge_management as edge_km,
)


class EdgeKnowledgeManagementTest(unittest.TestCase):
    """Constructing a job must not require writable edge storage."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.edge_output = self.root / "edge" / "kb"
        self.parameters = {
            "edge_output_url": str(self.edge_output),
            "unseen_save_url": str(self.root / "uploaded"),
        }
        self.parameter_patch = mock.patch.object(
            Context, "parameters", self.parameters)
        self.parameter_patch.start()
        self.addCleanup(self.parameter_patch.stop)

    def make_manager(self):
        return edge_km.EdgeKnowledgeManagement(
            {"ll_kb_server": None}, None, None)

    def test_lifelong_init_does_not_write_default_edge_directory(self):
        self.parameters.pop("edge_output_url")
        with mock.patch.dict(os.environ, {"BACKEND_TYPE": "UNKNOWN"}), \
                mock.patch.object(edge_km.os, "makedirs",
                                  side_effect=PermissionError) as make_dir:
            job = LifelongLearning(seen_estimator=None)
        make_dir.assert_not_called()
        self.assertEqual(job.edge_knowledge_management.local_unseen_save_url,
                         "/var/lib/sedna/kb/unseen_samples")
        self.assertFalse(job.start_inference_service)

    def test_custom_edge_directory_is_not_created_during_initialization(self):
        manager = self.make_manager()
        self.assertEqual(manager.local_unseen_save_url,
                         str(self.edge_output / "unseen_samples"))
        self.assertFalse(self.edge_output.exists())

    def test_services_create_directory_before_scheduling_observer(self):
        manager = self.make_manager()
        with mock.patch.object(edge_km, "Observer") as observer, \
                mock.patch.object(edge_km, "ModelHotUpdateThread") as updater:
            def check_directory(*args, **kwargs):
                self.assertTrue(Path(manager.local_unseen_save_url).is_dir())

            observer.return_value.schedule.side_effect = check_directory
            manager.start_services()
        observer.return_value.schedule.assert_called_once_with(
            mock.ANY, manager.local_unseen_save_url, recursive=True)
        observer.return_value.start.assert_called_once_with()
        updater.return_value.start.assert_called_once_with()

    def test_start_services_preserves_existing_samples(self):
        manager = self.make_manager()
        saved = Path(manager.local_unseen_save_url) / "existing.png"
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"existing sample")
        with mock.patch.object(edge_km, "Observer"), \
                mock.patch.object(edge_km, "ModelHotUpdateThread"):
            manager.start_services()
        self.assertEqual(saved.read_bytes(), b"existing sample")

    def test_services_report_permission_error_before_starting_threads(self):
        manager = self.make_manager()
        with mock.patch.object(edge_km.os, "makedirs",
                               side_effect=PermissionError), \
                mock.patch.object(edge_km, "Observer") as observer, \
                mock.patch.object(edge_km, "ModelHotUpdateThread") as updater:
            with self.assertRaises(PermissionError):
                manager.start_services()
        observer.assert_not_called()
        updater.assert_not_called()

    def test_real_read_only_directory_allows_initialization(self):
        if not hasattr(os, "geteuid") or os.geteuid() == 0:
            self.skipTest("requires a non-root POSIX user")
        read_only = self.root / "read_only"
        read_only.mkdir()
        self.addCleanup(read_only.chmod, 0o700)
        read_only.chmod(0o500)
        self.parameters["edge_output_url"] = str(read_only / "kb")
        manager = self.make_manager()
        with mock.patch.object(edge_km, "Observer") as observer:
            with self.assertRaises(PermissionError):
                manager.start_services()
        observer.assert_not_called()
        self.assertFalse((read_only / "kb").exists())

    def test_file_at_sample_directory_is_reported_when_storage_is_used(self):
        manager = self.make_manager()
        path = Path(manager.local_unseen_save_url)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"existing file")
        samples = BaseDataSource()
        samples.x = [b"sample"]
        with self.assertRaises(FileExistsError):
            manager.save_unseen_samples(samples, mock.Mock())
        with self.assertRaises(FileExistsError):
            manager.start_services()
        self.assertEqual(path.read_bytes(), b"existing file")

    def test_save_image_without_starting_services(self):
        manager = self.make_manager()
        samples = BaseDataSource()
        image = mock.Mock()
        image.save.side_effect = lambda path: Path(path).write_bytes(b"image")
        samples.x = [{"image": image}]
        manager.save_unseen_samples(samples, post_process=None)
        saved = list(Path(manager.local_unseen_save_url).glob("*.png"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_bytes(), b"image")
        self.assertIsNone(manager.unseen_sample_observer)

    def test_save_with_custom_postprocess_receives_existing_directory(self):
        manager = self.make_manager()
        samples = BaseDataSource()
        samples.x = [b"custom sample"]

        def save_sample(sample, directory):
            self.assertTrue(Path(directory).is_dir())
            (Path(directory) / "custom.sample").write_bytes(sample)

        manager.save_unseen_samples(samples, post_process=save_sample)
        saved = Path(manager.local_unseen_save_url) / "custom.sample"
        self.assertEqual(saved.read_bytes(), b"custom sample")

    def test_save_file_preserves_source_and_existing_samples(self):
        manager = self.make_manager()
        source = self.root / "source.png"
        source.write_bytes(b"sample data")
        samples = BaseDataSource()
        samples.x = [[str(source)]]
        manager.save_unseen_samples(samples, post_process=None)
        second = self.root / "second.png"
        second.write_bytes(b"another sample")
        samples.x = [[str(second)]]
        manager.save_unseen_samples(samples, post_process=None)
        self.assertEqual(source.read_bytes(), b"sample data")
        saved = Path(manager.local_unseen_save_url) / "source.png"
        self.assertEqual(saved.read_bytes(), b"sample data")
        self.assertEqual(len(list(saved.parent.iterdir())), 2)

    def test_save_samples_reports_permission_error_before_postprocess(self):
        manager = self.make_manager()
        samples = BaseDataSource()
        samples.x = [b"sample"]
        postprocess = mock.Mock()
        with mock.patch.object(edge_km.os, "makedirs",
                               side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                manager.save_unseen_samples(samples, postprocess)
        postprocess.assert_not_called()

    def test_real_observer_uploads_a_sample(self):
        manager = self.make_manager()
        with mock.patch.object(edge_km, "ModelHotUpdateThread"):
            manager.start_services()
        observer = manager.unseen_sample_observer
        self.addCleanup(observer.join, 5)
        self.addCleanup(observer.stop)
        samples = BaseDataSource()
        samples.x = [b"uploaded sample"]

        def save_sample(sample, directory):
            (Path(directory) / "sample.png").write_bytes(sample)

        manager.save_unseen_samples(samples, post_process=save_sample)
        uploaded = self.root / "uploaded" / "sample.png"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (uploaded.is_file() and
                    uploaded.read_bytes() == b"uploaded sample"):
                break
            time.sleep(0.05)
        self.assertTrue(uploaded.is_file())
        self.assertEqual(uploaded.read_bytes(), b"uploaded sample")


if __name__ == "__main__":
    unittest.main()
