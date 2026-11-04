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

"""Regression tests for isolated, fresh lifelong-learning KB indexes."""

import gc
import os
import tempfile
import threading
import unittest
import warnings
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.config import BaseConfig
from sedna.common.file_ops import FileOps
from sedna.core.lifelong_learning import LifelongLearning
from sedna.core.lifelong_learning.knowledge_management.\
    cloud_knowledge_management import CloudKnowledgeManagement
from sedna.datasources import BaseDataSource


class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class TestLocalKnowledgeIndex(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.work = self.root / "work"
        self.work.mkdir()
        previous_directory = os.getcwd()
        self.addCleanup(os.chdir, previous_directory)
        os.chdir(str(self.work))
        environment = patch.dict(os.environ, {
            "OUTPUT_URL": str(self.root / "cloud"),
            "edge_output_url": str(self.root / "edge"),
            "BACKEND_TYPE": "UNKNOWN",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def make_manager(self, output_url=None):
        return CloudKnowledgeManagement({
            "cloud_output_url": output_url or str(self.root / "cloud"),
            "task_index": "index.pkl",
        }, seen_estimator=None, unseen_estimator=None)

    def make_job(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            job = LifelongLearning(seen_estimator=None)
        job.report_task_info = Mock()
        seen = job.cloud_knowledge_management.seen_estimator

        def evaluate(data, task_index, **kwargs):
            return FileOps.load(task_index), []

        # Keep file transfer and evaluation orchestration real, replacing only
        # the ML estimator and external KB/LC services.
        seen.evaluate = Mock(side_effect=evaluate)
        job.kb_server.update_task_status = Mock(return_value=None)
        return job

    def write_index(self, path, version):
        FileOps.dump({"version": version}, str(path))
        return str(path)

    def test_local_index_is_absolute_and_outside_working_directory(self):
        manager = self.make_manager()

        local = Path(manager.local_task_index_url)
        self.assertTrue(local.is_absolute())
        self.assertEqual(local.name, "index.pkl")
        self.assertNotEqual(local.parent, self.work)
        self.assertTrue(local.parent.is_dir())
        self.assertNotEqual(str(local), manager.task_index)

    def test_jobs_with_the_same_output_have_independent_local_indexes(self):
        first = self.make_manager()
        second = self.make_manager()

        self.assertEqual(first.task_index, second.task_index)
        self.assertNotEqual(first.local_task_index_url,
                            second.local_task_index_url)
        self.write_index(Path(first.local_task_index_url), "first")
        self.assertFalse(Path(second.local_task_index_url).exists())

    def test_remote_output_keeps_a_local_working_copy(self):
        for output in ("s3://bucket/job", "https://example.com/job"):
            with self.subTest(output=output):
                manager = self.make_manager(output)
                self.assertEqual(manager.task_index, output + "/index.pkl")
                self.assertTrue(os.path.isabs(manager.local_task_index_url))
                self.write_index(Path(manager.local_task_index_url), "local")
                self.assertEqual(FileOps.load(manager.local_task_index_url),
                                 {"version": "local"})

    def test_local_index_is_stable_when_working_directory_changes(self):
        manager = self.make_manager()
        local = manager.local_task_index_url
        os.chdir(str(self.root))

        self.write_index(Path(local), "local")

        self.assertEqual(FileOps.load(local), {"version": "local"})
        self.assertTrue(os.path.isabs(local))
        self.assertFalse((self.root / "index.pkl").exists())

    def test_temporary_index_is_cleaned_up_with_its_manager(self):
        manager = self.make_manager()
        local = Path(manager.local_task_index_url)
        self.write_index(local, "temporary")

        del manager
        gc.collect()

        self.assertFalse(local.exists())
        self.assertNotEqual(local.parent, self.work)
        self.assertFalse(local.parent.exists())

    def test_evaluation_ignores_an_existing_working_directory_index(self):
        stale = self.work / "index.pkl"
        self.write_index(stale, "stale")
        source = self.write_index(self.root / "source.pkl", "fresh")
        os.environ["MODEL_URLS"] = source
        job = self.make_job()

        self.assertEqual(job.evaluate(data="validation"), {"version": "fresh"})
        self.assertEqual(FileOps.load(str(stale)), {"version": "stale"})

    def test_evaluation_does_not_create_a_working_directory_index(self):
        source = self.write_index(self.root / "source.pkl", "fresh")
        os.environ["MODEL_URLS"] = source
        job = self.make_job()
        job.kb_server.update_task_status.return_value = self.write_index(
            self.root / "kb-result.pkl", "filtered")

        self.assertEqual(job.evaluate(data="validation"), {"version": "fresh"})
        self.assertFalse((self.work / "index.pkl").exists())

    def test_evaluation_refreshes_an_existing_local_copy(self):
        source = self.write_index(self.root / "source.pkl", "fresh")
        os.environ["MODEL_URLS"] = source
        job = self.make_job()
        self.write_index(Path(job.cloud_knowledge_management.
                              local_task_index_url), "stale")

        self.assertEqual(job.evaluate(data="validation"), {"version": "fresh"})
        self.assertEqual(FileOps.load(source), {"version": "fresh"})

    def test_later_rounds_refresh_the_same_source_url(self):
        source = self.root / "source.pkl"
        os.environ["MODEL_URLS"] = str(source)
        job = self.make_job()

        # A KB service result leaves the downloaded working copy in place.
        def kb_result(*args, **kwargs):
            return self.write_index(self.root / "kb-result.pkl", "filtered")

        job.kb_server.update_task_status.side_effect = kb_result
        for version in ("first", "second"):
            self.write_index(source, version)
            self.assertEqual(job.evaluate(data="validation"),
                             {"version": version})
        self.assertEqual(job.cloud_knowledge_management.
                         seen_estimator.evaluate.call_count, 2)

    def test_later_rounds_use_a_changed_model_url(self):
        job = self.make_job()
        job.kb_server.update_task_status.side_effect = lambda *a, **k: (
            self.write_index(self.root / "kb-result.pkl", "filtered"))

        for version in ("first", "second"):
            os.environ["MODEL_URLS"] = self.write_index(
                self.root / (version + ".pkl"), version)
            self.assertEqual(job.evaluate(data="validation"),
                             {"version": version})

    def test_evaluation_falls_back_to_the_published_index(self):
        job = self.make_job()
        published = job.cloud_knowledge_management.task_index
        self.write_index(Path(published), "published")
        self.write_index(Path(job.cloud_knowledge_management.
                              local_task_index_url), "stale")

        self.assertEqual(job.evaluate(data="validation"),
                         {"version": "published"})
        self.assertEqual(FileOps.load(published), {"version": "published"})

    def test_keywords_callbacks_upload_and_reporting_are_preserved(self):
        os.environ["MODEL_URLS"] = self.write_index(
            self.root / "source.pkl", "fresh")
        job = self.make_job()
        callback = Mock(return_value="processed")

        result = job.evaluate("validation", post_process=callback, limit=7)

        self.assertEqual(result, "processed")
        callback.assert_called_once_with({"version": "fresh"})
        manager = job.cloud_knowledge_management
        manager.seen_estimator.evaluate.assert_called_once_with(
            data="validation", task_index=manager.local_task_index_url,
            limit=7)
        self.assertEqual(FileOps.load(manager.task_index),
                         {"version": "fresh"})
        self.assertEqual(job.report_task_info.call_args[1],
                         {"kind": "eval"})

    def test_registered_callback_is_preserved(self):
        os.environ["MODEL_URLS"] = self.write_index(
            self.root / "source.pkl", "fresh")
        job = self.make_job()
        callback = Mock(return_value="processed")

        with patch.object(ClassFactory, "get_cls",
                          return_value=callback) as get:
            result = job.evaluate("validation", post_process="callback")

        get.assert_called_once_with(ClassType.CALLBACK, "callback")
        callback.assert_called_once_with({"version": "fresh"})
        self.assertEqual(result, "processed")

    def test_training_writes_to_the_isolated_local_index(self):
        job = self.make_job()
        manager = job.cloud_knowledge_management
        manager.seen_estimator.train = Mock(return_value=({}, {"seen": True}))
        manager.unseen_estimator.train = Mock(return_value=({}, {}))
        manager.seen_estimator.model_info = Mock(return_value=[])
        manager.update_kb = Mock(return_value=manager.task_index)

        job.train("training")

        manager.update_kb.assert_called_once_with(manager.local_task_index_url)
        self.assertEqual(FileOps.load(manager.local_task_index_url),
                         {"seen_task": {"seen": True}, "unseen_task": {}})
        self.assertFalse((self.work / "index.pkl").exists())

    def test_update_refreshes_the_isolated_local_index(self):
        job = self.make_job()
        manager = job.cloud_knowledge_management
        source = self.write_index(self.root / "source.pkl", "fresh")
        os.environ["CLOUD_KB_INDEX"] = source
        self.write_index(Path(manager.local_task_index_url), "stale")
        seen = BaseDataSource(data_type="test")
        unseen = BaseDataSource(data_type="test")
        for samples in (seen, unseen):
            samples.x = np.array([[1]])
            samples.y = np.array([1])
        recognize = Mock(return_value=(seen, unseen))
        decide = Mock(return_value=([], []))
        manager.seen_estimator.update = Mock(return_value={"seen": True})
        manager.unseen_estimator.update = Mock(return_value={})
        manager.update_kb = Mock(return_value=manager.task_index)

        with patch.object(BaseConfig, "data_path_prefix", os.path.sep), \
                patch.object(ClassFactory, "get_cls", side_effect=[
                    Mock(return_value=recognize), Mock(return_value=decide)]):
            result = job.update("training")

        self.assertEqual(result, manager.task_index)
        self.assertEqual(FileOps.load(manager.local_task_index_url),
                         {"version": "fresh"})
        manager.seen_estimator.update.assert_called_once_with(
            [], [], task_index=manager.local_task_index_url)
        manager.unseen_estimator.update.assert_called_once_with(
            [], [], task_index=manager.local_task_index_url)
        manager.update_kb.assert_called_once_with(
            {"seen_task": {"seen": True}, "unseen_task": {}})
        self.assertFalse((self.work / "index.pkl").exists())

    def test_missing_local_source_does_not_evaluate_cached_index(self):
        job = self.make_job()
        manager = job.cloud_knowledge_management
        self.write_index(Path(manager.local_task_index_url), "stale")
        os.environ["MODEL_URLS"] = str(self.root / "missing.pkl")

        with self.assertRaises(FileNotFoundError):
            job.evaluate("validation")

        manager.seen_estimator.evaluate.assert_not_called()
        job.kb_server.update_task_status.assert_not_called()
        job.report_task_info.assert_not_called()

    def test_http_refresh_and_download_failure(self):
        source = self.root / "source.pkl"
        handler = partial(QuietHTTPRequestHandler, directory=str(self.root))
        server = HTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        os.environ["MODEL_URLS"] = "http://127.0.0.1:{}/source.pkl".format(
            server.server_port)
        job = self.make_job()
        job.kb_server.update_task_status.side_effect = lambda *a, **k: (
            self.write_index(self.root / "kb-result.pkl", "filtered"))

        for version in ("first", "second"):
            self.write_index(source, version)
            self.assertEqual(job.evaluate(data="validation"),
                             {"version": version})

        source.unlink()
        with self.assertRaises(OSError):
            job.evaluate(data="validation")
        self.assertEqual(job.cloud_knowledge_management.
                         seen_estimator.evaluate.call_count, 2)
        self.assertEqual(job.kb_server.update_task_status.call_count, 2)
        self.assertEqual(job.report_task_info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
