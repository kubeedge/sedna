# Copyright 2021 The KubeEdge Authors.
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

import os
import re
import glob
import gzip
import json
import shutil
import tempfile
import mimetypes
import concurrent.futures
from urllib.parse import urlparse

import requests

from robosdk.utils.context import Context
from robosdk.utils.util import singleton


@singleton
def _create_minio_client():
    import minio

    _url = Context.get("S3_ENDPOINT_URL", "http://s3.amazonaws.com")
    if not (_url.startswith("http://") or _url.startswith("https://")):
        _url = f"https://{_url}"
    url = urlparse(_url)
    use_ssl = url.scheme == 'https' if url.scheme else True

    s3 = minio.Minio(
        url.netloc,
        access_key=Context.get("ACCESS_KEY_ID", ""),
        secret_key=Context.get("SECRET_ACCESS_KEY", ""),
        secure=use_ssl
    )
    return s3


class FileOps:
    """
    This is a class with some class methods to handle some files or folder.
    """

    _S3_PREFIX = "s3://"
    _OBS_PREFIX = "obs://"
    _LOCAL_PREFIX = "file://"
    _URI_RE = "https?://(.+)/(.+)"
    _HTTP_PREFIX = "http(s)://"
    _HEADERS_SUFFIX = "-headers"
    SUPPORT_PROTOCOLS = (_OBS_PREFIX, _S3_PREFIX, _LOCAL_PREFIX, _HTTP_PREFIX)

    @classmethod
    def _normalize_uri(cls, uri: str) -> str:
        for src, dst in [
            ("/", cls._LOCAL_PREFIX),
            (cls._OBS_PREFIX, cls._S3_PREFIX)
        ]:
            if uri.startswith(src):
                return uri.replace(src, dst, 1)
        return uri

    @classmethod
    def download(cls, uri: str, out_dir: str = None,
                 untar: bool = False) -> str:
        """
        Download the uri to local directory.
        Support protocols: http(s), s3.
        """

        uri = cls._normalize_uri(uri)
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        elif not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        if uri.startswith(cls._S3_PREFIX):
            dst = cls.download_s3(uri, out_dir)
        elif uri.startswith(cls._LOCAL_PREFIX):
            dst = cls.download_local(uri, out_dir)
        elif re.search(cls._URI_RE, uri):
            dst = cls.download_from_uri(uri, out_dir)
        else:
            raise Exception("Cannot recognize storage type for %s.\n"
                            "%r are the current available storage type." %
                            (uri, cls.SUPPORT_PROTOCOLS))
        if os.path.isdir(dst):
            _dst = os.path.join(dst, os.path.basename(uri))
            if os.path.exists(_dst):
                dst = _dst
        if untar:
            if os.path.isfile(dst):
                return cls._untar(dst)
            if os.path.isdir(dst):
                _ = map(cls._untar, glob.glob(os.path.join(dst, "*")))
        return dst

    @classmethod
    def upload(cls, src: str, dst: str, tar=False, clean=True) -> str:
        basename = os.path.basename(src)
        dst = dst.rstrip(basename)
        if tar:
            src = cls._tar(src, f"{src.rstrip(os.path.sep)}.tar.gz")

        if dst.startswith(cls._S3_PREFIX):
            cls.upload_s3(src, dst)
        else:
            if not os.path.isdir(dst):
                os.makedirs(dst)
            cls.download_local(src, dst)
        if clean and os.path.exists(src):
            cls.delete(src)
        return dst

    @classmethod
    def upload_s3(cls, src, dst):

        s3 = _create_minio_client()
        parsed = urlparse(dst, scheme='s3')
        bucket_name = parsed.netloc

        def _s3_upload(_file, fname=""):
            _file_handle = open(_file, 'rb')
            _file_handle.seek(0, os.SEEK_END)
            size = _file_handle.tell()
            _file_handle.seek(0)
            if not fname:
                fname = os.path.basename(fname)
            s3.put_object(bucket_name, fname, _file_handle, size)
            _file_handle.close()
            return size

        if os.path.isdir(src):
            for root, _, files in os.walk(src):
                for file in files:
                    filepath = os.path.join(root, file)
                    name = os.path.relpath(filepath, src)
                    _s3_upload(filepath, name)
        elif os.path.isfile(src):
            _s3_upload(src, parsed.path.lstrip("/"))

    @classmethod
    def download_s3(cls, uri: str, out_dir: str = None) -> str:
        client = _create_minio_client()
        count = cls._download_s3(client, uri, out_dir)
        if count == 0:
            raise RuntimeError(
                f"Failed to fetch files. The path {uri} does not exist.")
        return out_dir

    @classmethod
    def download_local(cls, uri: str, out_dir: str) -> str:
        local_path = uri.replace(cls._LOCAL_PREFIX, "/", 1)
        if not os.path.exists(local_path):
            raise RuntimeError(f"Local path {uri} does not exist.")

        if os.path.isdir(local_path):
            local_path = os.path.join(local_path, "*")

        for src in glob.glob(local_path):
            _, tail = os.path.split(src)
            dest_path = os.path.join(out_dir, tail)
            if src == dest_path:
                continue
            shutil.copy(src, dest_path)
        return out_dir

    @classmethod
    def download_from_uri(cls, uri, out_dir=None) -> str:
        url = urlparse(uri)
        filename = os.path.basename(url.path)
        mimetype, encoding = mimetypes.guess_type(url.path)
        local_path = os.path.join(out_dir, filename)

        if not filename:
            raise ValueError(f'No filename contained in URI: {uri}')

        host_uri = url.hostname

        headers_json = os.getenv(host_uri + cls._HEADERS_SUFFIX, "{}")
        headers = json.loads(headers_json)

        with requests.get(uri, stream=True, headers=headers) as response:
            if response.status_code != 200:
                raise RuntimeError("URI: %s returned a %s response code." %
                                   (uri, response.status_code))

            if encoding == 'gzip':
                stream = gzip.GzipFile(fileobj=response.raw)
                local_path = os.path.join(out_dir, f'{filename}.tar')
            else:
                stream = response.raw
            with open(local_path, 'wb') as out:
                shutil.copyfileobj(stream, out)

        return local_path

    @classmethod
    def download_s3_with_multi_files(cls, download_files,
                                     base_uri, base_out_dir):
        client = _create_minio_client()
        total_count = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            todos = []
            for dfile in set(download_files):
                dir_ = os.path.dirname(dfile)
                uri = base_uri.rstrip("/") + "/" + dfile
                out_dir = os.path.join(base_out_dir, dir_)
                todos.append(
                    executor.submit(cls._download_s3, client, uri, out_dir))

            for done in concurrent.futures.as_completed(todos):
                count = done.result()
                if count == 0:
                    continue
                total_count += count

    @classmethod
    def _download_s3(cls, client, uri, out_dir):
        """
        The function downloads specified file or folder to local directory.
        this function supports:
        1. when downloading the specified file, keep the name of the file.
        2. when downloading the specified folder, keep the name of the folder.

        Parameters:
        client: s3 client
        uri(string): url in s3, e.g. file url: s3://dev/data/data.txt
        out_dir(string):  local directory address, e.g. /tmp/data/

        Returns:
        int: files of number in s3_url
        """
        bucket_args = uri.replace(cls._S3_PREFIX, "", 1).split("/", 1)
        bucket_name = bucket_args[0]
        bucket_path = len(bucket_args) > 1 and bucket_args[1] or ""

        objects = client.list_objects(bucket_name,
                                      prefix=bucket_path,
                                      recursive=True,
                                      use_api_v1=True)
        count = 0

        root_path = os.path.split(os.path.normpath(bucket_path))[0]
        for obj in objects:
            # Replace any prefix from the object key with out_dir
            subdir_object_key = obj.object_name[len(root_path):].strip("/")
            # fget_object handles directory creation if does not exist
            if not obj.is_dir:
                local_file = os.path.join(
                    out_dir,
                    subdir_object_key or os.path.basename(obj.object_name)
                )
                client.fget_object(bucket_name, obj.object_name, local_file)
                count += 1
        return count

    @classmethod
    def _untar(cls, src, dst=None):
        import tarfile
        import zipfile
        if not (os.path.isfile(src) and str(src).endswith((".gz", ".zip"))):
            return src
        if dst is None:
            dst = os.path.dirname(src)
        _bname, _bext = os.path.splitext(os.path.basename(src))
        if _bext == ".zip":
            with zipfile.ZipFile(src, 'r') as zip_ref:
                zip_ref.extractall(dst)
        else:
            with tarfile.open(src, 'r:gz') as tar_ref:
                tar_ref.extractall(path=dst)
        if os.path.isfile(src):
            cls.delete(src)
        checkname = os.path.join(dst, _bname)
        return checkname if os.path.exists(checkname) else dst

    @classmethod
    def _tar(cls, src, dst) -> str:
        import tarfile
        with tarfile.open(dst, 'w:gz') as tar:
            if os.path.isdir(src):
                for root, _, files in os.walk(src):
                    for file in files:
                        filepath = os.path.join(root, file)
                        tar.add(filepath)
            elif os.path.isfile(src):
                tar.add(os.path.realpath(src))
        return dst

    @classmethod
    def delete(cls, path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass
