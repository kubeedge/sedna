#!/usr/bin/env python3

# Copyright 2021 The KubeEdge Authors.
# Copyright 2020 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modify from https://github.com/kubeflow/kfserving/blob/master/python/kfserving/kfserving/storage.py # noqa

import concurrent.futures
import glob
import gzip
import json
import logging
import mimetypes
import os
import re
import sys
import shutil
import tempfile
import tarfile
import zipfile

import minio
import requests
from urllib.parse import urlparse

_S3_PREFIX = "s3://"
_OBS_PREFIX = "obs://"
_LOCAL_PREFIX = "file://"
_URI_RE = "https?://(.+)/(.+)"
_HTTP_PREFIX = "http(s)://"
_HEADERS_SUFFIX = "-headers"

SUPPORT_PROTOCOLS = (_OBS_PREFIX, _S3_PREFIX, _LOCAL_PREFIX, _HTTP_PREFIX)

LOG = logging.getLogger(__name__)


def setup_logger():
    format = '%(asctime)s %(levelname)s %(funcName)s:%(lineno)s] %(message)s'
    logging.basicConfig(format=format)
    LOG.setLevel(os.getenv('LOG_LEVEL', 'INFO'))


def _normalize_uri(uri: str) -> str:
    for src, dst in [
        ("/", _LOCAL_PREFIX),
        (_OBS_PREFIX, _S3_PREFIX)
    ]:
        if uri.startswith(src):
            return uri.replace(src, dst, 1)
    return uri


def download(uri: str, out_dir: str = None) -> str:
    """ Download the uri to local directory.

    Support procotols: http, s3.
    Note when uri ends with .tar.gz/.tar/.zip, this will extract it
    """
    LOG.info("Copying contents of %s to local %s", uri, out_dir)

    uri = _normalize_uri(uri)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if uri.startswith(_S3_PREFIX):
        download_s3(uri, out_dir)
    elif uri.startswith(_LOCAL_PREFIX):
        download_local(uri, out_dir)
    elif re.search(_URI_RE, uri):
        download_from_uri(uri, out_dir)
    else:
        raise Exception("Cannot recognize storage type for %s.\n"
                        "%r are the current available storage type." %
                        (uri, SUPPORT_PROTOCOLS))

    LOG.info("Successfully copied %s to %s", uri, out_dir)
    return out_dir


def indirect_download(indirect_uri: str, out_dir: str = None) -> str:
    """ Download the uri to local directory.

    Support procotols: http, s3.
    Note when uri ends with .tar.gz/.tar/.zip, this will extract it
    """
    tmpdir = tempfile.mkdtemp()
    download(indirect_uri, tmpdir)
    files = os.listdir(tmpdir)

    if len(files) != 1:
        raise Exception("indirect url %s should be file, not directory"
                        % indirect_uri)

    download_files = set()
    with open(os.path.join(tmpdir, files[0])) as f:
        base_uri = None
        for line_no, line in enumerate(f):
            line = line.strip()
            if line.startswith('#'):
                continue
            if line:
                if base_uri is None:
                    base_uri = line
                else:
                    file_name = line
                    download_files.add(file_name)

    if not download_files:
        LOG.info("no files to download for indirect url %s",
                 indirect_uri)
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    LOG.info("To download %s files IN-DIRECT %s to %s",
             len(download_files), indirect_uri, out_dir)

    uri = _normalize_uri(base_uri)
    # only support s3 for indirect download
    if uri.startswith(_S3_PREFIX):
        download_s3_with_multi_files(download_files, uri, out_dir)
    else:
        LOG.warning("unsupported %s for indirect url %s, skipped",
                    uri, indirect_uri)
        return
    LOG.info("Successfully download files IN-DIRECT %s to %s",
             indirect_uri, out_dir)
    return


def download_s3(uri, out_dir: str):
    client = _create_minio_client()
    count = _download_s3(client, uri, out_dir)
    if count == 0:
        raise RuntimeError("Failed to fetch files."
                           "The path %s does not exist." % (uri))
    LOG.info("downloaded %d files for %s.", count, uri)


def download_s3_with_multi_files(download_files,
                                 base_uri, base_out_dir):
    client = _create_minio_client()
    total_count = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        todos = []
        for dfile in set(download_files):
            dir_ = os.path.dirname(dfile)
            uri = base_uri.rstrip("/") + "/" + dfile
            out_dir = os.path.join(base_out_dir, dir_)
            todos.append(executor.submit(_download_s3, client, uri, out_dir))

        for done in concurrent.futures.as_completed(todos):
            count = done.result()
            if count == 0:
                LOG.warning("failed to download %s in base uri(%s)",
                            dfile, base_uri)
                continue

            total_count += count
    LOG.info("downloaded %d files for base_uri %s to local dir %s.",
             total_count, base_uri, base_out_dir)


def _download_s3(client, uri, out_dir):
    """
    The function downloads specified file or folder to local directory address.
    this function supports:
    1. when downloading the specified file, keep the name of the file itself.
    2. when downloading the specified folder, keep the name of the folder itself.

    Parameters:
    client: s3 client
    s3_url(string): url in s3, e.g. file url: s3://dev/data/data.txt, directory url: s3://dev/data
    out_dir(string):  local directory address, e.g. /tmp/data/

    Returns:
    int: files of number in s3_url
    """
    bucket_args = uri.replace(_S3_PREFIX, "", 1).split("/", 1)
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
            LOG.debug("downloading count:%d, file:%s",
                      count, subdir_object_key)
            client.fget_object(bucket_name, obj.object_name, local_file)
            _extract_compress(local_file, out_dir)

            count += 1

    return count


def download_local(uri, out_dir=None):
    local_path = uri.replace(_LOCAL_PREFIX, "/", 1)
    if not os.path.exists(local_path):
        raise RuntimeError("Local path %s does not exist." % (uri))

    if out_dir is None:
        return local_path
    elif not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if os.path.isdir(local_path):
        local_path = os.path.join(local_path, "*")

    for src in glob.glob(local_path):
        _, tail = os.path.split(src)
        dest_path = os.path.join(out_dir, tail)
        LOG.info("Linking: %s to %s", src, dest_path)
        os.symlink(src, dest_path)
    return out_dir


def download_from_uri(uri, out_dir=None):
    url = urlparse(uri)
    filename = os.path.basename(url.path)
    mimetype, encoding = mimetypes.guess_type(url.path)
    local_path = os.path.join(out_dir, filename)

    if filename == '':
        raise ValueError('No filename contained in URI: %s' % (uri))

    # Get header information from host url
    headers = {}
    host_uri = url.hostname

    headers_json = os.getenv(host_uri + _HEADERS_SUFFIX, "{}")
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
    return _extract_compress(local_path, out_dir)


def _extract_compress(local_path, out_dir):
    mimetype, encoding = mimetypes.guess_type(local_path)
    if mimetype in ["application/x-tar", "application/zip"]:
        if mimetype == "application/x-tar":
            archive = tarfile.open(local_path, 'r', encoding='utf-8')
        else:
            archive = zipfile.ZipFile(local_path, 'r')
        archive.extractall(out_dir)
        archive.close()
        os.remove(local_path)

    return out_dir


def _create_minio_client():
    url = urlparse(os.getenv("S3_ENDPOINT_URL", "http://s3.amazonaws.com"))
    use_ssl = url.scheme == 'https' if url.scheme else True
    return minio.Minio(
        url.netloc,
        access_key=os.getenv("ACCESS_KEY_ID", ""),
        secret_key=os.getenv("SECRET_ACCESS_KEY", ""),
        secure=use_ssl
    )


def main():
    setup_logger()
    if len(sys.argv) < 2 or len(sys.argv) % 2 == 0:
        LOG.error("Usage: download.py "
                  "src_uri dest_path [src_uri dest_path]")
        sys.exit(1)

    indirect_mark = os.getenv("INDIRECT_URL_MARK", "@")

    for i in range(1, len(sys.argv)-1, 2):
        src_uri = sys.argv[i]
        dest_path = sys.argv[i+1]

        LOG.info("Initializing, args: src_uri [%s] dest_path [%s]" %
                 (src_uri, dest_path))
        if dest_path.startswith(indirect_mark):
            indirect_download(src_uri, dest_path[len(indirect_mark):])
        else:
            download(src_uri, dest_path)


main()
