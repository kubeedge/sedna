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

"""FileOps class."""
import os
import re
import codecs
import pickle
import shutil
import tempfile


class FileOps:
    """This is a class with some class methods to handle some files or folder."""
    _GCS_PREFIX = "gs://"
    _S3_PREFIX = "s3://"
    _LOCAL_PREFIX = "file://"
    _URI_RE = "https?://(.+)/(.+)"
    _HTTP_PREFIX = "http(s)://"
    _HEADERS_SUFFIX = "-headers"

    @classmethod
    def make_dir(cls, *args):
        """Make new a local directory.

        :param * args: list of str path to joined as a new directory to make.

        """
        _path = cls.join_path(*args)
        if not os.path.isdir(_path):
            os.makedirs(_path, exist_ok=True)

    @classmethod
    def clean_folder(cls, target, clean=True):
        """Make new a local directory.

        :param target: list of str path need to init.
        :type target: list
        :param clean: clear target if exists.
        :type clean: bool
        """
        if isinstance(target, str):
            target = [target]
        for path in set(target):
            args = str(path).split(os.path.sep)
            if len(args) < 2:
                continue
            _path = cls.join_path(*args)
            if os.path.isdir(_path) and clean:
                shutil.rmtree(_path)
            if os.path.isfile(_path):
                os.remove(_path)
                _path = cls.join_path(*args[:len(args) - 1])
            os.makedirs(_path, exist_ok=True)
        return target

    @classmethod
    def make_base_dir(cls, *args):
        """Make new a base directory.

        :param * args: list of str path to joined as a new base directory to make.

        """
        _file = cls.join_path(*args)
        if os.path.isfile(_file):
            return
        _path, _ = os.path.split(_file)
        if not os.path.isdir(_path):
            os.makedirs(_path, exist_ok=True)

    @classmethod
    def join_path(cls, *args):
        """Join list of path and return.

        :param * args: list of str path to be joined.
        :return: joined path str.
        :rtype: str

        """
        if len(args) == 1:
            return args[0]
        args = list(map(lambda x: x.lstrip(os.path.sep), args))

        # local path
        if ":" not in args[0]:
            args = tuple(args)
            return os.path.join(*args)
        # http or s3 path
        prefix = args[0]
        tail = os.path.join(*args[1:])
        return os.path.join(prefix, tail)

    @classmethod
    def remove_path_prefix(cls, org_str: str, prefix: str):
        """remove the prefix, for converting path in container to path in host."""
        p = prefix[:-1] if prefix.endswith(os.path.sep) else prefix
        if org_str.startswith(p):
            out_str = org_str.replace(p, '', 1)
            return out_str
        else:
            return org_str

    @classmethod
    def dump_pickle(cls, obj, filename):
        """Dump a object to a file using pickle.

        :param object obj: target object.
        :param str filename: target pickle file path.

        """
        if not os.path.isfile(filename):
            cls.make_base_dir(filename)
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    @classmethod
    def load_pickle(cls, filename):
        """Load a pickle file and return the object.

        :param str filename: target pickle file path.
        :return: return the loaded original object.
        :rtype: object or None.

        """
        if not os.path.isfile(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)

    @classmethod
    def copy_folder(cls, src, dst):
        """Copy a folder from source to destination.

        :param str src: source path.
        :param str dst: destination path.

        """
        if dst is None or dst == "" or (not os.path.isdir(src)):
            return
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
        else:
            if os.path.samefile(src, dst):
                return
            for files in os.listdir(src):
                name = os.path.join(src, files)
                back_name = os.path.join(dst, files)
                if os.path.isfile(name):
                    if os.path.isfile(back_name):
                        shutil.copy(name, back_name)
                else:
                    if not os.path.isdir(back_name):
                        shutil.copytree(name, back_name)
                    else:
                        cls.copy_folder(name, back_name)

    @classmethod
    def copy_file(cls, src, dst):
        """Copy a file from source to destination.

        :param str src: source path.
        :param str dst: destination path.

        """
        if dst is None or dst == "":
            return

        if os.path.isfile(src):
            if os.path.isfile(dst) and os.path.samefile(src, dst):
                return
            if os.path.isdir(dst):
                basename = os.path.basename(src)
                dst = os.path.join(basename, basename)
            parent_dir = os.path.dirname(dst)
            cls.clean_folder([parent_dir], clean=False)

            shutil.copy(src, dst)
        elif os.path.isdir(src):
            cls.clean_folder([dst], clean=False)
            cls.copy_folder(src, dst)

    @classmethod
    def download(cls, src, dst, unzip=False) -> str:
        if dst is None:
            dst = tempfile.mkdtemp()

        if src.startswith(cls._GCS_PREFIX):
            cls.gcs_download(src, dst)
        elif src.startswith(cls._S3_PREFIX):
            cls.s3_download(src, dst)
        elif cls.is_local:
            cls.copy_file(src, dst)
        elif re.search(cls._URI_RE, src):
            cls.http_download(src, dst)
        if unzip is True and dst.endswith(".tar.gz"):
            cls._untar(dst)
        return dst

    @classmethod
    def upload(cls, src, dst, tar=False) -> str:
        if dst is None:
            dst = tempfile.mkdtemp()
        if tar:
            cls._tar(src, f"{src}.tar.gz")
            src = f"{src}.tar.gz"
        if src.startswith(cls._GCS_PREFIX):
            cls.gcs_upload(src, dst)
        elif src.startswith(cls._S3_PREFIX):
            cls.s3_upload(src, dst)
        elif cls.is_local:
            cls.copy_file(src, dst)
        return dst

    @classmethod
    def is_local(cls, src):
        return src.startswith(cls._LOCAL_PREFIX) or cls.exists(src)

    @classmethod
    def gcs_download(cls, src, dst):
        """todo: not support now"""

    @classmethod
    def gcs_upload(cls, src, dst):
        """todo: not support now"""

    @classmethod
    def s3_download(cls, src, dst):
        import boto3
        from urllib.parse import urlparse
        s3 = boto3.resource('s3',
                            endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://s3.amazonaws.com"),
                            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
                            aws_secret_access_key=os.getenv("S3_SECRET_KEY")
                            )
        parsed = urlparse(src, scheme='s3')
        bucket_name = parsed.netloc
        bucket_path = parsed.path.lstrip('/')
        cls.clean_folder([dst], clean=False)
        bucket = s3.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=bucket_path):
            # Skip where boto3 lists the directory as an object
            if obj.key.endswith("/"):
                continue
            # In the case where bucket_path points to a single object, set the target key to bucket_path
            # Otherwise, remove the bucket_path prefix, strip any extra slashes, then prepend the target_dir
            target_key = (
                obj.key
                if bucket_path == obj.key
                else obj.key.replace(bucket_path, "", 1).lstrip("/")
            )
            target = f"{dst}/{target_key}"
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target), exist_ok=True)
            bucket.download_file(obj.key, target)

    @classmethod
    def s3_upload(cls, src, dst):
        import boto3
        from urllib.parse import urlparse
        s3 = boto3.resource('s3',
                            endpoint_url=os.getenv("S3_ENDPOINT", "http://s3.amazonaws.com"),
                            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
                            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY")
                            )
        parsed = urlparse(dst, scheme='s3')
        bucket_name = parsed.netloc
        bucket = s3.Bucket(bucket_name)
        if os.path.isdir(src):
            for root, _, files in os.walk(src):
                for file in files:
                    filepath = os.path.join(root, file)
                    with open(filepath, 'rb') as data:
                        bucket.put_object(Key=file, Body=data)
        elif os.path.isfile(src):
            with open(src, 'rb') as data:
                bucket.put_object(Key=os.path.basename(src), Body=data)

    @classmethod
    def http_download(cls, src, dst):
        """Download data from http or https web site.

        :param src: the data path
        :type src: str
        :param dst: the data path
        :type dst: str
        :raises FileNotFoundError: if the file path is not exist, an error will raise
        """
        from six.moves import urllib
        import fcntl

        signal_file = cls.join_path(os.path.dirname(dst), ".{}.signal".format(os.path.basename(dst)))
        if not os.path.isfile(signal_file):
            with open(signal_file, 'w') as fp:
                fp.write('{}'.format(0))

        with open(signal_file, 'r+') as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            signal = int(fp.readline().strip())
            if signal == 0:
                try:
                    urllib.request.urlretrieve(src, dst)
                except (urllib.error.URLError, IOError) as e:
                    raise e

                with open(signal_file, 'w') as fn:
                    fn.write('{}'.format(1))
            fcntl.flock(fp, fcntl.LOCK_UN)

    @classmethod
    def _untar(cls, src, dst=None):
        import tarfile
        if dst is None:
            dst = os.path.dirname(src)
        with tarfile.open(src, 'r:gz') as tar:
            tar.extractall(path=dst)

    @classmethod
    def _tar(cls, src, dst):
        import tarfile
        with tarfile.open(dst, 'w:gz') as tar:
            if os.path.isdir(src):
                for root, _, files in os.walk(src):
                    for file in files:
                        filepath = os.path.join(root, file)
                        tar.add(filepath)
            elif os.path.isfile(src):
                tar.add(os.path.realpath(src))

    @classmethod
    def exists(cls, folder):
        """Is folder existed or not.

        :param folder: folder
        :type folder: str
        :return: folder existed or not.
        :rtype: bool
        """
        return os.path.isdir(folder) or os.path.isfile(folder)

    @classmethod
    def obj_to_pickle_string(cls, x):
        return codecs.encode(pickle.dumps(x), "base64").decode()

    @classmethod
    def pickle_string_to_obj(cls, s):
        return pickle.loads(codecs.decode(s.encode(), "base64"))
