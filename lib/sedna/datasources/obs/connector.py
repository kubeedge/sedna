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

import base64
from distutils import util
import json
import os
import random
import string
import time
from urllib import request
from sedna.common.log import LOGGER
from boto3.session import Session


class OBSClientWrapper():
    def __init__(
            self,
            file_server_url: string = '',
            vendor: string = "",
            region: string = "",
            bucket_name: string = "",
            app_token: string = ""
            ):

        # Constructs a obs client instance with your account for accessing OBS
        LOGGER.info(f"Create {self.__class__.__name__}")
        self.vendor = vendor
        self.region = region
        self.bucket_name = bucket_name
        self.app_token = app_token
        self.server = file_server_url

        LOGGER.info("Authenticating with the bucket \
             and setting up communication backend.")
        self._bucket_auth()
        self.s3_client_list = self._get_s3_client_list()

    # TODO: Create retry decorator.

    def _bucket_auth(self, retry=1):
        try:
            if self.bucket_name is None:
                raise Exception("Bucket name cannot be null!")

            if self.app_token is None:
                raise Exception("App token cannot be null!")

            bucket_auth_endpoint = f"{self.server}/rest/boto3/s3/\
                bucket-auth?vendor={self.vendor}&region={self.region}\
                    &bucketid={self.bucket_name}&apptoken={self.app_token}"
            self._do_request(bucket_auth_endpoint)

        except Exception as e:
            LOGGER.error(f"Error while connecting to bucket: {e}")
            if retry < 3:
                retry += 1
                time.sleep(retry*3)
                self._bucket_auth(retry)

        LOGGER.info("Bucket authentication successful")

    def _get_s3_client_list(self, retry=1):
        LOGGER.info("Get S3 backends.")
        s3_client_list = []
        try:
            query_urls_endpoint = f"{self.server}/rest/boto3/s3/query/\
                csb-file-server/all/ip-and-port?vendor={self.vendor}\
                    &region={self.region}"
            result_dict = self._do_request(query_urls_endpoint)

            if result_dict['fileServerUrlList'] is not None:
                for csb_file_server_url in result_dict['fileServerUrlList']:
                    if (csb_file_server_url is not None and
                            csb_file_server_url != '' and
                            csb_file_server_url != 'null'):
                        self.back_server_url = str(csb_file_server_url)

                    csb_obs_service_endpoint = f"{self.server}/rest/boto3/s3/\
                        {self.vendor}/{self.region}/{self.app_token}"
                    session = Session('Hello', 'CSB-OBS')
                    s3_client = session.client(
                        's3', endpoint_url=csb_obs_service_endpoint)
                    s3_client_list.append(s3_client)

            return s3_client_list

        except Exception as e:
            LOGGER.error(f"Error while getting S3 client list: {e}")
            if retry < 3:
                retry += 1
                time.sleep(retry*3)
                self._get_s3_client_list(retry)

    def download_single_object(
        self,
        remote_path,
        local_path=".",
        failed_count=1,
        selected_index=0
    ):

        try:
            wait_select_index_list = [
                index for index in range(len(self.s3_client_list))]
            if failed_count != 1 and len(self.s3_client_list) > 1:
                wait_select_index_list.remove(selected_index)

            random.shuffle(wait_select_index_list)
            selected_index = wait_select_index_list[0]
            s3_client = self.s3_client_list[selected_index]

            filename = os.path.basename(remote_path)

            if not os.path.exists(local_path):
                os.makedirs(local_path)

            objectKey_base64 = base64.urlsafe_b64encode(
                remote_path.encode(encoding="utf-8"))
            objectKey_base64 = str(objectKey_base64, encoding="utf-8")
            resp = s3_client.get_object(
                Bucket=self.bucket_name, Key=objectKey_base64)

            with open(os.path.join(local_path, filename), 'wb') as f:
                file = resp['Body']
                while True:
                    data = file.read(8192)
                    if data == b'':
                        break
                    f.write(data)
                    f.flush()

            LOGGER.info(f"File {filename} stored in {local_path}.")
            return filename

        except Exception as ex:
            if failed_count >= 3:
                LOGGER.error(f"Unable to download file {ex}")
                return None
            else:
                failed_count += 1
                time.sleep(failed_count)
                self.download_single_object(
                    remote_path, local_path=".",
                    failed_count=failed_count,
                    selected_index=selected_index)

    def list_objects(self, remote_path, next_marker=''):
        LOGGER.info(f"Listing objects in folder {remote_path}.")
        try:
            encoded_path = base64.urlsafe_b64encode(
                remote_path.encode(encoding="utf-8"))
            encoded_path = str(encoded_path, encoding="utf-8")

            list_objects_endpoint = f"{self.server}/rest/boto3/s3/list/\
                bucket/objectkeys?vendor={self.vendor}&region={self.region}\
                    &bucketid={self.bucket_name}&apptoken={self.app_token}\
                        &objectkey={encoded_path}&nextmarker={next_marker}"
            result_dict = self._do_request(list_objects_endpoint)

            if bool(util.strtobool(result_dict['success'])):
                for file in result_dict["objectKeys"]:
                    LOGGER.info(
                        f"Found {file['objectKey']} of size {file['size']}.")

        except Exception as ex:
            LOGGER.error(f"Unable to list objects {ex}")

    def upload_file(
            self,
            local_folder_absolute_path,
            filename,
            bucket_path,
            failed_count=1,
            selected_index=0):

        LOGGER.info(f"Uploading file {filename} (retry={failed_count}).")

        try:
            wait_select_index_list = [
                index for index in range(len(self.s3_client_list))]

            if failed_count != 1 and len(self.s3_client_list) > 1:
                wait_select_index_list.remove(selected_index)

            random.shuffle(wait_select_index_list)
            selected_index = wait_select_index_list[0]
            s3_client = self.s3_client_list[selected_index]

            key = bucket_path + filename
            key = base64.urlsafe_b64encode(key.encode(encoding="utf-8"))
            key = str(key, encoding="utf-8")

            with open(os.path.join(
                    local_folder_absolute_path, filename), 'rb') as file:
                _ = s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=file.read())
                LOGGER.info(f'File {filename} successfully uploaded.')

        except Exception as ex:
            LOGGER.error(f"Unable to upload file {filename}. Err: {ex}")

            if failed_count < 3:
                failed_count += 1
                time.sleep(failed_count)
                self.upload_file(
                    local_folder_absolute_path,
                    filename,
                    bucket_path,
                    failed_count=failed_count,
                    selected_index=selected_index)

    def _do_request(self, url):
        req = request.Request(url=url)
        res = request.urlopen(req, timeout=10)

        result = res.read().decode(encoding='utf-8')
        return json.loads(result)
