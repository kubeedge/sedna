#
# Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10)
 
set(UNIT_NAME "yolo3_post")

file(GLOB_RECURSE UNIT_SRC *.*)
file(GLOB_RECURSE CMAKELISTS_FILE "CMakeLists.txt")
list(REMOVE_ITEM UNIT_SRC ${CMAKELISTS_FILE})
group_files(UNIT_SRC UNIT_CONF .*.toml ${UNIT_SRC})
list(APPEND UNIT_TOML_JSON ${UNIT_CONF})
group_files(UNIT_SRC UNIT_CONF .*.json ${UNIT_SRC})
list(APPEND UNIT_TOML_JSON ${UNIT_CONF})

install(FILES ${UNIT_SRC}
        COMPONENT ${UNIT_COMPONENT}
        DESTINATION ${RELEASE_PACKAGE_DIR_MODEL}/${UNIT_NAME})

install(FILES ${UNIT_TOML_JSON}
        COMPONENT ${UNIT_COMPONENT}
        PERMISSIONS OWNER_READ
        DESTINATION ${RELEASE_PACKAGE_DIR_MODEL}/${UNIT_NAME})
