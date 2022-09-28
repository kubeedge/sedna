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

project(example)

include(ExternalProject)

set(THIRDPARTY_DOWNLOAD_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/download)
set(THIRDPARTY_DOWNLOAD_WORKING_DIR ${THIRDPARTY_DOWNLOAD_BINARY_DIR}/build)

# 预先下载的三方组件，当使用ADD_SUBDIRECTORY包含子项目时，采用此方式。
if (NOT LOCAL_PACKAGE_PATH) 
  set(THIRDPARTY_DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/download)
  configure_file(CMake/pre-download.in ${THIRDPARTY_DOWNLOAD_BINARY_DIR}/CMakeLists.txt @ONLY)
else()
  set(THIRDPARTY_DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/download)
  configure_file(CMake/local-package.in ${THIRDPARTY_DOWNLOAD_BINARY_DIR}/CMakeLists.txt @ONLY)
endif()

file(MAKE_DIRECTORY ${THIRDPARTY_DOWNLOAD_WORKING_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" ..
  RESULT_VARIABLE COMMAND_RESULT
  WORKING_DIRECTORY ${THIRDPARTY_DOWNLOAD_WORKING_DIR} 
)

if(COMMAND_RESULT)
  message(FATAL_ERROR "Download thirdparty failed: ${COMMAND_RESULT}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE COMMAND_RESULT
  WORKING_DIRECTORY ${THIRDPARTY_DOWNLOAD_WORKING_DIR} 
)

if(COMMAND_RESULT)
  message(FATAL_ERROR "Download thirdparty failed: ${COMMAND_RESULT}")
endif()

if (${WITH_TEST})
  set(CMAKE_CXX_FLAGS_OLD ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  set(GOOGLETEST_SOURCE_DIR ${THIRDPARTY_DOWNLOAD_DIR}/googletest)
  add_subdirectory(${GOOGLETEST_SOURCE_DIR} ${THIRDPARTY_DOWNLOAD_WORKING_DIR}/googletest EXCLUDE_FROM_ALL)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_OLD})
endif()
