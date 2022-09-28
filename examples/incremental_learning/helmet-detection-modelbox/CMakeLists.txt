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

project(helm_sedna)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "cannot build the project in the source directory! Out-of-source build is enforced!")
endif()

option(WITH_TEST "build with test" OFF)

set(MODELBOX_PROJECT_VERSION_MAJOR 1)
set(MODELBOX_PROJECT_VERSION_MINOR 0)
set(MODELBOX_PROJECT_VERSION_PATCH 0)

set(CMAKE_PROJECT_VERSION ${MODELBOX_PROJECT_VERSION_MAJOR})
set(CMAKE_PROJECT_VERSION_MAJOR ${MODELBOX_PROJECT_VERSION_MAJOR})
set(CMAKE_PROJECT_VERSION_MINOR ${MODELBOX_PROJECT_VERSION_MINOR})
set(CMAKE_PROJECT_VERSION_PATCH ${MODELBOX_PROJECT_VERSION_PATCH})

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

include(Options)
include(Function)
include(FindPkgConfig)
include(CMakeDependentOption)
include(GNUInstallDirs)

set(UNIT_COMPONENT ${CMAKE_PROJECT_NAME})
set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})

# User defined installation path
set(RELEASE_PACKAGE_DIR_ROOT /opt/modelbox/application/${CMAKE_PROJECT_NAME})
set(RELEASE_PACKAGE_DIR_LIB ${RELEASE_PACKAGE_DIR_ROOT}/flowunit) # c++ flowunit so path
set(RELEASE_PACKAGE_DIR_PYTHON ${RELEASE_PACKAGE_DIR_ROOT}/flowunit) # python flowunit path
set(RELEASE_PACKAGE_DIR_MODEL ${RELEASE_PACKAGE_DIR_ROOT}/flowunit) # inference flowunit path
set(RELEASE_PACKAGE_DIR_GRAPH ${RELEASE_PACKAGE_DIR_ROOT}/graph) # graph toml path
set(RELEASE_PACKAGE_DIR_ETC ${RELEASE_PACKAGE_DIR_ROOT}/etc) 
set(RELEASE_PACKAGE_DIR_BIN ${RELEASE_PACKAGE_DIR_ROOT}/bin) 
set(RELEASE_PACKAGE_DIR_THIRDPARTY ${RELEASE_PACKAGE_DIR_ROOT}/thirdparty)

find_package(CUDA 10.0)
find_package(ACL)
find_package(DSMI)
find_package(OpenCV)

add_subdirectory(src)
add_subdirectory(thirdparty)
add_subdirectory(package)
add_subdirectory(test EXCLUDE_FROM_ALL)
