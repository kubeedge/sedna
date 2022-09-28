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

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Do not build in source directory!")
endif()

file(GLOB MOCKFLOW_SOURCE *.cpp *.cc *.c)

include_directories(${LIBMODELBOX_INCLUDE})
include_directories(${LIBMODELBOX_BASE_INCLUDE})
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${MODELBOX_SERVER_INCLUDE})

set(MOCKFLOW_LIB flowmock-lib)
add_library(${MOCKFLOW_LIB} ${MOCKFLOW_SOURCE})

set(MOCKFLOW_LIB ${MOCKFLOW_LIB} CACHE INTERNAL "")
set(MOCKFLOW_INCLUDE ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")


