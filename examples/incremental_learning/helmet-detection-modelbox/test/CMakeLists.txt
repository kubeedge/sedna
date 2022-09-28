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

if(NOT TEST_WORKING_DIR)
	set(TEST_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}/test-working-dir)
	file(MAKE_DIRECTORY ${TEST_WORKING_DIR})
endif()
set(TEST_WORKING_DATA_DIR "${TEST_WORKING_DIR}/data")
file(MAKE_DIRECTORY ${TEST_WORKING_DATA_DIR})
set(TEST_WORKING_LIB_DIR "${TEST_WORKING_DIR}/lib")
file(MAKE_DIRECTORY ${TEST_WORKING_LIB_DIR})
set(TEST_WORKING_BIN_DIR "${TEST_WORKING_DIR}/bin")
file(MAKE_DIRECTORY ${TEST_WORKING_BIN_DIR})
set(TEST_WORKING_MODEL_DIR "${TEST_WORKING_DIR}/model")
file(MAKE_DIRECTORY ${TEST_WORKING_MODEL_DIR})
set(TEST_WORKING_PYTHON_DIR "${TEST_WORKING_DIR}/python")
file(MAKE_DIRECTORY ${TEST_WORKING_PYTHON_DIR})
set(TEST_ASSETS ${CMAKE_CURRENT_LIST_DIR}/assets)
set(TEST_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

add_definitions(-DBUILD_TEST)
set(TEST_MAIN_SOURCE ${CMAKE_CURRENT_LIST_DIR}/test_main.cc)
configure_file(${CMAKE_CURRENT_LIST_DIR}/test_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/test_config.h @ONLY)

list(APPEND TEST_INCLUDE ${LIBMODELBOX_INCLUDE})
list(APPEND TEST_INCLUDE ${MODELBOX_SERVER_INCLUDE})
list(APPEND TEST_INCLUDE ${LIBMODELBOX_BASE_INCLUDE})
list(APPEND TEST_INCLUDE ${TOML_INCLUDE_DIR})
list(APPEND TEST_INCLUDE ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND TEST_INCLUDE ${MODELBOX_TOP_DIR})
list(REMOVE_DUPLICATES TEST_INCLUDE)

set(TEST_SOURCE 
    ${MODELBOX_SERVER_SOURCES} 
)

set(TEST_LINK_LIBRARIES
    ${MODELBOX_SERVER_LINK_LIBRARIES}
	${LIBMODELBOX_LINK_SOURCES} 
)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${gmock_SOURCE_DIR}/include ${gmock_SOURCE_DIR})
include_directories(${MODELBOX_SERVER_INCLUDE})
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS_OLD ${CMAKE_CXX_FLAGS})
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-gnu-unique")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()
add_subdirectory(mock)
add_subdirectory(flowunit)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_OLD})