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
 
file(GLOB_RECURSE UNIT_TEST_SOURCE *.cpp *.cc *.c)
 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MOCKFLOW_INCLUDE})
include_directories(${TEST_INCLUDE})

add_executable(unit-test EXCLUDE_FROM_ALL
	${UNIT_TEST_SOURCE}
	${TEST_MAIN_SOURCE}
)

add_custom_target(all-drivers)
add_custom_command(TARGET all-drivers PRE_BUILD
	COMMAND rm -fr ${TEST_WORKING_LIB_DIR}/*
)

add_custom_command(TARGET all-drivers PRE_BUILD
	COMMAND rm -fr ${TEST_WORKING_LIB_DIR}/*
)

foreach (ITR ${UNIT_TEST_TARGET})
	add_dependencies(all-drivers ${ITR})
	add_custom_command(TARGET all-drivers POST_BUILD
	COMMAND cp $<TARGET_FILE:${ITR}> ${TEST_WORKING_LIB_DIR}/
	)
endforeach(ITR) 
set(UNIT_TEST_TARGET "" CACHE INTERNAL "")

add_custom_command(TARGET all-drivers POST_BUILD
	COMMAND ldconfig ${TEST_WORKING_LIB_DIR} -n
)

foreach (ITR ${DRIVER_UNIT_TEST_LINK_LIBRARIES})
	target_link_libraries(unit-test ${ITR})
endforeach(ITR) 
set(DRIVER_UNIT_TEST_LINK_LIBRARIES "" CACHE INTERNAL "")

target_link_libraries(unit-test pthread)
target_link_libraries(unit-test rt)
target_link_libraries(unit-test dl)
target_link_libraries(unit-test gtest_main)
target_link_libraries(unit-test gmock_main)
target_link_libraries(unit-test modelbox)
target_link_libraries(unit-test ${OpenCV_LIBS})
target_link_libraries(unit-test ${MOCKFLOW_LIB})
add_dependencies(unit-test all-drivers)

add_custom_target(unittest
	COMMAND ${TEST_RUNNER_LIST} ${CMAKE_CURRENT_BINARY_DIR}/unit-test
	DEPENDS  unit-test
	WORKING_DIRECTORY ${TEST_WORKING_DIR}
	COMMENT "Run Unit Test..."
)

list(APPEND MODELBOX_UNIT_TEST_TARGETS unit-test)
set(MODELBOX_UNIT_TEST_TARGETS ${MODELBOX_UNIT_TEST_TARGETS} CACHE INTERNAL "")

 