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


option(ENABLE_FRAME_POINTER "enable frame pointer on 64bit system with flag -fno-omit-frame-pointer, on 32bit system, it is always enabled" ON)

include (CheckFunctionExists)
check_function_exists(dladdr HAVE_DLADDR)
check_function_exists(nanosleep HAVE_NANOSLEEP)

option(STANDALONE "build standalone modelbox" OFF)
option(PYTHONE_DISABLED "Disable build python" OFF)
option(WITH_SECURE_C "include libsecurec.so" ON)
option(WITH_ALL_DEMO "build with all demo with large model file" OFF)
option(TEST_COVERAGE "build with coverage" OFF)
option(WITH_JAVA "build java support" OFF)
option(CLANG_TIDY "build with clang tidy" OFF)
option(CLANG_TIDY_FIX "do auto fix" ON)
option(CLANG_TIDY_AS_ERROR "make clang-tidy warning as error" OFF)
option(USE_CN_MIRROR "download from cn mirror" OFF)
option(WITH_WEBUI "build modelbox webui" ON)
option(WITH_MINDSPORE "build mindspore" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 


# speedup compilation
find_program(CCACHE ccache)
if(CCACHE)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    message(STATUS "Enable ccache")
endif(CCACHE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fno-strict-aliasing")

if (TEST_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif(TEST_COVERAGE)

if(ENABLE_FRAME_POINTER STREQUAL ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
endif(ENABLE_FRAME_POINTER STREQUAL ON) 

add_definitions(-D__STDC_FORMAT_MACROS)
add_definitions(-D_GNU_SOURCE)

if(OS_LINUX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--export-dynamic")
endif(OS_LINUX)

set(CUDA_NVCC_FLAGS "-Xcompiler -Wall,-fno-strict-aliasing,${CMAKE_CXX_FLAGS_DEBUG}" CACHE INTERNAL "") 
set(CUDA_PROPAGATE_HOST_FLAGS OFF CACHE INTERNAL "")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBASE_FILE_NAME='\"$(notdir $<)\"'")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBASE_FILE_NAME='\"$(notdir $<)\"'")
