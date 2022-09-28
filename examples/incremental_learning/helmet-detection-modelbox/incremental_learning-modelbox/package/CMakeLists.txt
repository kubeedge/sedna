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
set(CPACK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
set(CPACK_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

function(FUNC_CPACK)
    set(CPACK_PACKAGE_NAME "modelbox-application" PARENT_SCOPE)
    set(CPACK_RPM_COMPONENT_INSTALL ON PARENT_SCOPE)
    set(CPACK_DEB_COMPONENT_INSTALL ON PARENT_SCOPE)
    set(CPACK_ARCHIVE_COMPONENT_INSTALL OFF PARENT_SCOPE)
    set(CPACK_SET_DESTDIR ON PARENT_SCOPE)
    set(CPACK_STRIP_FILES ON PARENT_SCOPE)
    find_program(TAR tar)
    find_program(RPM rpm)
    find_program(DPKG dpkg)
    set(MODELBOX_PACK_NAME "${CPACK_PACKAGE_NAME}")
    if(NOT CPACK_PACKAGE_NAME)
        set(MODELBOX_PACK_NAME "${CMAKE_PROJECT_NAME}")
    endif()
    if(TAR) 
        set(CPACK_GENERATOR "${CPACK_GENERATOR}TGZ;")
    endif()
    if(RPM) 
        set(CPACK_GENERATOR "${CPACK_GENERATOR}RPM;")
    endif()
    if(DPKG) 
        set(CPACK_GENERATOR "${CPACK_GENERATOR}DEB;")
    endif()    
    set(CPACK_GENERATOR "${CPACK_GENERATOR}" PARENT_SCOPE)
    
    string(TOUPPER ${UNIT_COMPONENT} UNIT_COMPONENT_UPPER_NAME)

    # deb package configuration
    set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON PARENT_SCOPE)
    set(CPACK_COMPONENT_${UNIT_COMPONENT_UPPER_NAME}_DESCRIPTION "${UNIT_COMPONENT} for modelbox" PARENT_SCOPE)
    configure_file(${CPACK_SOURCE_DIR}/debian/postinst.in ${CPACK_BINARY_DIR}/debian/postinst @ONLY)
    configure_file(${CPACK_SOURCE_DIR}/debian/postrm.in ${CPACK_BINARY_DIR}/debian/postrm @ONLY)
    set(CPACK_DEBIAN_${UNIT_COMPONENT_UPPER_NAME}_PACKAGE_CONTROL_EXTRA
        ${CPACK_BINARY_DIR}/debian/postinst
        ${CPACK_BINARY_DIR}/debian/postrm PARENT_SCOPE)
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "undefined." PARENT_SCOPE)

    # rpm package configuration
    set(CPACK_PACKAGE_RELOCATABLE OFF PARENT_SCOPE)
    set(CPACK_RPM_PACKAGE_AUTOREQ OFF PARENT_SCOPE)
    set(CPACK_RPM_SPEC_MORE_DEFINE "%define _build_id_links none" PARENT_SCOPE)
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
        "/usr/local"
        "/usr/local/bin"
        "/usr/local/lib"
        "/usr/local/lib64"
        "/usr/local/etc"
        "/usr/local/include"
        "/usr/lib/systemd"
        ${CMAKE_INSTALL_FULL_BINDIR}
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        ${CMAKE_INSTALL_FULL_LIBDIR}
        "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig"
        ${SYSTEMDSYSTEMUNITDIR} PARENT_SCOPEN)
    configure_file(${CPACK_SOURCE_DIR}/rpm/postinscript.in ${CPACK_BINARY_DIR}/rpm/postinscript @ONLY)
    configure_file(${CPACK_SOURCE_DIR}/rpm/postunscript.in ${CPACK_BINARY_DIR}/rpm/postunscript @ONLY)
    set(CPACK_RPM_${UNIT_COMPONENT_UPPER_NAME}_POST_INSTALL_SCRIPT_FILE ${CPACK_BINARY_DIR}/rpm/postinscript PARENT_SCOPE)
    set(CPACK_RPM_${UNIT_COMPONENT_UPPER_NAME}_POST_UNINSTALL_SCRIPT_FILE ${CPACK_BINARY_DIR}/rpm/postunscript PARENT_SCOPE)
    set(CPACK_OUTPUT_FILE_PREFIX ${CMAKE_BINARY_DIR}/release PARENT_SCOPE)
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR}/cpack PARENT_SCOPE)
    
    get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
    list(REMOVE_ITEM CPACK_COMPONENTS_ALL "Unspecified")
    set(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} PARENT_SCOPE)
endfunction(FUNC_CPACK)

set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})

include(CPackComponent)
FUNC_CPACK()
include(CPack)
