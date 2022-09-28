set(HITS_DRIVER_PATH $ENV{DRIVER_PATH})
find_path(DSMI_INCLUDE
  NAMES dsmi_common_interface.h
  HINTS ${CMAKE_INSTALL_FULL_INCLUDEDIR} 
        ${HITS_DRIVER_PATH}/include
        /usr/local/Ascend/driver/include
)
mark_as_advanced(DSMI_INCLUDE)

# Look for the library (sorted from most current/relevant entry to least).
set(DRIVER_LIB_PATH ${HITS_DRIVER_PATH}/lib64)
find_library(DSMI_LIBRARY NAMES
    drvdsmi_host
    HINTS ${CMAKE_INSTALL_FULL_LIBDIR} 
          ${DRIVER_LIB_PATH}
          ${DRIVER_LIB_PATH}/driver
          /usr/local/Ascend/driver/lib64
)
mark_as_advanced(DSMI_LIBRARY)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DSMI
                                  REQUIRED_VARS DSMI_LIBRARY DSMI_INCLUDE
                                  VERSION_VAR DSMI_VERSION_STRING)

if(DSMI_FOUND)
  set(DSMI_LIBRARIES ${DSMI_LIBRARY})
  set(DSMI_INCLUDE_DIR ${DSMI_INCLUDE})
endif()
