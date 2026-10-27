# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_stereo_vio_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED stereo_vio_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(stereo_vio_FOUND FALSE)
  elseif(NOT stereo_vio_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(stereo_vio_FOUND FALSE)
  endif()
  return()
endif()
set(_stereo_vio_CONFIG_INCLUDED TRUE)

# output package information
if(NOT stereo_vio_FIND_QUIETLY)
  message(STATUS "Found stereo_vio: 0.0.0 (${stereo_vio_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'stereo_vio' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${stereo_vio_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(stereo_vio_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${stereo_vio_DIR}/${_extra}")
endforeach()
