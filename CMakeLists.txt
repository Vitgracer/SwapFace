cmake_minimum_required (VERSION 2.8.11)

if (WIN32)
    set(CMAKE_CONFIGURATION_TYPE Debug Release)
endif()

set(PROJECT_NAME SwapFace)
project(${PROJECT_NAME})
message("Project name: ${PROJECT_NAME}")

option(VIDEO_MODE "Add possibility to get frames from camera" OFF)
if(VIDEO_MODE)
    add_definitions(-DVIDEO_MODE)
endif()

option(TIMERS_ENABLE "Add possibility to obtain time profile" OFF)
if(TIMERS_ENABLE)
    add_definitions(-DTIMERS_ENABLE)
endif()

option(LOGGER_ENABLE "Add possibility to get log prints" OFF)
if(LOGGER_ENABLE)
    add_definitions(-DLOGGER_ENABLE)
endif()

if (WIN32)
    # TODO: add options for windows configuration
elseif(ANDROID)
    # TODO: add options for android configuration
endif()

find_package(OpenCV REQUIRED)

add_subdirectory(src)
add_subdirectory(launcher)
