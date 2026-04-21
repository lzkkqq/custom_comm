cmake_minimum_required(VERSION 3.14)
project(custom_comm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# CMake here is ONLY for local syntax/static check on Mac.
# The real build is driven by setup.py (which globs sources itself).
# ---------------------------------------------------------------------------

include(cmake/FindCANN.cmake OPTIONAL)

# Glob all C++ sources under ops/ (matches setup.py convention).
file(GLOB_RECURSE CUSTOM_COMM_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cc")

add_library(custom_comm_ops SHARED ${CUSTOM_COMM_SOURCES})

target_include_directories(custom_comm_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/allgather_batch/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/allgather_batch/src
    ${CANN_INCLUDE_DIRS}
    ${CANN_PKG_INC_DIRS}
)

if(APPLE)
    target_link_options(custom_comm_ops PRIVATE -undefined dynamic_lookup)
endif()
