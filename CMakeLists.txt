project(SeeSewFar)
cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
find_package(Pangolin 0.6 REQUIRED)

find_package(Sophus REQUIRED)

add_executable(stitch src/main.cpp)
target_link_libraries(stitch pango_display pango_video pango_plot Sophus::Sophus)

set(ENABLE_OPENMP false)
set(LIBRAW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/LibRaw/")
add_subdirectory(LibRaw-cmake)
add_subdirectory(asyncplusplus)
target_link_libraries(stitch libraw::libraw_r Async++)

#target_link_libraries(stitch "/opt/local/lib/libraw_r.dylib")
