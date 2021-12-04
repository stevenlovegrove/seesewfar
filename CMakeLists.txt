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
set(BUILD_TZ_LIB true)
set(LIBRAW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/LibRaw/")
add_subdirectory(LibRaw-cmake)
add_subdirectory(asyncplusplus)
add_subdirectory(date)
target_link_libraries(stitch libraw::libraw_r Async++ date date-tz)

target_include_directories(stitch PRIVATE "/Users/stevenlovegrove/local/include")

target_link_libraries(stitch "/Users/stevenlovegrove/local/lib/libjpl.a")
target_link_libraries(stitch "/Users/stevenlovegrove/local/lib/liblunar.a")

target_include_directories(stitch PRIVATE "/usr/local/include")
#target_link_libraries(stitch "/usr/local/lib/libcalceph.dylib")
target_link_libraries(stitch "/Users/stevenlovegrove/code/calceph-3.5.0/src/.libs/libcalceph.dylib")
