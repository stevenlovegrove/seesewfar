project(SeeSewFar)
cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
find_package(Pangolin 0.6 REQUIRED)

find_package(Sophus REQUIRED)

add_executable(stitch src/main.cpp)
target_link_libraries(stitch pango_display pango_video Sophus::Sophus)
target_link_libraries(stitch "/opt/local/lib/libraw_r.dylib")
