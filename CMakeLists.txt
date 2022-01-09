cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchd)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCE_FILES src/torchd.cpp src/httplib.cpp src/simdjson.cpp)

add_executable(torchd ${SOURCE_FILES})
target_link_libraries(torchd "${TORCH_LIBRARIES}")
set_property(TARGET torchd PROPERTY CXX_STANDARD 14)
