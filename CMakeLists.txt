cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchd)

find_package(Threads REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(ZLIB QUIET)
if (ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPPHTTPLIB_ZLIB_SUPPORT")
endif()

set(SOURCE_FILES src/torchd.cpp src/httplib.cpp src/simdjson.cpp)

add_executable(torchd ${SOURCE_FILES})
target_link_libraries(torchd "${TORCH_LIBRARIES}" ZLIB::ZLIB)
set_property(TARGET torchd PROPERTY CXX_STANDARD 14)
