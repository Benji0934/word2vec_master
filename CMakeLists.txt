cmake_minimum_required(VERSION 3.6)
project(wordClustering)

set(CMAKE_CXX_STANDARD 14)

set ( CMAKE_CXX_FLAGS "-lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result")
include_directories()

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS )


set(SOURCE_FILES
        src/WrapperW2V.cpp src/WrapperW2V.h
        src/main.cpp)

include_directories(${Boost_INCLUDE_DIRS})
add_executable(wordClustering ${SOURCE_FILES})
target_link_libraries(wordClustering ${Boost_LIBRARIES})