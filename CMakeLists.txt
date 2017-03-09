cmake_minimum_required(VERSION 3.6)
project(word2vec_master)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/compute-accuracy.c
        src/distance.c
        src/word-analogy.c
        src/word2phrase.c
        src/word2vec.c
        src/main.cpp)

add_executable(word2vec_master ${SOURCE_FILES})