cmake_minimum_required(VERSION 3.6)
project(word2vec)

set (MY_SOURCE_DIR "/home/benjamin/Downloads/boost_1_63_0")

include_directories( ${MY_SOURCE_DIR}/src )

set(CMAKE_CXX_STANDARD 11)


set(SOURCE_FILES
        compute-accuracy.c
        distance.c
        makefile
        word-analogy.c
        word2phrase.c
        word2vec.c)

set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CXX )

#add_executable(word2vec ${SOURCE_FILES})