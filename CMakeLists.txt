cmake_minimum_required(VERSION 3.6)
project(word2vec_master)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        bin/compute-accuracy
        bin/distance
        bin/word-analogy
        bin/word2phrase
        bin/word2vec
        data/classes.sorted.txt
        data/classes.txt
        data/questions-phrases.txt
        data/questions-words.txt
        data/text8
        data/text8-vector.bin
        data/text8.zip
        scripts/demo-analogy.sh
        scripts/demo-classes.sh
        scripts/demo-phrase-accuracy.sh
        scripts/demo-phrases.sh
        scripts/demo-word-accuracy.sh
        scripts/demo-word.sh
        src/compute-accuracy.c
        src/distance.c
        src/word-analogy.c
        src/word2phrase.c
        src/word2vec.c
        LICENSE
        README.md)

add_executable(word2vec_master ${SOURCE_FILES})