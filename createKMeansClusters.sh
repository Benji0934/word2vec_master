#!/bin/bash

DATA_DIR=~/CLionProjects/wordClustering/vectors
BIN_DIR=~/CLionProjects/wordClustering/cmake-build-debug

VECTOR_FILE=$DATA_DIR/$1
AMOUNT_OF_CLUSTERS=$2


#time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1
$BIN_DIR/wordClustering $VECTOR_FILE $AMOUNT_OF_CLUSTERS
