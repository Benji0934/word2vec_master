SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = g++
#changed from gcc to g++ in the above line

#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

all: word2vec word2phrase distance word-analogy compute-accuracy WrapperW2V

word2vec : word2vec.c
	$(CC) word2vec.c -o ${BIN_DIR}/word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o ${BIN_DIR}/word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o ${BIN_DIR}/distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o ${BIN_DIR}/word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o ${BIN_DIR}/compute-accuracy $(CFLAGS)
	chmod +x ${SCRIPTS_DIR}/*.sh

#The lines below are added.
WrapperW2V : WrapperW2V.cpp WrapperW2V.h
	$(CC) -L/home/benjamin/Downloads/boost_1_63_0 -lboost_filesystem WrapperW2V.cpp -o ${BIN_DIR}/WrapperW2V $(CFLAGS)

clean:
	pushd ${BIN_DIR} && rm -rf word2vec word2phrase distance word-analogy compute-accuracy WrapperW2V; popd
