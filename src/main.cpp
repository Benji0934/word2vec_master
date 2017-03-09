//
// Created by benjamin on 3/9/17.
//

#include <iostream>

#include <string>
#include <cstring>
#include "WrapperW2V.h"

using namespace std;

WrapperW2V wrapper = WrapperW2V("../word2vecFiles/text8-vector.bin");
long vocab_size = wrapper.getWords().size();
long long vocab_max_size = 1000;

struct vocab_word {
    long long cn;
    int *point;
    char *word, *code, codelen;
};

struct vocab_word *vocab;


int testing() {
    int dims = WrapperW2V("../word2vecFiles/text8-vector.bin").getNumDimensions();
    cout << "tihih" + to_string(dims);
    //WrapperW2V wrapper = WrapperW2V("../word2vecFiles/text8-vector.bin");
    //   float f = wrapper.getWordVectors().front().;

    vector<float> vec = wrapper.getVectorForKnownWord("hi");
    boost::optional<vector<float>> optional = wrapper.getVectorForWord("hi");

    cout << "The 'hi' vector for knowWord" << endl;
    for (auto it = vec.begin(); it!=vec.end(); ++it) {
        cout << to_string(*it.base()) + " ";
    }
    cout << endl << "The 'hi' vector for getVectorForWord" << endl;
    for (auto it = optional.get().begin(); it!=optional.get().end(); ++it) {
        cout << to_string(*it.base()) + " ";
    }

    cout << endl << "Testing the Word map" << endl;
    cout << "First value of 'hi' lookup" + wrapper.getWords().find("hi")->first << endl;
    cout << "Second val of 'hi' lookup" + to_string(wrapper.getWords().find("hi")->second) << endl;

    cout << "looking up the word 6922 in the wordVectors vector" << endl;
    vector<float> wordVec = wrapper.getWordVectors().at(6922);
    for (auto it = wordVec.begin(); it!=wordVec.end(); ++it) {
        cout << to_string(*it.base()) + " ";
    }


    /*
    for (auto it = wrapper.getWordVectors().begin(); it != wrapper.getWordVectors().end(); ++it) {
        for (auto it1 = *it.base()->begin(); it1 != *it.base()->end(); ++it1) {
            cout << "Test" + to_string(it1) << endl;

        }
    }
*/  return 0;
}

void DestroyVocab() {
    int a;

    for (a = 0; a < vocab_size; a++) {
        if (vocab[a].word != NULL) {
            free(vocab[a].word);
        }
        if (vocab[a].code != NULL) {
            free(vocab[a].code);
        }
        if (vocab[a].point != NULL) {
            free(vocab[a].point);
        }
    }
    free(vocab[vocab_size].word);
    free(vocab);
}

int insertIntoVocab() {
    std::string str = "hej";
    cout << str << endl;
 //   std::copy(str.begin(), str.end(), vocab[1].word);
    string s = vocab[1].word;
    cout << s << vocab[1].word << endl << "hello" <<endl;

}
/*
 * Code taken from the word2vec.c file published by Mikolov.
 */
int kMeans(int amountOfClusters) {
    long a, b, c, d;
    long long layer1_size = 100;
    long *syn0;

// might have to find the maximum length for a string, instead of 10000.
    char output_file[10000];
    strcpy(output_file, "../word2vecFiles/classes.txt");
    FILE *fo;
    fo = fopen(output_file, "wb");



    // Run K-means on the word vectors
    int clcn = amountOfClusters, iter = 10, closeid;
    int *centcn = (int *)malloc(amountOfClusters * sizeof(int));
    if (centcn == NULL) {
        fprintf(stderr, "cannot allocate memory for centcn\n");
        exit(1);
    }
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    float closev, x;
    float *cent = (float *)calloc(amountOfClusters * layer1_size, sizeof(float));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
        for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
        for (b = 0; b < clcn; b++) centcn[b] = 1;
        for (c = 0; c < vocab_size; c++) {
            for (d = 0; d < layer1_size; d++) {
                cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
                centcn[cl[c]]++;
            }
        }
        for (b = 0; b < clcn; b++) {
            closev = 0;
            for (c = 0; c < layer1_size; c++) {
                cent[layer1_size * b + c] /= centcn[b];
                closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
            }
            closev = sqrt(closev);
            for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
        }
        for (c = 0; c < vocab_size; c++) {
            closev = -10;
            closeid = 0;
            for (d = 0; d < clcn; d++) {
                x = 0;
                for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
                if (x > closev) {
                    closev = x;
                    closeid = d;
                }
            }
            cl[c] = closeid;
        }
    }
    // Save the K-means classes


    /* TODO:
     * I need to create the struct vocab which is required to connect the words with their assigned clusters.
     * The problem is that the vocab has to be created with specific indices.
     * The fault I'm getting now, is a seg fault. Which means I try to access some memory that I should not.
     * Maybe I should create another data structure to keep the words and the position of it in.
     */
    //insertIntoVocab();
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
    DestroyVocab();
}

void test2() {
    cout << to_string(wrapper.getWords().size());
    for(auto it = wrapper.getWords().begin(); it != wrapper.getWords().end(); ++it) {
  //      cout << it->first + " " << it->second << endl;
    }
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    insertIntoVocab();
    DestroyVocab();
}

int main() {
    cout << "Hello World!" << endl;
    //testing();
    test2();
    return 0;
}

