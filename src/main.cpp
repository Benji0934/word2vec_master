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


void creationOfSyn();

void createInverseMapping();

int testing() {
    int dims = WrapperW2V("../word2vecFiles/text8-vector.bin").getNumDimensions();
    cout << "tihih" + to_string(dims);
    //WrapperW2V wrapper = WrapperW2V("../word2vecFiles/text8-vector.bin");
    //   float f = wrapper.getWordVectors().front().;

    vector<float> vec = wrapper.getVectorForKnownWord("hi");
    boost::optional<vector<float>> optional = wrapper.getVectorForWord("hi");

    cout << "The 'hi' vector for knownWord" << endl;
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
    long long layer1_size = 200; //Amount of features/amount of weights in the NN.
    long *syn0;

// might have to find the maximum length for a string, instead of 10000.
    char output_file[10000];
    strcpy(output_file, "../word2vecFiles/classes.txt");
    FILE *fo;
    fo = fopen(output_file, "wb");



    // Run K-means on the word vectors.

    //Allocates memory for arrays.
    int clcn = amountOfClusters, iter = 10, closeid;
    int *centcn = (int *)malloc(amountOfClusters * sizeof(int));
    if (centcn == NULL) {
        fprintf(stderr, "cannot allocate memory for centcn\n");
        exit(1);
    }
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    float closev, x;
    float *cent = (float *)calloc(amountOfClusters * layer1_size, sizeof(float));

    //For every word we assign it to a cluster? So it is a random start.
    //cl is an array for all the words and their assigned cluster. cl[wordID]=clusterID
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;

    //10 iterations of Kmeans.
    for (a = 0; a < iter; a++) {
        //Setting the whole cent array to 0, for 10*100 indices.
        for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;

        //Setting the array centcn to 1 for indices 0-9. Centcn seems to be the amount of points in clusters?
        for (b = 0; b < clcn; b++) centcn[b] = 1;

        //For all words we set the cent[] to be the weights of the NN. That is, we store the weigths for the words into the cent array.
        //Also counts the amount of points in each cluster.
        //The syn0 can be read in from the text-8-vector.bin or text-8-vector.txt
        for (c = 0; c < vocab_size; c++) {
            for (d = 0; d < layer1_size; d++) {
                cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d]; //You could use a = sign instead of +=, right?
                centcn[cl[c]]++;
            }
        }

        //For all clusters. We find the centroids?
        for (b = 0; b < clcn; b++) {
            closev = 0;
            for (c = 0; c < layer1_size; c++) {
                cent[layer1_size * b + c] /= centcn[b]; //ASK: Why do we do this???
                closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
            }
            closev = sqrt(closev);
            for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
        }

        //For every word.
        for (c = 0; c < vocab_size; c++) {
            closev = -10;
            closeid = 0;
            //For every cluster
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
    //TODO: What I need is to get syn0[] out, since it contains all the weights in the NN, which is the features of the word vectors.

    //TODO: Need inverse mapping for getWords(): int->string

    /* TODO:
     * I need to create the struct vocab which is required to connect the words with their assigned clusters.
     * The problem is that the vocab has to be created with specific indices.
     * The fault I'm getting now, is a seg fault. Which means I try to access some memory that I should not.
     * Maybe I should create another data structure to keep the words and the position of it in.
     */
    //insertIntoVocab();
    for (a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    }
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
    cout << endl << "Testing the Word map" << endl;
    cout << "First value of 'and' lookup" + wrapper.getWords().find("and")->first << endl;
    cout << "Second val of 'and' lookup" + to_string(wrapper.getWords().find("and")->second) << endl;

    cout << endl << "Testing the InverseWord map" << endl;
    cout << "First value of '3' lookup" + to_string(wrapper.getInverseWords().find(3)->first) << endl;
    cout << "Second val of '3' lookup" + (wrapper.getInverseWords().find(3)->second) << endl;

    //cout << "looking up the first word in getWords" << endl;
    //int  a = wrapper.getWords().;

    cout << "looking up the word 1 in the wordVectors vector" << endl;
    vector<float> wordVec = wrapper.getWordVectors().at(2);

    for (auto it = wordVec.begin(); it!=wordVec.end(); ++it) {
        cout << to_string(*it.base()) + " ";
    }
    /*vector<float> wordVecs = wrapper.getWordVectors().at(6922);
    unordered_map<string, uint32_t> words = wrapper.getWords();
    //words.

     */

    //vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  //  insertIntoVocab();
  //  DestroyVocab();
}

int main() {
    cout << "Hello World!" << endl;
    creationOfSyn();
    //kMeans(10);
    //testing();
    test2();
    return 0;
}

void creationOfSyn() {

}

