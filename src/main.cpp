//
// Created by benjamin on 3/9/17.
//

#include <iostream>

#include <string>
#include <cstring>
#include <fstream>
#include "WrapperW2V.h"
#include <chrono>
#include <sstream>
#include <string>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <math.h>
#include <cfloat>


using namespace std;
using namespace std::chrono;

WrapperW2V wrapper = WrapperW2V("../word2vecFiles/text8-250kb-vector.bin");
long vocab_size = wrapper.getWords().size();
long long dims = wrapper.getNumDimensions();

double maxVal = DBL_MAX;
vector<int> usedClusters;

int testing() {
    long dims = WrapperW2V("../word2vecFiles/text8-vector.bin").getNumDimensions();
    cout << "tihih" + to_string(dims);
    //WrapperW2V wrapper = WrapperW2V("../word2vecFiles/text8-vector.bin");
    //   float f = wrapper.getWordVectors().front().;

    vector<float> vec = wrapper.getVectorForKnownWord("hi");
    boost::optional<vector<float>> optional = wrapper.getVectorForWord("hi");

    cout << "The 'hi' vector for knownWord" << endl;
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        cout << to_string(*it.base()) + " ";
    }
    cout << endl << "The 'hi' vector for getVectorForWord" << endl;
    for (auto it = optional.get().begin(); it != optional.get().end(); ++it) {
        cout << to_string(*it.base()) + " ";
    }

    cout << endl << "Testing the Word map" << endl;
    cout << "First value of 'hi' lookup" + wrapper.getWords().find("hi")->first << endl;
    cout << "Second val of 'hi' lookup" + to_string(wrapper.getWords().find("hi")->second) << endl;

    cout << "looking up the word 6922 in the wordVectors vector" << endl;
    vector<float> wordVec = wrapper.getWordVectors().at(6922);
    for (auto it = wordVec.begin(); it != wordVec.end(); ++it) {
        cout << to_string(*it.base()) + " ";
    }


    /*
    for (auto it = wrapper.getWordVectors().begin(); it != wrapper.getWordVectors().end(); ++it) {
        for (auto it1 = *it.base()->begin(); it1 != *it.base()->end(); ++it1) {
            cout << "Test" + to_string(it1) << endl;

        }
    }
*/
    return 0;

}

/*
 * Code taken from the word2vec.c file published by Mikolov.
 */
void kMeans(int amountOfClusters) {
    milliseconds startTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    milliseconds endTime;
    long a, b, c, d;
    long long layer1_size = wrapper.getNumDimensions(); //Amount of features/amount of weights in the NN. 200 originally
    float weight = 0;
    vector<vector<float>> wordVectors;
    wordVectors = wrapper.getWordVectors();
    vector<float> wordVec;

// might have to find the maximum length for a string, instead of 10000.
    char output_file[10000];
    strcpy(output_file, "../word2vecFiles/classes.txt");
    // FILE *fo;
    // fo = fopen(output_file, "wb");

    // Run K-means on the word vectors.
    //Allocates memory for arrays.
    int clcn = amountOfClusters, iter = 10, closeid;
    int *centcn = (int *) malloc(amountOfClusters * sizeof(int));
    if (centcn == NULL) {
        fprintf(stderr, "cannot allocate memory for centcn\n");
        exit(1);
    }
    int *cl = (int *) calloc(vocab_size, sizeof(int));
    float closev, x;
    float *cent = (float *) calloc(amountOfClusters * layer1_size, sizeof(float));

    //For every word we assign it to a cluster? So it is a random start.
    //cl is an array for all the words and their assigned cluster. cl[wordID]=clusterID
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;

    //10 iterations of Kmeans.
    for (a = 0; a < iter; a++) {
        cout << to_string(a) << "th iter." << endl;
        //Setting the whole cent array to 0, for 10*100 indices.
        for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;

        //Setting the array centcn to 1 for indices 0-9. Centcn seems to be the amount of points in clusters?
        for (b = 0; b < clcn; b++) centcn[b] = 1;

        //For all words we set the cent[] to be the weights of the NN. That is, we store the weigths for the words into the cent array.
        //Also counts the amount of points in each cluster.
        //The syn0 can be read in from the text-8-vector.bin or text-8-vector.txt
        // C is iterating over the words. and d is iterating over the weights/features in the vector.
        //cout << "First time running through the weights of the vectors." << endl;
        for (c = 0; c < vocab_size; c++) {
            for (d = 0; d < layer1_size; d++) {
                //cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d]; //Original line
                //Finding the wordVector corresponding to the c'th index:
                wordVec = wordVectors.at(c);
                //cout << to_string(wordVec.at(d)) << endl;
                //Getting the weight corresponding to the d'th index in the c'th vector.
                weight = wordVec.at(d);
                cent[layer1_size * cl[c] + d] += weight; //You could use a = sign instead of +=, right?
                centcn[cl[c]]++;
            }
        }
        //cout << "Finished running through the weights of the vectors the first time." << endl;

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
        // C is iterating over the words. and b is iterating over the weights/features in the vector.
        //Replace these with iterations over the
        //cout << "second time running through the word vectors." << endl;
        for (c = 0; c < vocab_size; c++) {
            closev = -10;
            closeid = 0;

            if (c == 10000) {
                cout << "Reached 10k words" << endl;
            }
            if (c == 20000) {
                cout << "Reached 20k words" << endl;
            }
            if (c == 50000) {
                cout << "Reached 50k words" << endl;
            }
            //For every cluster
            for (d = 0; d < clcn; d++) {
                x = 0;
                for (b = 0; b < layer1_size; b++) {
                    //x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
                    wordVec = wordVectors.at(c);
                    //Getting the weight corresponding to the d'th index in the c'th vector.
                    weight = wordVec.at(b);
                    x += cent[layer1_size * d + b] * weight;
                }
                if (x > closev) {
                    closev = x;
                    closeid = d;
                }
            }
            cl[c] = closeid;
        }
        //cout << "FINISHED: second time running through the word vectors." << endl;

    }
    // Save the K-means classes
    ofstream myfile;
    myfile.open("../classes.txt");
    cout << "writing to file" << endl;
    //std::ofstream log("example.txt", std::ios_base::app | std::ios_base::out);

    //Appending a lot of zeros to have label to look for.
    for (a = 0; a < vocab_size; a++) {

        /* if (cl[a]<10) {
             string val = "0" + to_string(cl[a]);
             //myfile << wrapper.getInverseWords().find(a)->second + " " + val + "\n";
             myfile << wrapper.getInverseWords().find(a)->second + " " + val + "\n";
             // cout << wrapper.getInverseWords().find(a)->second << cl[a] << endl;
             //log << wrapper.getInverseWords().find(a)->second + " " + to_string(cl[a]) + "\n";
         } else {
             myfile << wrapper.getInverseWords().find(a)->second + " " + to_string(cl[a]) + "\n";

         }*/
        myfile << wrapper.getInverseWords().find(a)->second + " " + to_string(cl[a]) + "\n";
    }
    endTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    milliseconds runTime = endTime - startTime;
    cout << "runTime: " << runTime.count() << endl;

    //Create avgs.


    //Cleaning up.
    myfile.close();
    free(centcn);
    free(cent);
    free(cl);
}

void test2() {
    cout << "amount of words: " << to_string(wrapper.getWords().size());
    for (auto it = wrapper.getWords().begin(); it != wrapper.getWords().end(); ++it) {
        //      cout << it->first + " " << it->second << endl;
    }
    cout << endl << "Testing the Word map find" << endl;
    cout << "First value of 'and' lookup" + wrapper.getWords().find("and")->first << endl;
    cout << "Second val of 'and' lookup" + to_string(wrapper.getWords().find("and")->second) << endl;

    cout << endl << "Testing the InverseWord map find" << endl;
    cout << "First value of '3' lookup" + to_string(wrapper.getInverseWords().find(3)->first) << endl;
    cout << "Second val of '3' lookup" + (wrapper.getInverseWords().find(3)->second) << endl;

    //cout << "looking up the first word in getWords" << endl;
    //int  a = wrapper.getWords().;

    cout << "Testing WordVectors at position 3" << endl;
    vector<float> wordVec = wrapper.getWordVectors().at(3);
    for (auto it = wordVec.begin(); it != wordVec.end(); ++it) {
        cout << to_string(*it.base()) + " ";
    }
    cout << endl << "Front: " << wordVec.front() << endl;
    cout << "Back: " << wordVec.back() << endl;
    cout << "Pos1: " << wordVec.at(1) << endl;

    cout << "Vector for word 'and'" << endl;
    boost::optional<vector<float>> optVec = wrapper.getVectorForWord("and");
    vector<float> andVec = (vector<float> &&) optVec.get();
    for (auto it = andVec.begin(); it != andVec.end(); ++it) {
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

void test3() {
    cout << wrapper.getNumDimensions() << endl;
    cout << vocab_size << endl;
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

double getDistanceBetweenVectors(vector<double> v1, vector<double> v2) {

    double dot = 0, v1Length = 0, v2Length = 0, v1val = 0, v2val = 0;
    for (unsigned int i = 0; i < dims; i++) {
        v1val = v1.at(i);
        v2val = v2.at(i);
        dot = v1val * v2val + dot;
        v1Length = v1val * v1val + v1Length;
        v2Length = v2val * v2val + v2Length;
    }
    v1Length = sqrt(v1Length);
    v2Length = sqrt(v2Length);
    //cout << "v1 length: " << v1Length << endl;
    //cout << "v2 length: " << v2Length << endl;
    double cosSim = dot / (v1Length * v2Length);
    //cout << "cosSim: " << cosSim << endl;

    return 1 - cosSim;
    //cos(d1, d2) = 1-(d1 ⋅ d2) / ||d1||*||d2||

}

void hierarchicalClustering(int amountOfClusters) {
    //0. Load numbers
    int clusterCnt = amountOfClusters;
    string clusterCntString = to_string(amountOfClusters);
    vector<vector<double>> means;
    vector<vector<double>> distances;
    vector<int> vectorsInClusters;
    for (int i = 0; i < amountOfClusters; i++) {
        vectorsInClusters.push_back(0);
    }
    //Creating means
    vector<double> temp;
    for (int i = 0; i < dims; i++) {
        temp.push_back(0);
    }
    for (int i = 0; i < amountOfClusters; i++) {
        means.push_back(temp);
    }

    //Creating distance
    temp.clear();
    for (int i = 0; i < amountOfClusters; i++) {
        temp.push_back(0);
    }
    for (int i = 0; i < amountOfClusters; i++) {
        distances.push_back(temp);
    }

    //1. Create means (mean-points)
    cout << "Calculating initial means..." << endl;

    std::ifstream infile("../classes.txt");
    //int vectorsInClusters = 0;
    std::string line;
    while (std::getline(infile, line)) {
        //std::istringstream iss(line);
        string word;
        int clusterID;
        vector<string> stringAndPos = split(line, ' ');
        word = stringAndPos.front();
        clusterID = stoi(stringAndPos.back());
        vector<float> wordVector = wrapper.getVectorForKnownWord(word);
        vectorsInClusters[clusterID]++;
        for (unsigned int i = 0; i < dims; i++) {
            //means[i][clusterID] = (means[i][clusterID] + wordVector.at(i))/vectorsInClusters[clusterID];
            means[clusterID][i] = (means[clusterID][i] + wordVector.at(i)) / vectorsInClusters[clusterID];
            //cout << wordVector.at(i) << endl;
            //cout << means[i][clusterID] << endl;
            //cout << clusterID << endl;
            //cout << vectorsInClusters[clusterID] << endl;
        }
    }
    //2. Measure cosine distance between means
    //Creating vectors

    for (int k = 0; k < amountOfClusters; k++) {
        vector<double> v1;
        vector<double> v2;
        v1 = means[k];
        for (unsigned int j = 0; j < amountOfClusters; j++) {
            v2 = means[j];
            if (k == j) {
                distances[k][j] = 0;
            } else {
                distances[k][j] = getDistanceBetweenVectors(v1, v2);
                if (distances[k][j] < 0) {
                    cout << "ERROR: " << distances[k][j] << endl;
                    cout << k << j << endl;
                }
            }
            v2.clear();
        }
        v1.clear();
    }
    cout << "Finding shortest distance between clusters..." << endl;
    //3. Merge the two closest clusters
    Step3:
    double shortestDist = std::numeric_limits<double>::max();
    int iVal = 0;
    int jVal = 0;
    for (unsigned int i = 0; i < clusterCnt; i++) {
        for (unsigned int j = 0; j < clusterCnt; j++) {
            if (!i == j) {
                double currentDist = distances[i][j];
                //cout << "Distance found: " << currentDist << endl;
                if (currentDist < shortestDist) {
                    //     cout << "New shortest found: " << i << j << endl;
                    iVal = i;
                    jVal = j;
                    shortestDist = currentDist;
                }
            }
        }
    }
    string iString = to_string(iVal);
    string jString = to_string(jVal);
    cout << "Adding numbers to usedClusters..." << endl;
    //Adding clusters that have been merged.
    usedClusters.push_back(iVal);
    usedClusters.push_back(jVal);
    cout << "Merging..." << endl;
    //Creating copy of classes
    std::ifstream mergedClassesReadFile("../mergedClasses.txt");
    std::string mergedClassesStr;
    ofstream oldClassesWriteFile;
    oldClassesWriteFile.open("../oldClasses.txt");

    while (std::getline(mergedClassesReadFile, mergedClassesStr)) {
        oldClassesWriteFile << mergedClassesStr + "\n";
    }
    mergedClassesReadFile.close();
    oldClassesWriteFile.close();

    ofstream mergedClassesWriteFile;
    mergedClassesWriteFile.open("../mergedClasses.txt");

    std::ifstream oldClassesReadFile("../oldClasses.txt");
    std::string str;

    while (std::getline(oldClassesReadFile, str)) {
        str = str + "\n";
        std::size_t foundJ = str.find(jString + "\n");
        std::size_t foundI = str.find(iString + "\n");
        if (foundJ != std::string::npos) {
            string newString = str.substr(0, str.find(jString));
            newString = newString + clusterCntString + "\n";
            mergedClassesWriteFile << newString;
        } else if (foundI != std::string::npos) {
            string newString = str.substr(0, str.find(iString));
            newString = newString + clusterCntString + "\n";
            mergedClassesWriteFile << newString;
        } else {
            mergedClassesWriteFile << str;
        }
    }
    mergedClassesWriteFile.close();
    oldClassesReadFile.close();

    cout << "Creating tree and hierarchy files..." << endl;
    //Writing tree
    //Format: label,parent
    ofstream treeFile;
    treeFile.open("../tree.txt",std::ios_base::app);
    treeFile << iString << "," << clusterCntString << "\n";
    treeFile << jString << "," << clusterCntString << "\n";
    treeFile.close();
    //Writing hierarchy
    //Format: level,label<0>,label<1>,...,label<amountOfWords>
    ofstream hierarchyFile;
    hierarchyFile.open("../hierarchy.txt",std::ios_base::app);
    //treeFile << iString << "," << jString << "\n";
    hierarchyFile.close();
    clusterCnt++;
    clusterCntString = to_string(clusterCnt);

    //4. Recalculate mean/avg for the new cluster
    cout << "Recalculating mean..." << endl;
    //double newMean;
    vector<double> iVec;
    vector<double> jVec;
    iVec = means[iVal];
    jVec = means[jVal];
    vector<double> newMean;
    for (int i = 0; i < dims; i++) {
        double newMeanVal = (iVec[i] + jVec[i]) / 2;
        newMean.push_back(newMeanVal);
    }
    means.push_back(newMean);

    //5. Recalculate the distance from the new cluster to the existing ones
    cout << "Recalculating distances..." << endl;
    //Copy distances..
    vector<vector<double>> newDistances;
    vector<double> newDistVec;
    for (int i = 0; i < clusterCnt; i++) {
        newDistVec.push_back(maxVal);
    }
    for (int l = 0; l < clusterCnt; ++l) {
        newDistances.push_back(newDistVec);
    }
    for (int i = 0; i < clusterCnt - 1; i++) {
        for (int j = 0; j < clusterCnt - 1; j++) {
            if ((std::find(usedClusters.begin(), usedClusters.end(), j) == usedClusters.end()) ||
                    (std::find(usedClusters.begin(), usedClusters.end(), i) == usedClusters.end())) {
                newDistances[i][j] = distances[i][j];
            } else {
                newDistances[i][j] = maxVal;
            }
        }
    }
    cout << "Calculating distances to new cluster..." << endl;
    //Calc new distances
    vector<double> v1;
    vector<double> v2;
    v1 = means[clusterCnt - 1];
    newDistances[clusterCnt-1][clusterCnt-1] = 0;
    for (unsigned int j = 0; j < clusterCnt - 1; j++) {
        v2 = means[j];
        if (std::find(usedClusters.begin(), usedClusters.end(), j) == usedClusters.end()) {
            newDistances[clusterCnt-1][j] = getDistanceBetweenVectors(v1, v2);
            newDistances[j][clusterCnt-1] = getDistanceBetweenVectors(v1, v2);
            if (newDistances[clusterCnt-1][j] < 0) {
                cout << "ERROR(Rounding to 0 instead): " << newDistances[clusterCnt-1][j] << endl;
                cout << clusterCnt << j << endl;
                newDistances[clusterCnt-1][j] = 0;
                newDistances[j][clusterCnt-1] = 0;
            }
            v2.clear();
        } else {
            newDistances[clusterCnt-1][j] = maxVal;
            newDistances[j][clusterCnt-1] = maxVal;
        }

    }
    v1.clear();

    cout << "Setting the new distances up for use..." << endl;

    distances = newDistances;
    //6. Goto step 3
    if (usedClusters.size() < (amountOfClusters - 1) * 2) {
        cout << "Looping..." << endl;
        goto Step3;
    }

    usedClusters.clear();
    //Clean up
    //free(means);
    //free(distances);
    //free(vectorsInClusters);
}

/* TODO:
     * I need to do the hierarchical clustering after the flat clustering is done.
     * Maybe in another method. I should find the two clusters that are closest and then merge those.
     * Then repeat until there is only one cluster left.
     */
int main() {
    int amountOfClusters = 10;
    //   kMeans(amountOfClusters);
    hierarchicalClustering(amountOfClusters);
    //testing();
    //test3();
    return 0;
}

