//
// Created by benjamin on 3/9/17.
//

#include <iostream>

#include <boost/algorithm/string/find.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <cstring>
#include <fstream>
#include "WrapperW2V.h"
#include <chrono>
#include <sstream>
#include <iterator>
#include <cfloat>
#include <string>


using namespace std;
using namespace std::chrono;
using namespace boost;

int amountOfClusters = 0;
//string vectorFilePath = "notSet";
//string vectorFilePath = "../word2vecFiles/text8-250kb-vector.bin";


double maxVal = DBL_MAX;
vector<int> usedClusters;

void createTxtVectorFile(string basicString);

/*
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



    for (auto it = wrapper.getWordVectors().begin(); it != wrapper.getWordVectors().end(); ++it) {
        for (auto it1 = *it.base()->begin(); it1 != *it.base()->end(); ++it1) {
            cout << "Test" + to_string(it1) << endl;

        }
    }
//
    return 0;

}
*/
void createClusterBitString(string vectorFilePath);

/*
 * Code taken from the word2vec.c file published by Mikolov.
 */
void kMeans(int amountOfClusters, string vectorFilePath) {
    WrapperW2V wrapper = WrapperW2V(vectorFilePath);
    long vocab_size = wrapper.getWords().size();
    //long long dims = wrapper.getNumDimensions();
    milliseconds startTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    milliseconds endTime;
    long a, b, c, d;
    long long layer1_size = wrapper.getNumDimensions(); //Amount of features/amount of weights in the NN. 200 originally
    float weight = 0;
    vector<vector<float>> wordVectors;
    wordVectors = wrapper.getWordVectors();
    vector<float> wordVec;

// might have to find the maximum length for a string, instead of 10000.
    //char output_file[10000];
    //strcpy(output_file, "../word2vecFiles/classes.txt");
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
    //myfile.open("/home/benjamin/CLionProjects/wordClustering/classes.txt");
    myfile.open("../classes/classes.txt");
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

double getDistanceBetweenVectors(vector<double> v1, vector<double> v2, long dims) {
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
    if ((1 - cosSim) < 0) {
        cout << "ERROR(Rounding to 0 instead): " << endl;
        cout << "Wrong value: " << cosSim << endl;
        cosSim = 1;
    }
    return 1 - cosSim;
    //cos(d1, d2) = 1-(d1 ⋅ d2) / ||d1||*||d2||
}

void hierarchicalClustering(int amountOfClusters, string vectorFilePath, long amountOfClasses) {
    WrapperW2V wrapper = WrapperW2V(vectorFilePath);
    //long vocab_size = wrapper.getWords().size();
    long dims = wrapper.getNumDimensions();
    //0. Creating variables and files.
    long clusterCnt = amountOfClusters;
    string clusterCntString = to_string(clusterCnt);
    vector<vector<double>> means;
    vector<vector<double>> distances;
    vector<int> amountOfVectorsInClusters;
    unsigned int level = 0;
    int iVal = 0;
    int jVal = 0;
    string currentString;
    string currentClass;
    //Deleting old tree and hierarchy files
    //remove("/home/benjamin/CLionProjects/wordClustering/tree.txt");
    remove("../tree.txt");
    //remove("/home/benjamin/CLionProjects/wordClustering/hierarchy.txt");
    remove("../hierarchy.txt");

    //Initializing 'means' and 'distances'
    for (int i = 0; i < amountOfClusters; i++) {
        amountOfVectorsInClusters.push_back(0);
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
        temp.push_back(maxVal);
    }
    for (int i = 0; i < amountOfClusters; i++) {
        distances.push_back(temp);
    }

    //1. Calculate means (mean-points)
    //cout << "Calculating initial means..." << endl;
    //std::ifstream infile("/home/benjamin/CLionProjects/wordClustering/classes/classes.txt");
    std::ifstream infile("../classes/classes.txt");
    std::string line;
    while (std::getline(infile, line)) {
        string word;
        int clusterID;
        //Finding the current clusterID.
        vector<string> stringAndPos = split(line, ' ');
        word = stringAndPos.front();
        clusterID = stoi(stringAndPos.back());
        //Finding the vector for the word.
        vector<float> wordVector = wrapper.getVectorForKnownWord(word);
        amountOfVectorsInClusters[clusterID]++;

        //Calculating new mean for the cluster found.
        for (unsigned int i = 0; i < dims; i++) {
            means[clusterID][i] = (means[clusterID][i] + wordVector.at(i)) / amountOfVectorsInClusters[clusterID];
        }
    }
    //2. Measure cosine distance between means
    //Going through all pairs of vectors.
    for (int k = 0; k < amountOfClusters; k++) {

        vector<double> v1;
        vector<double> v2;
        v1 = means[k];
        for (int j = 0; j < amountOfClusters; j++) {
            //Checking if the clusternumber is in use.
            if (amountOfVectorsInClusters[k] != 0) {
                if (amountOfVectorsInClusters[j] != 0) {
                    v2 = means[j];
                    //The distance from the same vector to itself is 0.
                    if (k == j) {
                        distances[k][j] = 0;
                    } else {
                        distances[k][j] = getDistanceBetweenVectors(v1, v2, dims);
                        if (distances[k][j] < 0) {
                            cout << "ERROR in initialization of clusters: " << distances[k][j] << endl;
                            cout << k << j << endl;
                        }
                    }
                    v2.clear();
                } else {
                    distances[k][j] = maxVal;
                }
            } else {
                distances[k][j] = maxVal;
            }
        }
        v1.clear();
    }
    Step3:
    //cout << "Finding shortest distance between clusters..." << endl;
    //3. Merge the two closest clusters
    double shortestDist = maxVal;
    iVal = -1;
    jVal = -1;
    //Running through the
    for (int i = 0; i < clusterCnt; i++) {
        for (int j = 0; j < clusterCnt; j++) {
            if (i != j) {
                //If the cluster is in use, and it has not been deleted yet, then we can check the dist.
                //cout << (amountOfVectorsInClusters[i] != 0) << endl;
                //cout << (amountOfVectorsInClusters[j] != 0) << endl;
           //     cout << (find(usedClusters.begin(), usedClusters.end(), i) == usedClusters.end()) << endl;
           //     cout << (find(usedClusters.begin(), usedClusters.end(), j) == usedClusters.end()) << endl;
                if ((amountOfVectorsInClusters[i] != 0)
                    && (amountOfVectorsInClusters[j] != 0))
                {
                  //  cout << "currentDist " << distances[i][j] << endl;
                    double currentDist = distances[i][j];
                    if (currentDist < shortestDist) {
                        iVal = i;
                        jVal = j;
                        shortestDist = currentDist;
                    }
                }
            }
        }
    }
    if (jVal == -1 || iVal == -1) {
            cout << "ERROR distance not found: iVAL or jVAL equals zero" << endl;
            cout << "Printing the distances:" << endl;
        for (int i = 0; i < clusterCnt; i++) {
            for (int j = 0; j < clusterCnt; j++) {
                if (i != j) {
                    //If the cluster is in use, and it has not been deleted yet, then we can check the dist.
                    //cout << (amountOfVectorsInClusters[i] != 0) << endl;
                    //cout << (amountOfVectorsInClusters[j] != 0) << endl;
                    //     cout << (find(usedClusters.begin(), usedClusters.end(), i) == usedClusters.end()) << endl;
                    //     cout << (find(usedClusters.begin(), usedClusters.end(), j) == usedClusters.end()) << endl;
                    if (distances[i][j] < maxVal) {
                        cout << "Distance for " << i << " " << j << endl;
                        cout << distances[i][j] << endl;
                    }
                }
            }
        }
    }

    string iString = to_string(iVal);
    string jString = to_string(jVal);
    //cout << "Adding numbers to usedClusters..." << jString << " " << iString << endl;
    //Adding clusters that have been merged.
    usedClusters.push_back(iVal);
    usedClusters.push_back(jVal);

    //cout << "Creating tree and hierarchy files..." << endl;
    //Writing tree
    //Format: label,parent,level
    ofstream treeFile;
    //treeFile.open("/home/benjamin/CLionProjects/wordClustering/tree.txt", std::ios_base::app);
    treeFile.open("../tree.txt", std::ios_base::app);
    treeFile << iString << "," << clusterCntString << "," << level << "\n";
    treeFile << jString << "," << clusterCntString << "," << level << "\n";
    treeFile.close();
    //Writing hierarchy. It should keep track of the tree levels.
    //Format: level,label<0>,label<1>,...,label<amountOfWords>
    ofstream hierarchyFile;
    //hierarchyFile.open("/home/benjamin/CLionProjects/wordClustering/hierarchy.txt", std::ios_base::app);
    hierarchyFile.open("../hierarchy.txt", std::ios_base::app);
    //string s = to_string(level) + ",";
    string s = "";
    ifstream mergedClassesReadFile1("../classes/mergedClasses.txt");
    //ifstream mergedClassesReadFile1("/home/benjamin/CLionProjects/wordClustering/classes/mergedClasses.txt");
    string mergedClassesStr1;
    while (std::getline(mergedClassesReadFile1, mergedClassesStr1)) {
        //Finding the class of the word.
        size_t found = mergedClassesStr1.find(' ');
        if (found != string::npos) {
            currentClass = mergedClassesStr1.substr(found + 1, mergedClassesStr1.length() - found - 1);
        }
        //Adding the word's class to the hierarchy file.
        s = s + currentClass + ",";
    }
    mergedClassesReadFile1.close();
    hierarchyFile << s + "\n";
    hierarchyFile.close();
    level++;

    //cout << "Merging..." << endl;
    //Creating copy of classes
    //ifstream mergedClassesReadFile("/home/benjamin/CLionProjects/wordClustering/classes/mergedClasses.txt");
    ifstream mergedClassesReadFile("../classes/mergedClasses.txt");
    string mergedClassesStr;
    ofstream oldClassesWriteFile;
    //remove("/home/benjamin/CLionProjects/wordClustering/classes/oldClasses.txt");
    remove("../classes/oldClasses.txt");
    //oldClassesWriteFile.open("/home/benjamin/CLionProjects/wordClustering/classes/oldClasses.txt");
    oldClassesWriteFile.open("../classes/oldClasses.txt");
    while (getline(mergedClassesReadFile, mergedClassesStr)) {
        oldClassesWriteFile << mergedClassesStr + "\n";
    }
    mergedClassesReadFile.close();
    oldClassesWriteFile.close();

    //Merging clusters in mergedClasses.txt
    ofstream mergedClassesWriteFile;
    mergedClassesWriteFile.open("../classes/mergedClasses.txt");
    //mergedClassesWriteFile.open("/home/benjamin/CLionProjects/wordClustering/classes/mergedClasses.txt");
    ifstream oldClassesReadFile("../classes/oldClasses.txt");
    //ifstream oldClassesReadFile("/home/benjamin/CLionProjects/wordClustering/classes/oldClasses.txt");
    string str;
    while (std::getline(oldClassesReadFile, str)) {
        str = str + "\n";
        //Cutting it up.
        std::size_t found = str.find(' ');
        if (found != std::string::npos) {
            //Finding the old class of the word.
            currentString = str.substr(0, found);
            currentClass = str.substr(found + 1, str.length() - found - 1);
            //If the old class is one of the merged classes, then rename it to the new one.
            if (stoi(currentClass) == iVal) {
      //          cout << "Found iVal " << iVal << endl;
                string newString = str.substr(0, str.find(iString));
                newString = newString + clusterCntString + "\n";
                mergedClassesWriteFile << newString;
            } else if (stoi(currentClass) == jVal) {
      //          cout << "Found jVal " << jVal << endl;

                string newString = str.substr(0, str.find(jString));
                newString = newString + clusterCntString + "\n";
                mergedClassesWriteFile << newString;
            } //Otherwise we write the same line again.
            else {
                mergedClassesWriteFile << str;
            }
        }
    }
    mergedClassesWriteFile.close();
    oldClassesReadFile.close();

    //4. Recalculate mean/avg for the new cluster
    //cout << "Recalculating mean..." << endl;
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
    //cout << "Recalculating distances..." << endl;
    //Creating new distance matrix..
    vector<vector<double>> newDistances;
    vector<double> newDistVec;
    for (unsigned int i = 0; i < clusterCnt + 1; i++) {
        newDistVec.push_back(maxVal);
    }
    for (unsigned int l = 0; l < clusterCnt + 1; ++l) {
        newDistances.push_back(newDistVec);
    }


    //Copy the old distances..
    //Going through all pairs of clusters.
    for (int i = 0; i < clusterCnt; i++) {
        for (int j = 0; j < clusterCnt; j++) {
            //If the clusters are not among the merged/deleted ones then we find the new distance.
            if ((std::find(usedClusters.begin(), usedClusters.end(), j) == usedClusters.end())) {
                //Otherwise we just use the old distance.
                if ((std::find(usedClusters.begin(), usedClusters.end(), i) == usedClusters.end())) {
                    newDistances[i][j] = distances[i][j];
                } else {
                    newDistances[i][j] = maxVal;
                }
            } else {
                newDistances[i][j] = maxVal;
            }
        }
    }

    //cout << "Calculating distances to new cluster..." << endl;
    //Calc new distances
    vector<double> v1;
    vector<double> v2;
    //cout << clusterCnt << endl;
    //Finding mean for the new cluster
    v1 = means[clusterCnt];
    //Setting distance to self to 0.
    newDistances[clusterCnt][clusterCnt] = 0;
    //Finding distance to all other clusters.
    for (int j = 0; j < clusterCnt; j++) {
        //Unless they are gone.
        if (std::find(usedClusters.begin(), usedClusters.end(), j) == usedClusters.end()) {
            //Setting distances
            v2 = means[j];
            newDistances[clusterCnt][j] = getDistanceBetweenVectors(v1, v2, dims);
            newDistances[j][clusterCnt] = getDistanceBetweenVectors(v1, v2, dims);
            //Checking for rounding errors.
            if (newDistances[clusterCnt][j] < 0) {
                cout << "ERROR(Rounding to 0 instead): " << newDistances[clusterCnt][j] << endl;
                cout << clusterCnt << j << endl;
                newDistances[clusterCnt][j] = 0;
                newDistances[j][clusterCnt] = 0;
            }
            v2.clear();
        } else {
            newDistances[clusterCnt][j] = maxVal;
            newDistances[j][clusterCnt] = maxVal;
        }
    }
    v1.clear();

    //Counting up, the amount of clusters in our matrix, and the amount of clusters produced.
    clusterCnt++;
    clusterCntString = to_string(clusterCnt);
    //Updating vectors in clusters. The amount doesnt really mean anything as long as it above 1.
    amountOfVectorsInClusters.push_back(1);
    //cout << "Setting the new distances up for use..." << endl;
    distances = newDistances;
    //6. Goto step 3

    cout << "Merged old clusters and created new cluster with number:" << clusterCnt << endl;
    if (clusterCnt < (amountOfClasses + amountOfClusters - 1)) {
        //if (usedClusters.size() < (amountOfClasses - 1) * 2) {
   //     cout << "Looping..." << endl;
        goto Step3;
    }

    usedClusters.clear();
    //Clean up
    //free(means);
    //free(distances);
    //free(amountOfVectorsInClusters);
}

/*
void TrainModel() {
    WrapperW2V wrapper = WrapperW2V(vectorFilePath);
    long vocab_size = wrapper.getWords().size();
    long long dims = wrapper.getNumDimensions();
    long a, b, c, d;
    FILE *fo;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    printf("Starting training using file %s\n", train_file);
    starting_alpha = alpha;
    if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;
    InitNet();
    if (negative > 0) InitUnigramTable();
    start = clock();
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    fo = fopen(output_file, "wb");
    if (classes == 0) {
        // Save the word vectors
        fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
        for (a = 0; a < vocab_size; a++) {
            fprintf(fo, "%s ", vocab[a].word);
            if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
            fprintf(fo, "\n");
        }
    } else {
        // Run K-means on the word vectors
        int clcn = classes, iter = 10, closeid;
        int *centcn = (int *)malloc(classes * sizeof(int));
        int *cl = (int *)calloc(vocab_size, sizeof(int));
        real closev, x;
        real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
        for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
        for (a = 0; a < iter; a++) {
            for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
            for (b = 0; b < clcn; b++) centcn[b] = 1;
            for (c = 0; c < vocab_size; c++) {
                for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
                centcn[cl[c]]++;
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
        for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);

        free(centcn);
        free(cent);
        free(cl);
    }
    fclose(fo);
}
*/
int main(int ac, char **av) {

    if (ac == 3) {
        string amountOfClustersStr = av[2];
        string vectorFilePath = av[1];
        amountOfClusters = stoi(amountOfClustersStr);


        cout << amountOfClusters << vectorFilePath << endl;
        cout << "Running Kmeans on the vectors" << endl;
        kMeans(amountOfClusters, vectorFilePath);
        //check classes.txt file.
        //std::ifstream infile("/home/benjamin/CLionProjects/wordClustering/classes/classes.txt");
        std::ifstream infile("../classes/classes.txt");
        std::string line;
        vector<int> seenClasses;
        int currentClass = -1;
        while (std::getline(infile, line)) {
            int classesStartIdx = line.find(" ");
            currentClass = stoi(line.substr(classesStartIdx + 1, line.length()));
            //cout << currentClass << endl;
            if (std::find(seenClasses.begin(), seenClasses.end(), currentClass) != seenClasses.end()) {
                /* v contains x */
                //Do nothing
            } else {
                /* v does not contain x */
                seenClasses.push_back(currentClass);
            }
        }
        for (int i = 0; i < amountOfClusters; i++) {
            if (std::find(seenClasses.begin(), seenClasses.end(), i) != seenClasses.end()) {
                /* v contains x */
                //Do nothing
            } else {
                /* v does not contain x */
                // cout << "ERROR: The classes file does not contain this clusterNumber: " << i << endl;
            }
        }
        cout << "Seen classes.size = " << seenClasses.size() << endl;

        //Creating mergedClasses..
        //std::ifstream src("/home/benjamin/CLionProjects/wordClustering/classes/classes.txt", std::ios::binary);
        std::ifstream src("../classes/classes.txt", std::ios::binary);
        //std::ofstream dst("/home/benjamin/CLionProjects/wordClustering/classes/mergedClasses.txt", std::ios::binary);
        std::ofstream dst("../classes/mergedClasses.txt", std::ios::binary);
        dst << src.rdbuf();
        src.close();
        dst.close();
        cout << "Doing hierarchical clustering.." << endl;
        hierarchicalClustering(amountOfClusters, vectorFilePath, seenClasses.size());
        cout << "Creating bitstrings.." << endl;
        createClusterBitString(vectorFilePath);
        cout << "Creating txtVec file.." << endl;
        createTxtVectorFile(vectorFilePath);
        return 0;
    } else {
        return 1;
    }
}

void createTxtVectorFile(string vectorFilePath) {
    vectorFilePath = vectorFilePath;
    WrapperW2V wrapper = WrapperW2V(vectorFilePath);
    long vocab_size = wrapper.getWords().size();
    long long dims = wrapper.getNumDimensions();
    ofstream txtFile;
    boost::replace_all(vectorFilePath, ".bin", ".txt");
    txtFile.open(vectorFilePath);
    txtFile << vocab_size << " " << dims << "\n";
    for (int i = 0; i < vocab_size; i++) {
        txtFile << wrapper.getInverseWords().at(i);
        for (int j = 0; j < dims; j++) {
            txtFile << " " << wrapper.getWordVectors().at(i).at(j);
        }
        txtFile << "\n";
    }
    txtFile.close();
}

/*
 *
 *
 * Example: 00000	‘Forgive	2
            00000	doin	2
            00000	managers’	2
   Name of the file should be something similar to: gha.250M-c2000.paths
   Should sort the file according to the string.
   Noise is not relevant in this case. But it becomes relevant in the HDBSCAN.
 */
void createClusterBitString(string vectorFilePath) {
    WrapperW2V wrapper = WrapperW2V(vectorFilePath);
    long vocab_size = wrapper.getWords().size();
   // cout << "Initialized wrapper" << endl;
    //long long dims = wrapper.getNumDimensions();
    //Assign 0 or 1 to each branch in the tree. Use the tree file.
    unordered_map<int, int> branches;
    std::ifstream treeFile("../tree.txt");
    //std::ifstream treeFile("/home/benjamin/CLionProjects/wordClustering/tree.txt");
    bool first = true;
    for (std::string line; getline(treeFile, line);) {
        int clusterNumber = stoi(line.substr(0, line.find(",")));
        if (first) {
            branches.insert(make_pair<int, int>((int &&) clusterNumber, 0));
            first = false;
        } else {
            branches.insert(make_pair<int, int>((int &&) clusterNumber, 1));
            first = true;
        }
    }

    //Assign paths to each word. Use the hierarchy file.
    vector<int> prevClusters;
    vector<string> clusterStrings;
    //int prevCluster = -1;

    //Initializing vector.
    for (int i = 0; i < vocab_size; i++) {
        prevClusters.push_back(-1);
        clusterStrings.push_back("");
    }
    //Reading hFile.
    std::ifstream hierarchyFile("../hierarchy.txt");
    //std::ifstream hierarchyFile("/home/benjamin/CLionProjects/wordClustering/hierarchy.txt");
    for (std::string line; getline(hierarchyFile, line);) {
        int wordNum = 0;
        while (line.length() > 1) {
            string currentCluster = line.substr(0, line.find(","));
            int currentClusterInt = stoi(currentCluster);
            line = line.substr(line.find(",") + 1, line.length() - (line.find(",") + 1));

            if (prevClusters[wordNum] != currentClusterInt) {
                clusterStrings[wordNum] = clusterStrings[wordNum] + to_string(branches[currentClusterInt]);
            }
            prevClusters[wordNum] = currentClusterInt;
            wordNum++;

            /*iterator_range<string::iterator> r = find_nth(line, ",", 1);
            long secondComma = distance(line.begin(), r.begin());
            //line = "";
            long firstComma = line.find(",");
            string currentClusterline = line.substr(firstComma, secondComma-firstComma);
             */
        }
    }

    // cout << wrapper.getInverseWords().at(0) << endl;

    const unordered_map<uint32_t, string> &inverseWords = wrapper.getInverseWords();
    remove("../paths.txt");
    //remove("/home/benjamin/CLionProjects/wordClustering/paths.txt");
    ofstream myfile;
    myfile.open("../paths.txt");
    //myfile.open("/home/benjamin/CLionProjects/wordClustering/paths.txt");
    for (int i = 0; i < vocab_size; i++) {
        myfile << clusterStrings[i] << " " << inverseWords.at(i) << " " << "1" << "\n";
    }
    myfile.close();
}

