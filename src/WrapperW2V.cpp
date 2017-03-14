//
// Created by Manuel Ciosici on 15/11/16.
//

#include "WrapperW2V.h"
#include <iostream>
#include <fstream>
//#include "../Utils.h"

using namespace std;

WrapperW2V::WrapperW2V(const std::string &fileName) {
    ifstream file(fileName, ios::in | ios::binary);
    if (file.is_open()) {
        file.seekg(0, ios::beg);
        file >> numWords;
        file >> numDimensions;
        wordVectors = std::vector<std::vector<float>>(numWords, vector<float>(numDimensions, 0));
        char w[maxWordLength];
        for (long long wordID = 0; wordID < numWords; ++wordID) {
            long long a = 0;
            while (1) {
                file.read(&w[a], sizeof(char));
                if (file.eof() || (w[a] == ' ')) {
                    break;
                }
                if ((a < maxWordLength) && (w[a] != '\n')) {
                    a++;
                }
            }
            w[a] = 0;
//            vector<float> wordVector(numDimensions, 0);
            float len = 0;
            for (unsigned int dimensionIterator = 0; dimensionIterator < numDimensions; dimensionIterator++) {
                file.read(reinterpret_cast<char *>(&wordVectors[wordID][dimensionIterator]), sizeof(float));
                len += wordVectors[wordID][dimensionIterator] * wordVectors[wordID][dimensionIterator];
            }
            len = sqrt(len);
#pragma omp simd
            for (unsigned int dimensionIterator = 0; dimensionIterator < numDimensions; dimensionIterator++) {
                wordVectors[wordID][dimensionIterator] /= len;
            }
            words.insert(make_pair(string(w), wordID));
            inverseWords.insert(make_pair(wordID, string(w)));
        }
        file.close();
    }
}

WrapperW2V::~WrapperW2V() {

}

boost::optional<std::vector<float>> WrapperW2V::getVectorForWord(const std::string &word) const {
    const auto &it = words.find(word);
    if (it != words.end()) {
        return wordVectors[(*it).second];
    } else {
        return boost::optional<vector<float>>();
    }
}

std::vector<float> WrapperW2V::getVectorForKnownWord(const std::string &word) const {
    const auto &it = words.find(word);
    return wordVectors[(*it).second];
}

std::pair<bool, uint32_t> WrapperW2V::wordIsPresentInVectorSpace(const std::string &word) const {
    const auto &it = words.find(word);
    if (words.find(word) != words.end()) {
        return {words.find(word) != words.end(), (*it).second};
    } else {
        return {false, 0};
    }
}

boost::optional<float>
WrapperW2V::calculateDistance(const string &word1, const string &word2) const {
    float distance = 0;
    const auto vectorWord1 = this->getVectorForWord(word1);
    const auto vectorWord2 = this->getVectorForWord(word2);
    if (vectorWord1 && vectorWord2) {
        const std::vector<float> &vW1 = *vectorWord1;
        const std::vector<float> &vW2 = *vectorWord2;
        for (unsigned int dimensionIterator = 0; dimensionIterator < numDimensions; dimensionIterator++) {
            distance += vW1[dimensionIterator] * vW2[dimensionIterator];
        }
        return distance;
    }
    return boost::optional<float>();
}

float
WrapperW2V::calculateDistanceForKnownWords(const string &word1, const string &word2) const {
    const auto &it1 = words.find(word1);
    const auto id1 = (*it1).second;
    const auto &it2 = words.find(word2);
    const auto id2 = (*it2).second;
    return calculateDistanceForKnownWords(id1, id2);
}

float WrapperW2V::calculateDistanceForKnownWords(const uint32_t &wordID1, const uint32_t &wordID2) const {
    float distance = 0;
    const auto &wv1 = wordVectors[wordID1];
    const auto &wv2 = wordVectors[wordID2];

    for (uint32_t dimensionIterator = 0; dimensionIterator < numDimensions; dimensionIterator++) {
        distance += wv1[dimensionIterator] * wv2[dimensionIterator];
    }
    return distance;
}

long long int WrapperW2V::getNumDimensions() const {
    return numDimensions;
}

const unordered_map<string, uint32_t> &WrapperW2V::getWords() const {
    return words;
}

const unordered_map<uint32_t, string> &WrapperW2V::getInverseWords() const {
    return inverseWords;
}


const vector<vector<float>> &WrapperW2V::getWordVectors() const {
    return wordVectors;
}
