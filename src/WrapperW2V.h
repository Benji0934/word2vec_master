//
// Created by Manuel Ciosici on 15/11/16.
//

#ifndef BROWN_WRAPPERW2V_H
#define BROWN_WRAPPERW2V_H
#include <string>
#include <vector>
#include <unordered_map>
#include <boost/optional/optional.hpp>
#include <cmath>

class WrapperW2V {
private:
    long long numWords;
    long long numDimensions;
    std::unordered_map<std::string, uint32_t > words;
    std::vector<std::vector<float>> wordVectors;
    const long long maxWordLength = 50;
public:
    WrapperW2V(const std::string& fileName);

    long long int getNumDimensions() const;

    const std::unordered_map<std::string, uint32_t> &getWords() const;

    const std::vector<std::vector<float>> &getWordVectors() const;

    virtual ~WrapperW2V();
    /**
     * Returns the vector of a word if the word exists in the vector space.
     * @todo this method should return an optional<vector<float>> from the standard library as soon as that it
     * comes out of TS
     * @param word
     * @return
     */
    boost::optional<std::vector<float>> getVectorForWord(const std::string& word) const;
//    bool wordIsPresentInVectorSpace(const std::string& word) const;
    std::pair<bool, uint32_t> wordIsPresentInVectorSpace(const std::string& word) const;
    /**
     * Returns distance between words if both words are present in the vector space.
     * @todo this method should return an optional<float> from the standard library as soon as that it
     * comes out of TS
     * @param word1
     * @param word2
     * @return
     */
    boost::optional<float> calculateDistance(const std::string& word1, const std::string& word2) const;

    std::vector<float> getVectorForKnownWord(const std::string &word) const;

    float calculateDistanceForKnownWords(const std::string &word1, const std::string &word2) const;
    float calculateDistanceForKnownWords(const uint32_t &wordID1, const uint32_t &wordID2) const;
};


#endif //BROWN_WRAPPERW2V_H
