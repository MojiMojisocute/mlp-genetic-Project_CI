#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

class Dataset {
private:
    std::vector<std::vector<double>> features;  
    std::vector<int> labels;                    
    std::vector<std::string> ids;               
    
    std::vector<double> feature_means;
    std::vector<double> feature_stds;
    
    int num_samples;
    int num_features;
    
    std::vector<int> fold_indices; 
    
public:
    Dataset();
    ~Dataset();
    
    bool loadFromFile(const std::string& filename);

    void normalize();
    void normalizeWithStats(const std::vector<double>& means, 
                           const std::vector<double>& stds);
    
    void createKFolds(int k = 10, unsigned int seed = 42);
        void getTrainTestSplit(int test_fold, 
                          std::vector<std::vector<double>>& train_X,
                          std::vector<int>& train_y,
                          std::vector<std::vector<double>>& test_X,
                          std::vector<int>& test_y) const;
    
    int getNumSamples() const { return num_samples; }
    int getNumFeatures() const { return num_features; }
    const std::vector<std::vector<double>>& getFeatures() const { return features; }
    const std::vector<int>& getLabels() const { return labels; }
    const std::vector<double>& getFeatureMeans() const { return feature_means; }
    const std::vector<double>& getFeatureStds() const { return feature_stds; }

    void printStatistics() const;
};

#endif // DATASET_H