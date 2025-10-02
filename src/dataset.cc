#include "dataset.h"


Dataset::Dataset() : num_samples(0), num_features(30) {}

Dataset::~Dataset() {}

bool Dataset::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int line_count = 0;
    
    while (std::getline(file, line)) {
        line_count++;
        std::stringstream ss(line);
        std::string token;
        std::vector<double> feature_row;
        
        if (!std::getline(ss, token, ',')) {
            std::cerr << "Error reading ID at line " << line_count << std::endl;
            continue;
        }
        ids.push_back(token);
        
        if (!std::getline(ss, token, ',')) {
            std::cerr << "Error reading diagnosis at line " << line_count << std::endl;
            continue;
        }
        
        if (token == "M") {
            labels.push_back(1);
        } else if (token == "B") {
            labels.push_back(0);
        } else {
            std::cerr << "Unknown diagnosis label: " << token << std::endl;
            continue;
        }
        
        int feature_count = 0;
        while (std::getline(ss, token, ',') && feature_count < 30) {
            try {
                double value = std::stod(token);
                feature_row.push_back(value);
                feature_count++;
            } catch (const std::exception& e) {
                std::cerr << "Error parsing feature at line " << line_count 
                         << ": " << e.what() << std::endl;
                break;
            }
        }
        
        if (feature_count == 30) {
            features.push_back(feature_row);
        } else {
            std::cerr << "Incomplete feature set at line " << line_count 
                     << " (found " << feature_count << " features)" << std::endl;
            if (!labels.empty()) labels.pop_back();
            if (!ids.empty()) ids.pop_back();
        }
    }
    
    file.close();
    
    num_samples = features.size();
    
    if (num_samples == 0) {
        std::cerr << "Error: No valid samples loaded" << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded " << num_samples << " samples" << std::endl;
    return true;
}

void Dataset::normalize() {
    if (features.empty()) return;
    
    feature_means.resize(num_features, 0.0);
    feature_stds.resize(num_features, 0.0);
    
    for (int j = 0; j < num_features; j++) {
        double sum = 0.0;
        for (int i = 0; i < num_samples; i++) {
            sum += features[i][j];
        }
        feature_means[j] = sum / num_samples;
    }
    
    for (int j = 0; j < num_features; j++) {
        double sum_sq = 0.0;
        for (int i = 0; i < num_samples; i++) {
            double diff = features[i][j] - feature_means[j];
            sum_sq += diff * diff;
        }
        feature_stds[j] = std::sqrt(sum_sq / num_samples);
        
        if (feature_stds[j] < 1e-10) {
            feature_stds[j] = 1.0;
        }
    }
    
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            features[i][j] = (features[i][j] - feature_means[j]) / feature_stds[j];
        }
    }
    
    std::cout << "Data normalized using Z-score normalization" << std::endl;
}

void Dataset::normalizeWithStats(const std::vector<double>& means, 
                                 const std::vector<double>& stds) {
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            features[i][j] = (features[i][j] - means[j]) / stds[j];
        }
    }
}

void Dataset::createKFolds(int k, unsigned int seed) {
    fold_indices.resize(num_samples);
    
    std::vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; i++) {
        indices[i] = i;
    }
    
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int i = 0; i < num_samples; i++) {
        fold_indices[indices[i]] = i % k;
    }
    
    std::cout << "Created " << k << "-fold cross validation splits" << std::endl;
}

void Dataset::getTrainTestSplit(int test_fold,
                                std::vector<std::vector<double>>& train_X,
                                std::vector<int>& train_y,
                                std::vector<std::vector<double>>& test_X,
                                std::vector<int>& test_y) const {
    train_X.clear();
    train_y.clear();
    test_X.clear();
    test_y.clear();
    
    for (int i = 0; i < num_samples; i++) {
        if (fold_indices[i] == test_fold) {
            test_X.push_back(features[i]);
            test_y.push_back(labels[i]);
        } else {
            train_X.push_back(features[i]);
            train_y.push_back(labels[i]);
        }
    }
}

void Dataset::printStatistics() const {
    std::cout << "\n===== Dataset Statistics =====" << std::endl;
    std::cout << "Number of samples: " << num_samples << std::endl;
    std::cout << "Number of features: " << num_features << std::endl;
    
    int num_benign = 0, num_malignant = 0;
    for (int label : labels) {
        if (label == 0) num_benign++;
        else num_malignant++;
    }
    
    std::cout << "Class distribution:" << std::endl;
    std::cout << "  Benign (B): " << num_benign 
              << " (" << std::fixed << std::setprecision(2) 
              << (100.0 * num_benign / num_samples) << "%)" << std::endl;
    std::cout << "  Malignant (M): " << num_malignant 
              << " (" << (100.0 * num_malignant / num_samples) << "%)" << std::endl;
    
    if (!feature_means.empty()) {
        std::cout << "\nFeature statistics (after normalization):" << std::endl;
        std::cout << "  Mean should be ~0, Std should be ~1" << std::endl;
    }
    std::cout << "==============================\n" << std::endl;
}