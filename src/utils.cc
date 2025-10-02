#include "utils.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace Utils {

std::mt19937 rng;

void initRandom(unsigned int seed) {
    if (seed == 0) {
        std::random_device rd;
        seed = rd();
    }
    rng.seed(seed);
}

double randomDouble(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

int randomInt(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

std::vector<double> randomVector(int size, double min, double max) {
    std::vector<double> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = randomDouble(min, max);
    }
    return vec;
}

std::vector<int> shuffleIndices(int size) {
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    return indices;
}

double mean(const std::vector<double>& vec) {
    if (vec.empty()) return 0.0;
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

double stddev(const std::vector<double>& vec) {
    if (vec.size() <= 1) return 0.0;
    
    double m = mean(vec);
    double sq_sum = 0.0;
    for (double val : vec) {
        sq_sum += (val - m) * (val - m);
    }
    return std::sqrt(sq_sum / (vec.size() - 1));
}

double normalize(double value, double min, double max) {
    if (max - min < 1e-10) return 0.0;
    return (value - min) / (max - min);
}

double clamp(double value, double min, double max) {
    return std::max(min, std::min(max, value));
}

std::string vectorToString(const std::vector<double>& vec, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision);
    oss << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        oss << vec[i];
        if (i < vec.size() - 1) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

void printProgress(int current, int total, const std::string& prefix) {
    int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);
    
    std::cout << prefix << "[";
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% (" 
              << current << "/" << total << ")\r";
    std::cout.flush();
    
    if (current == total) {
        std::cout << std::endl;
    }
}

void ClassificationMetrics::calculate() {
    int total = true_positive + true_negative + false_positive + false_negative;
    
    if (total == 0) {
        accuracy = precision = recall = f1_score = 0.0;
        return;
    }
    
    accuracy = static_cast<double>(true_positive + true_negative) / total;
    
    if (true_positive + false_positive > 0) {
        precision = static_cast<double>(true_positive) / (true_positive + false_positive);
    } else {
        precision = 0.0;
    }
    
    if (true_positive + false_negative > 0) {
        recall = static_cast<double>(true_positive) / (true_positive + false_negative);
    } else {
        recall = 0.0;
    }
    
    if (precision + recall > 0) {
        f1_score = 2.0 * (precision * recall) / (precision + recall);
    } else {
        f1_score = 0.0;
    }
}

void ClassificationMetrics::print() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Classification Metrics:\n";
    std::cout << "  Accuracy:  " << accuracy * 100 << "%\n";
    std::cout << "  Precision: " << precision * 100 << "%\n";
    std::cout << "  Recall:    " << recall * 100 << "%\n";
    std::cout << "  F1-Score:  " << f1_score << "\n";
    std::cout << "  TP: " << true_positive << ", TN: " << true_negative 
              << ", FP: " << false_positive << ", FN: " << false_negative << "\n";
}

ClassificationMetrics calculateMetrics(
    const std::vector<int>& predictions,
    const std::vector<int>& actual
) {
    ClassificationMetrics metrics = {0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0};
    
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == 1 && actual[i] == 1) {
            metrics.true_positive++;
        } else if (predictions[i] == 0 && actual[i] == 0) {
            metrics.true_negative++;
        } else if (predictions[i] == 1 && actual[i] == 0) {
            metrics.false_positive++;
        } else if (predictions[i] == 0 && actual[i] == 1) {
            metrics.false_negative++;
        }
    }
    
    metrics.calculate();
    return metrics;
}

} // namespace Utils