#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>

namespace Utils {
    // Random number generator
    extern std::mt19937 rng;
    
    // Initialize random seed
    void initRandom(unsigned int seed = 0);
    
    // Generate random double in range [min, max]
    double randomDouble(double min, double max);
    
    // Generate random integer in range [min, max]
    int randomInt(int min, int max);
    
    // Generate random vector of doubles
    std::vector<double> randomVector(int size, double min, double max);
    
    // Shuffle vector indices
    std::vector<int> shuffleIndices(int size);
    
    // Calculate mean of vector
    double mean(const std::vector<double>& vec);
    
    // Calculate standard deviation
    double stddev(const std::vector<double>& vec);
    
    // Normalize value using min-max normalization
    double normalize(double value, double min, double max);
    
    // Clamp value between min and max
    double clamp(double value, double min, double max);
    
    // Convert vector to string for printing
    std::string vectorToString(const std::vector<double>& vec, int precision = 4);
    
    // Print progress bar
    void printProgress(int current, int total, const std::string& prefix = "");
    
    // Calculate confusion matrix metrics
    struct ClassificationMetrics {
        int true_positive;
        int true_negative;
        int false_positive;
        int false_negative;
        double accuracy;
        double precision;
        double recall;
        double f1_score;
        
        void calculate();
        void print() const;
    };
    
    ClassificationMetrics calculateMetrics(
        const std::vector<int>& predictions,
        const std::vector<int>& actual
    );
}

#endif // UTILS_H