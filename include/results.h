#ifndef RESULTS_H
#define RESULTS_H

#include <vector>
#include <string>
#include "utils.h"

struct FoldResult {
    int fold_number;
    double train_accuracy;
    double test_accuracy;
    Utils::ClassificationMetrics train_metrics;
    Utils::ClassificationMetrics test_metrics;
    int generations_used;
    double best_fitness;
};

struct ExperimentResult {
    std::vector<int> network_structure;
    int run_id;
    unsigned int seed;
    std::vector<FoldResult> fold_results;
    double mean_test_accuracy;
    double std_test_accuracy;
    double mean_train_accuracy;
    double std_train_accuracy;
    
    ExperimentResult() : run_id(0), seed(0), mean_test_accuracy(0.0), 
                        std_test_accuracy(0.0), mean_train_accuracy(0.0), 
                        std_train_accuracy(0.0) {}
    
    void calculate();
    void print() const;
    void saveToFile(const std::string& filename) const;
};

class ResultsManager {
private:
    std::vector<ExperimentResult> experiments;
    
public:
    ResultsManager();
    ~ResultsManager();
    
    void addExperiment(const ExperimentResult& result);
    void printSummary() const;
    void printComparison() const;
    
    void saveAllResults(const std::string& filename) const;
    void saveSummaryResults(const std::string& filename) const;
    
    const std::vector<ExperimentResult>& getExperiments() const { 
        return experiments; 
    }
    
    void clear() { experiments.clear(); }
    int size() const { return experiments.size(); }
};

#endif // RESULTS_H