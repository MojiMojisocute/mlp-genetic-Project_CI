#include "results.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <algorithm>

void ExperimentResult::calculate() {
    if (fold_results.empty()) {
        mean_test_accuracy = mean_train_accuracy = 0.0;
        std_test_accuracy = std_train_accuracy = 0.0;
        return;
    }
    
    double sum_test = 0.0, sum_train = 0.0;
    for (const auto& fold : fold_results) {
        sum_test += fold.test_accuracy;
        sum_train += fold.train_accuracy;
    }
    mean_test_accuracy = sum_test / fold_results.size();
    mean_train_accuracy = sum_train / fold_results.size();
    
    double sq_sum_test = 0.0, sq_sum_train = 0.0;
    for (const auto& fold : fold_results) {
        sq_sum_test += (fold.test_accuracy - mean_test_accuracy) * 
                       (fold.test_accuracy - mean_test_accuracy);
        sq_sum_train += (fold.train_accuracy - mean_train_accuracy) * 
                        (fold.train_accuracy - mean_train_accuracy);
    }
    
    if (fold_results.size() > 1) {
        std_test_accuracy = std::sqrt(sq_sum_test / (fold_results.size() - 1));
        std_train_accuracy = std::sqrt(sq_sum_train / (fold_results.size() - 1));
    } else {
        std_test_accuracy = std_train_accuracy = 0.0;
    }
}

void ExperimentResult::print() const {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Run ID: " << run_id << " | Seed: " << seed << "\n";
    std::cout << "Network Structure: ";
    for (size_t i = 0; i < network_structure.size(); i++) {
        std::cout << network_structure[i];
        if (i < network_structure.size() - 1) std::cout << "-";
    }
    std::cout << "\n" << std::string(70, '=') << "\n";
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nCross-Validation Summary:\n";
    std::cout << "  Mean Train Accuracy: " << mean_train_accuracy * 100 
              << "% (Â±" << std_train_accuracy * 100 << "%)\n";
    std::cout << "  Mean Test Accuracy:  " << mean_test_accuracy * 100 
              << "% (Â±" << std_test_accuracy * 100 << "%)\n";
}

void ExperimentResult::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    
    file << "Run_ID: " << run_id << "\n";
    file << "Seed: " << seed << "\n";
    file << "Network Structure: ";
    for (size_t i = 0; i < network_structure.size(); i++) {
        file << network_structure[i];
        if (i < network_structure.size() - 1) file << "-";
    }
    file << "\n\n";
    
    file << std::fixed << std::setprecision(4);
    file << "Fold,Train_Accuracy,Test_Accuracy,Generations,Best_Fitness\n";
    
    for (const auto& fold : fold_results) {
        file << fold.fold_number << ","
             << fold.train_accuracy << ","
             << fold.test_accuracy << ","
             << fold.generations_used << ","
             << fold.best_fitness << "\n";
    }
    
    file << "\nMean Train Accuracy," << mean_train_accuracy << "\n";
    file << "Std Train Accuracy," << std_train_accuracy << "\n";
    file << "Mean Test Accuracy," << mean_test_accuracy << "\n";
    file << "Std Test Accuracy," << std_test_accuracy << "\n";
    
    file.close();
}

ResultsManager::ResultsManager() {}

ResultsManager::~ResultsManager() {}

void ResultsManager::addExperiment(const ExperimentResult& result) {
    experiments.push_back(result);
}

void ResultsManager::printSummary() const {
    for (const auto& exp : experiments) {
        exp.print();
    }
}

void ResultsManager::printComparison() const {
    if (experiments.empty()) {
        std::cout << "No experiments to compare.\n";
        return;
    }
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "SUMMARY OF ALL EXPERIMENTS\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << "Total Experiments: " << experiments.size() << "\n";
    
    auto best = std::max_element(experiments.begin(), experiments.end(),
        [](const ExperimentResult& a, const ExperimentResult& b) {
            return a.mean_test_accuracy < b.mean_test_accuracy;
        });
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nBest Result:\n";
    std::cout << "  Run ID: " << best->run_id << "\n";
    std::cout << "  Architecture: ";
    for (size_t i = 0; i < best->network_structure.size(); i++) {
        std::cout << best->network_structure[i];
        if (i < best->network_structure.size() - 1) std::cout << "-";
    }
    std::cout << "\n  Test Accuracy: " << best->mean_test_accuracy * 100 << "%\n";
}

void ResultsManager::saveAllResults(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    
    file << std::fixed << std::setprecision(6);
    
    file << "Run_ID,Seed,Architecture,Fold,Train_Accuracy,Test_Accuracy,"
         << "Generations,Best_Fitness,"
         << "Train_TP,Train_TN,Train_FP,Train_FN,Train_Precision,Train_Recall,Train_F1,"
         << "Test_TP,Test_TN,Test_FP,Test_FN,Test_Precision,Test_Recall,Test_F1\n";
    
    for (const auto& exp : experiments) {
        std::string arch_str;
        for (size_t i = 0; i < exp.network_structure.size(); i++) {
            arch_str += std::to_string(exp.network_structure[i]);
            if (i < exp.network_structure.size() - 1) arch_str += "-";
        }
        
        for (const auto& fold : exp.fold_results) {
            file << exp.run_id << ","
                 << exp.seed << ","
                 << arch_str << ","
                 << fold.fold_number << ","
                 << fold.train_accuracy << ","
                 << fold.test_accuracy << ","
                 << fold.generations_used << ","
                 << fold.best_fitness << ","
                 << fold.train_metrics.true_positive << ","
                 << fold.train_metrics.true_negative << ","
                 << fold.train_metrics.false_positive << ","
                 << fold.train_metrics.false_negative << ","
                 << fold.train_metrics.precision << ","
                 << fold.train_metrics.recall << ","
                 << fold.train_metrics.f1_score << ","
                 << fold.test_metrics.true_positive << ","
                 << fold.test_metrics.true_negative << ","
                 << fold.test_metrics.false_positive << ","
                 << fold.test_metrics.false_negative << ","
                 << fold.test_metrics.precision << ","
                 << fold.test_metrics.recall << ","
                 << fold.test_metrics.f1_score << "\n";
        }
    }
    
    file.close();
    std::cout << "Detailed results saved to " << filename << std::endl;
}

void ResultsManager::saveSummaryResults(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    
    file << std::fixed << std::setprecision(6);
    file << "Run_ID,Seed,Architecture,Mean_Test_Accuracy,Std_Test_Accuracy,"
         << "Mean_Train_Accuracy,Std_Train_Accuracy,"
         << "Min_Test_Acc,Max_Test_Acc,Median_Test_Acc,"
         << "Mean_Precision,Mean_Recall,Mean_F1\n";
    
    for (const auto& exp : experiments) {
        std::string arch_str;
        for (size_t i = 0; i < exp.network_structure.size(); i++) {
            arch_str += std::to_string(exp.network_structure[i]);
            if (i < exp.network_structure.size() - 1) arch_str += "-";
        }
        
        std::vector<double> test_accs;
        double sum_precision = 0.0, sum_recall = 0.0, sum_f1 = 0.0;
        
        for (const auto& fold : exp.fold_results) {
            test_accs.push_back(fold.test_accuracy);
            sum_precision += fold.test_metrics.precision;
            sum_recall += fold.test_metrics.recall;
            sum_f1 += fold.test_metrics.f1_score;
        }
        
        std::sort(test_accs.begin(), test_accs.end());
        double min_acc = test_accs.front();
        double max_acc = test_accs.back();
        double median_acc = test_accs[test_accs.size() / 2];
        
        double mean_precision = sum_precision / exp.fold_results.size();
        double mean_recall = sum_recall / exp.fold_results.size();
        double mean_f1 = sum_f1 / exp.fold_results.size();
        
        file << exp.run_id << ","
             << exp.seed << ","
             << arch_str << ","
             << exp.mean_test_accuracy << ","
             << exp.std_test_accuracy << ","
             << exp.mean_train_accuracy << ","
             << exp.std_train_accuracy << ","
             << min_acc << ","
             << max_acc << ","
             << median_acc << ","
             << mean_precision << ","
             << mean_recall << ","
             << mean_f1 << "\n";
    }
    
    file.close();
    std::cout << "Summary results saved to " << filename << std::endl;
}