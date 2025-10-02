#include <iostream>
#include <vector>
#include <string>
#include "dataset.h"
#include "mlp.h"
#include "ga.h"
#include "utils.h"
#include "results.h"

void runExperiment(Dataset& dataset, 
                   const std::vector<int>& architecture,
                   ResultsManager& results_manager,
                   const GAConfig& ga_config) {
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "EXPERIMENT: Network Architecture ";
    for (size_t i = 0; i < architecture.size(); i++) {
        std::cout << architecture[i];
        if (i < architecture.size() - 1) std::cout << "-";
    }
    std::cout << "\n" << std::string(80, '=') << "\n";
    
    ExperimentResult exp_result;
    exp_result.network_structure = architecture;
    
    // 10-fold cross validation
    for (int fold = 0; fold < 10; fold++) {
        std::cout << "\n--- Fold " << (fold + 1) << "/10 ---\n";
        
        // Get train/test split
        std::vector<std::vector<double>> train_X, test_X;
        std::vector<int> train_y, test_y;
        dataset.getTrainTestSplit(fold, train_X, train_y, test_X, test_y);
        
        std::cout << "Train samples: " << train_X.size() 
                  << ", Test samples: " << test_X.size() << "\n";
        
        // Create MLP
        MLP mlp(architecture, ActivationType::SIGMOID);
        mlp.printStructure();
        
        // Create GA
        GeneticAlgorithm ga(mlp.getChromosomeLength(), ga_config);
        
        // Set fitness function
        auto fitness_func = createMLPFitnessFunction(mlp, train_X, train_y);
        ga.setFitnessFunction(fitness_func);
        
        // Train
        ga.evolve();
        
        // Set best weights
        mlp.setWeights(ga.getBestIndividual().chromosome);
        
        // Evaluate on train and test
        double train_acc = mlp.evaluateAccuracy(train_X, train_y);
        double test_acc = mlp.evaluateAccuracy(test_X, test_y);
        
        // Get predictions for metrics
        std::vector<int> train_pred, test_pred;
        for (const auto& x : train_X) {
            train_pred.push_back(mlp.predict(x));
        }
        for (const auto& x : test_X) {
            test_pred.push_back(mlp.predict(x));
        }
        
        // Calculate metrics
        auto train_metrics = Utils::calculateMetrics(train_pred, train_y);
        auto test_metrics = Utils::calculateMetrics(test_pred, test_y);
        
        std::cout << "\nTrain Accuracy: " << train_acc * 100 << "%\n";
        std::cout << "Test Accuracy:  " << test_acc * 100 << "%\n";
        std::cout << "Test Precision: " << test_metrics.precision * 100 << "%\n";
        std::cout << "Test Recall:    " << test_metrics.recall * 100 << "%\n";
        std::cout << "Test F1-Score:  " << test_metrics.f1_score << "\n";
        
        // Store fold result
        FoldResult fold_result;
        fold_result.fold_number = fold + 1;
        fold_result.train_accuracy = train_acc;
        fold_result.test_accuracy = test_acc;
        fold_result.train_metrics = train_metrics;
        fold_result.test_metrics = test_metrics;
        fold_result.generations_used = ga_config.max_generations;
        fold_result.best_fitness = ga.getBestFitness();
        
        exp_result.fold_results.push_back(fold_result);
    }
    
    // Calculate experiment statistics
    exp_result.calculate();
    
    // Add to results manager
    results_manager.addExperiment(exp_result);
    
    // Print experiment summary
    exp_result.print();
}

int main(int argc, char* argv[]) {
    std::cout << "======================================\n";
    std::cout << "MLP Training with Genetic Algorithm\n";
    std::cout << "Wisconsin Diagnostic Breast Cancer\n";
    std::cout << "======================================\n\n";
    
    // Initialize random seed
    Utils::initRandom(42);
    
    // Load dataset
    Dataset dataset;
    std::string filename = "data/wdbc.data";
    
    if (argc > 1) {
        filename = argv[1];
    }
    
    if (!dataset.loadFromFile(filename)) {
        std::cerr << "Failed to load dataset\n";
        return 1;
    }
    
    // Print dataset statistics
    dataset.printStatistics();
    
    // Normalize data
    dataset.normalize();
    
    // Create k-folds for cross-validation
    dataset.createKFolds(10, 42);
    
    // Configure GA
    GAConfig ga_config;
    ga_config.population_size = 50;
    ga_config.max_generations = 100;
    ga_config.crossover_rate = 0.8;
    ga_config.mutation_rate = 0.15;
    ga_config.mutation_strength = 0.3;
    ga_config.elitism_rate = 0.1;
    ga_config.tournament_size = 3;
    ga_config.verbose = true;
    
    std::cout << "\nGA Configuration:\n";
    std::cout << "  Population size: " << ga_config.population_size << "\n";
    std::cout << "  Max generations: " << ga_config.max_generations << "\n";
    std::cout << "  Crossover rate: " << ga_config.crossover_rate << "\n";
    std::cout << "  Mutation rate: " << ga_config.mutation_rate << "\n";
    std::cout << "  Elitism rate: " << ga_config.elitism_rate << "\n\n";
    
    // Results manager
    ResultsManager results_manager;
    
    // Define different network architectures to test
    std::vector<std::vector<int>> architectures = {
        {30, 10, 1},      // 1 hidden layer with 10 neurons
        {30, 20, 1},      // 1 hidden layer with 20 neurons
        {30, 15, 5, 1},   // 2 hidden layers with 15 and 5 neurons
        {30, 20, 10, 1},  // 2 hidden layers with 20 and 10 neurons
        {30, 30, 15, 1}   // 2 hidden layers with 30 and 15 neurons
    };
    
    // Run experiments for each architecture
    for (const auto& arch : architectures) {
        runExperiment(dataset, arch, results_manager, ga_config);
    }
    
    // Print comparison of all architectures
    results_manager.printComparison();
    
    // Save results to CSV files
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Saving results to CSV files...\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Save detailed results (all folds, all metrics)
    results_manager.saveAllResults("all_results.csv");
    
    // Save summary statistics
    results_manager.saveSummaryResults("results_summary.csv");
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "All experiments completed!\n";
    std::cout << "Output files:\n";
    std::cout << "  - all_results.csv      (detailed per-fold results)\n";
    std::cout << "  - results_summary.csv  (summary statistics)\n";
    std::cout << std::string(80, '=') << "\n";
    
    return 0;
}