#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include "dataset.h"
#include "mlp.h"
#include "ga.h"
#include "utils.h"
#include "results.h"

void runExperiment(Dataset& dataset, 
                   const std::vector<int>& architecture,
                   ResultsManager& results_manager,
                   const GAConfig& ga_config,
                   int run_id,
                   unsigned int seed) {
    
    ExperimentResult exp_result;
    exp_result.network_structure = architecture;
    exp_result.run_id = run_id;
    exp_result.seed = seed;
    
    for (int fold = 0; fold < 10; fold++) {
        std::vector<std::vector<double>> train_X, test_X;
        std::vector<int> train_y, test_y;
        dataset.getTrainTestSplit(fold, train_X, train_y, test_X, test_y);

        MLP mlp(architecture, ActivationType::SIGMOID);

        GeneticAlgorithm ga(mlp.getChromosomeLength(), ga_config);

        auto fitness_func = createMLPFitnessFunction(mlp, train_X, train_y);
        ga.setFitnessFunction(fitness_func);
        ga.evolve();

        mlp.setWeights(ga.getBestIndividual().chromosome);
        
        double train_acc = mlp.evaluateAccuracy(train_X, train_y);
        double test_acc = mlp.evaluateAccuracy(test_X, test_y);

        std::vector<int> train_pred, test_pred;
        for (const auto& x : train_X) {
            train_pred.push_back(mlp.predict(x));
        }
        for (const auto& x : test_X) {
            test_pred.push_back(mlp.predict(x));
        }
        
        auto train_metrics = Utils::calculateMetrics(train_pred, train_y);
        auto test_metrics = Utils::calculateMetrics(test_pred, test_y);

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

    exp_result.calculate();
    
    results_manager.addExperiment(exp_result);
}

int main(int argc, char* argv[]) {
    std::cout << "======================================\n";
    std::cout << "MLP Training with Genetic Algorithm\n";
    std::cout << "Wisconsin Diagnostic Breast Cancer\n";
    std::cout << "Large Scale Experiment\n";
    std::cout << "======================================\n\n";

    Utils::initRandom(42);

    Dataset dataset;
    std::string filename = "data/wdbc.data";
    
    if (argc > 1) {
        filename = argv[1];
    }
    
    if (!dataset.loadFromFile(filename)) {
        std::cerr << "Failed to load dataset\n";
        return 1;
    }
    
    dataset.printStatistics();
    
    dataset.normalize();

    GAConfig ga_config;
    ga_config.population_size = 50;
    ga_config.max_generations = 100;
    ga_config.crossover_rate = 0.8;
    ga_config.mutation_rate = 0.15;
    ga_config.mutation_strength = 0.3;
    ga_config.elitism_rate = 0.1;
    ga_config.tournament_size = 3;
    ga_config.verbose = false; 
    
    std::cout << "\nGA Configuration:\n";
    std::cout << "  Population size: " << ga_config.population_size << "\n";
    std::cout << "  Max generations: " << ga_config.max_generations << "\n";
    std::cout << "  Crossover rate: " << ga_config.crossover_rate << "\n";
    std::cout << "  Mutation rate: " << ga_config.mutation_rate << "\n";
    std::cout << "  Elitism rate: " << ga_config.elitism_rate << "\n\n";
    
    ResultsManager results_manager;
    
    const int NUM_RUNS = 100;

    std::vector<std::vector<int>> architectures = {
        // 1 hidden layer (neurons: 5-50)
        {30, 5, 1}, {30, 8, 1}, {30, 10, 1}, {30, 12, 1}, {30, 15, 1},
        {30, 18, 1}, {30, 20, 1}, {30, 25, 1}, {30, 30, 1}, {30, 40, 1}, {30, 50, 1},
        
        // 2 hidden layers
        {30, 20, 10, 1}, {30, 25, 15, 1}, {30, 30, 15, 1}, {30, 20, 5, 1},
        {30, 15, 10, 1}, {30, 15, 5, 1}, {30, 10, 5, 1}, {30, 25, 10, 1},
        {30, 30, 20, 1}, {30, 40, 20, 1}, {30, 25, 5, 1},
        
        // 3 hidden layers
        {30, 20, 15, 10, 1}, {30, 25, 20, 10, 1}, {30, 30, 20, 10, 1},
        {30, 20, 10, 5, 1}, {30, 15, 10, 5, 1}, {30, 25, 15, 5, 1},
        {30, 30, 15, 5, 1}, {30, 40, 20, 10, 1},
        
        // 4 hidden layers
        {30, 30, 20, 10, 5, 1}, {30, 25, 20, 15, 10, 1},
        {30, 20, 15, 10, 5, 1}, {30, 40, 30, 20, 10, 1}
    };
    
    int total_experiments = NUM_RUNS * architectures.size();
    int current_exp = 0;
    
    std::cout << "\nTotal Experiments: " << total_experiments << "\n";
    std::cout << "Expected Output Lines: " << (total_experiments * 10) << "\n\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < NUM_RUNS; run++) {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "RUN " << (run + 1) << "/" << NUM_RUNS << "\n";
        std::cout << std::string(80, '=') << "\n";
        
        unsigned int seed = 42 + run * 1000;
        Utils::initRandom(seed);

        dataset.createKFolds(10, seed);

        for (const auto& arch : architectures) {
            current_exp++;
            
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - start_time).count();
            
            double avg_time_per_exp = (current_exp > 0) ? 
                (double)elapsed / current_exp : 0;
            int remaining = total_experiments - current_exp;
            int eta_seconds = (int)(avg_time_per_exp * remaining);
            
            std::cout << "\rProgress: " << current_exp << "/" << total_experiments 
                      << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * current_exp / total_experiments) << "%)";
            std::cout << " | Elapsed: " << (elapsed / 60) << "m " << (elapsed % 60) << "s";
            std::cout << " | ETA: " << (eta_seconds / 60) << "m " 
                      << (eta_seconds % 60) << "s" << std::flush;
            
            runExperiment(dataset, arch, results_manager, ga_config, 
                         run + 1, seed);
        }
        
        std::cout << "\n";
        
        if ((run + 1) % 10 == 0) {
            std::string checkpoint_file = "checkpoint_run_" + 
                                         std::to_string(run + 1) + ".csv";
            results_manager.saveAllResults(checkpoint_file);
            std::cout << "Checkpoint saved: " << checkpoint_file << "\n";
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(
        end_time - start_time).count();
    
    std::cout << "\n\nTotal training time: " << total_duration << " minutes\n";
    
    results_manager.printComparison();
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Saving final results to CSV files...\n";
    std::cout << std::string(80, '=') << "\n";
    
    results_manager.saveAllResults("all_results_final.csv");
    results_manager.saveSummaryResults("results_summary_final.csv");
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "All experiments completed!\n";
    std::cout << "Total Experiments: " << results_manager.size() << "\n";
    std::cout << "Total Output Lines: " << (results_manager.size() * 10) << "\n";
    std::cout << "Output files:\n";
    std::cout << "  - all_results_final.csv      (detailed per-fold results)\n";
    std::cout << "  - results_summary_final.csv  (summary statistics)\n";
    std::cout << "  - checkpoint_run_*.csv       (intermediate checkpoints)\n";
    std::cout << std::string(80, '=') << "\n";
    
    return 0;
}