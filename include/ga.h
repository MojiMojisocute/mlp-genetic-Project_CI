#ifndef GA_H
#define GA_H

#include <vector>
#include <functional>
#include "mlp.h"

struct Individual {
    std::vector<double> chromosome;
    double fitness;
    
    Individual() : fitness(0.0) {}
    Individual(int size) : chromosome(size), fitness(0.0) {}
};

struct GAConfig {
    int population_size;
    int max_generations;
    double crossover_rate;
    double mutation_rate;
    double mutation_strength;
    double elitism_rate;
    int tournament_size;
    bool verbose;
    
    // Default values
    GAConfig() 
        : population_size(50),
          max_generations(100),
          crossover_rate(0.8),
          mutation_rate(0.1),
          mutation_strength(0.3),
          elitism_rate(0.1),
          tournament_size(3),
          verbose(true) {}
};

class GeneticAlgorithm {
private:
    GAConfig config;
    std::vector<Individual> population;
    int chromosome_length;
    double best_fitness;
    Individual best_individual;
    
    // Fitness function (external)
    std::function<double(const std::vector<double>&)> fitness_function;
    
    // GA operations
    void initializePopulation(double min_val = -1.0, double max_val = 1.0);
    void evaluateFitness();
    Individual tournamentSelection();
    std::pair<Individual, Individual> crossover(const Individual& parent1, 
                                                const Individual& parent2);
    void mutate(Individual& individual);
    void replacePopulation(std::vector<Individual>& offspring);
    
    // Statistics
    std::vector<double> fitness_history;
    std::vector<double> best_fitness_history;
    std::vector<double> avg_fitness_history;
    
public:
    GeneticAlgorithm(int chrom_length, const GAConfig& cfg = GAConfig());
    ~GeneticAlgorithm();
    
    // Set fitness function
    void setFitnessFunction(std::function<double(const std::vector<double>&)> func);
    
    // Run GA
    void evolve();
    
    // Get results
    const Individual& getBestIndividual() const { return best_individual; }
    double getBestFitness() const { return best_fitness; }
    const std::vector<double>& getBestFitnessHistory() const { 
        return best_fitness_history; 
    }
    const std::vector<double>& getAvgFitnessHistory() const { 
        return avg_fitness_history; 
    }
    
    // Print statistics
    void printStatistics() const;
    void printGenerationStats(int generation) const;
};

// Helper function to create fitness function for MLP
std::function<double(const std::vector<double>&)> createMLPFitnessFunction(
    MLP& mlp,
    const std::vector<std::vector<double>>& X_train,
    const std::vector<int>& y_train
);

#endif // GA_H