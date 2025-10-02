#include "ga.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

GeneticAlgorithm::GeneticAlgorithm(int chrom_length, const GAConfig& cfg)
    : config(cfg), chromosome_length(chrom_length), best_fitness(0.0) {
    
    population.resize(config.population_size);
    for (auto& ind : population) {
        ind.chromosome.resize(chromosome_length);
    }
}

GeneticAlgorithm::~GeneticAlgorithm() {}

void GeneticAlgorithm::setFitnessFunction(
    std::function<double(const std::vector<double>&)> func) {
    fitness_function = func;
}

void GeneticAlgorithm::initializePopulation(double min_val, double max_val) {
    for (auto& individual : population) {
        individual.chromosome = Utils::randomVector(chromosome_length, min_val, max_val);
        individual.fitness = 0.0;
    }
}

void GeneticAlgorithm::evaluateFitness() {
    for (auto& individual : population) {
        individual.fitness = fitness_function(individual.chromosome);
    }
    
    // Update best individual
    auto best_it = std::max_element(population.begin(), population.end(),
        [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });
    
    if (best_it->fitness > best_fitness) {
        best_fitness = best_it->fitness;
        best_individual = *best_it;
    }
}

Individual GeneticAlgorithm::tournamentSelection() {
    Individual best;
    best.fitness = -1.0;
    
    for (int i = 0; i < config.tournament_size; i++) {
        int idx = Utils::randomInt(0, population.size() - 1);
        if (population[idx].fitness > best.fitness) {
            best = population[idx];
        }
    }
    
    return best;
}

std::pair<Individual, Individual> GeneticAlgorithm::crossover(
    const Individual& parent1, const Individual& parent2) {
    
    Individual child1 = parent1;
    Individual child2 = parent2;
    
    if (Utils::randomDouble(0.0, 1.0) < config.crossover_rate) {
        // Uniform crossover
        for (int i = 0; i < chromosome_length; i++) {
            if (Utils::randomDouble(0.0, 1.0) < 0.5) {
                child1.chromosome[i] = parent2.chromosome[i];
                child2.chromosome[i] = parent1.chromosome[i];
            }
        }
    }
    
    return {child1, child2};
}

void GeneticAlgorithm::mutate(Individual& individual) {
    for (int i = 0; i < chromosome_length; i++) {
        if (Utils::randomDouble(0.0, 1.0) < config.mutation_rate) {
            // Gaussian mutation
            double noise = Utils::randomDouble(-config.mutation_strength, 
                                              config.mutation_strength);
            individual.chromosome[i] += noise;
            
            // Clamp to reasonable range
            individual.chromosome[i] = Utils::clamp(individual.chromosome[i], -5.0, 5.0);
        }
    }
}

void GeneticAlgorithm::replacePopulation(std::vector<Individual>& offspring) {
    // Sort by fitness (descending)
    std::sort(population.begin(), population.end(),
        [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });
    
    std::sort(offspring.begin(), offspring.end(),
        [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });
    
    // Elitism: keep top individuals
    int elites = static_cast<int>(config.population_size * config.elitism_rate);
    
    std::vector<Individual> new_population;
    new_population.reserve(config.population_size);
    
    // Add elites
    for (int i = 0; i < elites && i < static_cast<int>(population.size()); i++) {
        new_population.push_back(population[i]);
    }
    
    // Add offspring
    for (size_t i = 0; i < offspring.size() && 
         new_population.size() < static_cast<size_t>(config.population_size); i++) {
        new_population.push_back(offspring[i]);
    }
    
    population = new_population;
}

void GeneticAlgorithm::evolve() {
    if (!fitness_function) {
        throw std::runtime_error("Fitness function not set");
    }
    
    // Initialize
    initializePopulation();
    evaluateFitness();
    
    if (config.verbose) {
        std::cout << "\n=== Starting Genetic Algorithm ===\n";
        std::cout << "Population size: " << config.population_size << "\n";
        std::cout << "Max generations: " << config.max_generations << "\n";
        std::cout << "Chromosome length: " << chromosome_length << "\n\n";
    }
    
    // Evolution loop
    for (int gen = 0; gen < config.max_generations; gen++) {
        // Create offspring
        std::vector<Individual> offspring;
        offspring.reserve(config.population_size);
        
        while (offspring.size() < static_cast<size_t>(config.population_size)) {
            // Selection
            Individual parent1 = tournamentSelection();
            Individual parent2 = tournamentSelection();
            
            // Crossover
            auto [child1, child2] = crossover(parent1, parent2);
            
            // Mutation
            mutate(child1);
            mutate(child2);
            
            offspring.push_back(child1);
            if (offspring.size() < static_cast<size_t>(config.population_size)) {
                offspring.push_back(child2);
            }
        }
        
        // Evaluate offspring
        for (auto& individual : offspring) {
            individual.fitness = fitness_function(individual.chromosome);
        }
        
        // Replace population
        replacePopulation(offspring);
        
        // Update statistics
        double total_fitness = 0.0;
        for (const auto& ind : population) {
            total_fitness += ind.fitness;
        }
        double avg_fitness = total_fitness / population.size();
        
        best_fitness_history.push_back(best_fitness);
        avg_fitness_history.push_back(avg_fitness);
        
        // Print progress
        if (config.verbose) {
            if (gen % 10 == 0 || gen == config.max_generations - 1) {
                printGenerationStats(gen);
            }
        }
    }
    
    if (config.verbose) {
        std::cout << "\n=== Evolution Complete ===\n";
        printStatistics();
    }
}

void GeneticAlgorithm::printGenerationStats(int generation) const {
    double avg_fitness = avg_fitness_history.back();
    
    std::cout << "Gen " << std::setw(4) << generation 
              << " | Best: " << std::fixed << std::setprecision(4) << best_fitness
              << " | Avg: " << avg_fitness << "\n";
}

void GeneticAlgorithm::printStatistics() const {
    std::cout << "\nFinal Statistics:\n";
    std::cout << "  Best Fitness: " << std::fixed << std::setprecision(4) 
              << best_fitness << "\n";
    std::cout << "  Generations: " << best_fitness_history.size() << "\n";
}

std::function<double(const std::vector<double>&)> createMLPFitnessFunction(
    MLP& mlp,
    const std::vector<std::vector<double>>& X_train,
    const std::vector<int>& y_train
) {
    return [&mlp, &X_train, &y_train](const std::vector<double>& chromosome) {
        mlp.setWeights(chromosome);
        return mlp.evaluateAccuracy(X_train, y_train);
    };
}