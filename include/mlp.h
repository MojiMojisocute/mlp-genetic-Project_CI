#ifndef MLP_H
#define MLP_H

#include "utils.h"

#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <string>

enum class ActivationType {
    SIGMOID,
    TANH,
    RELU
};

class MLP {
private:
    std::vector<int> layer_sizes;  // [input, hidden1, hidden2, ..., output]
    ActivationType activation_type;
    int total_params;  // Total number of weights and biases
    
    // Network parameters (weights and biases for each layer)
    std::vector<std::vector<std::vector<double>>> weights;  // [layer][from][to]
    std::vector<std::vector<double>> biases;  // [layer][neuron]
    
    // For forward pass
    std::vector<std::vector<double>> layer_outputs;  // Store outputs of each layer
    
    // Activation functions
    double sigmoid(double x) const;
    double tanh_activation(double x) const;
    double relu(double x) const;
    double activate(double x) const;
    
    // Helper for chromosome conversion
    void decodeChromosome(const std::vector<double>& chromosome);
    
public:
    MLP(const std::vector<int>& layers, ActivationType act_type = ActivationType::SIGMOID);
    ~MLP();
    
    // Get chromosome length (number of parameters)
    int getChromosomeLength() const { return total_params; }
    
    // Convert weights/biases to chromosome (flat vector)
    std::vector<double> encodeChromosome() const;
    
    // Set weights/biases from chromosome
    void setWeights(const std::vector<double>& chromosome);
    
    // Forward pass - returns output layer activations
    std::vector<double> forward(const std::vector<double>& input);
    
    // Predict class (0 or 1)
    int predict(const std::vector<double>& input);
    
    // Evaluate accuracy on dataset
    double evaluateAccuracy(const std::vector<std::vector<double>>& X,
                           const std::vector<int>& y);
    
    // Get network structure info
    const std::vector<int>& getLayerSizes() const { return layer_sizes; }
    int getNumLayers() const { return layer_sizes.size(); }
    
    // Random initialization
    void randomInitialize(double min_val = -1.0, double max_val = 1.0);
    
    // Print network structure
    void printStructure() const;
};

#endif // MLP_H