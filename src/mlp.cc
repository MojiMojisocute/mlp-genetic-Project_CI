#include "mlp.h"

MLP::MLP(const std::vector<int>& layers, ActivationType act_type) 
    : layer_sizes(layers), activation_type(act_type), total_params(0) {
    
    if (layers.size() < 2) {
        throw std::invalid_argument("Network must have at least input and output layers");
    }
    
    // Calculate total number of parameters
    for (size_t i = 0; i < layers.size() - 1; i++) {
        total_params += layers[i] * layers[i + 1];  // weights
        total_params += layers[i + 1];  // biases
    }
    
    // Initialize weight and bias structures
    weights.resize(layers.size() - 1);
    biases.resize(layers.size() - 1);
    layer_outputs.resize(layers.size());
    
    for (size_t i = 0; i < layers.size() - 1; i++) {
        weights[i].resize(layers[i]);
        for (int j = 0; j < layers[i]; j++) {
            weights[i][j].resize(layers[i + 1], 0.0);
        }
        biases[i].resize(layers[i + 1], 0.0);
    }
    
    // Initialize layer outputs
    for (size_t i = 0; i < layers.size(); i++) {
        layer_outputs[i].resize(layers[i], 0.0);
    }
}

MLP::~MLP() {}

double MLP::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double MLP::tanh_activation(double x) const {
    return std::tanh(x);
}

double MLP::relu(double x) const {
    return std::max(0.0, x);
}

double MLP::activate(double x) const {
    switch (activation_type) {
        case ActivationType::SIGMOID:
            return sigmoid(x);
        case ActivationType::TANH:
            return tanh_activation(x);
        case ActivationType::RELU:
            return relu(x);
        default:
            return sigmoid(x);
    }
}

void MLP::randomInitialize(double min_val, double max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_val, max_val);
    
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < weights[i].size(); j++) {
            for (size_t k = 0; k < weights[i][j].size(); k++) {
                weights[i][j][k] = dis(gen);
            }
        }
        for (size_t j = 0; j < biases[i].size(); j++) {
            biases[i][j] = dis(gen);
        }
    }
}

std::vector<double> MLP::encodeChromosome() const {
    std::vector<double> chromosome;
    chromosome.reserve(total_params);
    
    // Encode all weights and biases into a flat vector
    for (size_t i = 0; i < weights.size(); i++) {
        // Encode weights
        for (size_t j = 0; j < weights[i].size(); j++) {
            for (size_t k = 0; k < weights[i][j].size(); k++) {
                chromosome.push_back(weights[i][j][k]);
            }
        }
        // Encode biases
        for (size_t j = 0; j < biases[i].size(); j++) {
            chromosome.push_back(biases[i][j]);
        }
    }
    
    return chromosome;
}

void MLP::decodeChromosome(const std::vector<double>& chromosome) {
    if (chromosome.size() != static_cast<size_t>(total_params)) {
        throw std::invalid_argument("Chromosome size mismatch");
    }
    
    int idx = 0;
    
    // Decode weights and biases from flat vector
    for (size_t i = 0; i < weights.size(); i++) {
        // Decode weights
        for (size_t j = 0; j < weights[i].size(); j++) {
            for (size_t k = 0; k < weights[i][j].size(); k++) {
                weights[i][j][k] = chromosome[idx++];
            }
        }
        // Decode biases
        for (size_t j = 0; j < biases[i].size(); j++) {
            biases[i][j] = chromosome[idx++];
        }
    }
}

void MLP::setWeights(const std::vector<double>& chromosome) {
    decodeChromosome(chromosome);
}

std::vector<double> MLP::forward(const std::vector<double>& input) {
    if (input.size() != static_cast<size_t>(layer_sizes[0])) {
        throw std::invalid_argument("Input size mismatch");
    }
    
    // Set input layer
    layer_outputs[0] = input;
    
    // Forward propagation through each layer
    for (size_t layer = 0; layer < weights.size(); layer++) {
        for (size_t j = 0; j < layer_outputs[layer + 1].size(); j++) {
            double sum = biases[layer][j];
            
            for (size_t i = 0; i < layer_outputs[layer].size(); i++) {
                sum += layer_outputs[layer][i] * weights[layer][i][j];
            }
            
            // Apply activation function
            // Use sigmoid for output layer (binary classification)
            if (layer == weights.size() - 1) {
                layer_outputs[layer + 1][j] = sigmoid(sum);
            } else {
                layer_outputs[layer + 1][j] = activate(sum);
            }
        }
    }
    
    return layer_outputs.back();
}

int MLP::predict(const std::vector<double>& input) {
    std::vector<double> output = forward(input);
    
    // For binary classification with single output neuron
    if (output.size() == 1) {
        return output[0] >= 0.5 ? 1 : 0;
    }
    
    // For multi-output (softmax-like)
    int max_idx = 0;
    for (size_t i = 1; i < output.size(); i++) {
        if (output[i] > output[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

double MLP::evaluateAccuracy(const std::vector<std::vector<double>>& X,
                             const std::vector<int>& y) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y size mismatch");
    }
    
    int correct = 0;
    for (size_t i = 0; i < X.size(); i++) {
        int pred = predict(X[i]);
        if (pred == y[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / X.size();
}

void MLP::printStructure() const {
    std::cout << "MLP Structure: ";
    for (size_t i = 0; i < layer_sizes.size(); i++) {
        std::cout << layer_sizes[i];
        if (i < layer_sizes.size() - 1) {
            std::cout << " -> ";
        }
    }
    std::cout << std::endl;
    std::cout << "Total parameters: " << total_params << std::endl;
    std::cout << "Activation: ";
    switch (activation_type) {
        case ActivationType::SIGMOID:
            std::cout << "Sigmoid";
            break;
        case ActivationType::TANH:
            std::cout << "Tanh";
            break;
        case ActivationType::RELU:
            std::cout << "ReLU";
            break;
    }
    std::cout << std::endl;
}