#include "neural_net.hpp"

#ifdef DEBUG
  #include <iostream>
  using std::cout;
  using std::endl;
#endif

#include <cstdlib>
#include <cmath>
using std::rand;
using std::exp;

struct NeuralNet::Dendrite{
  int to; // the index of the dendrite in the next layer
  double strength;
};

struct NeuralNet::Neuron{
  vector<Dendrite> connections;
  double error;
  double activation;
  double input;
  double bias;
};


NeuralNet::NeuralNet(vector<int> layerSizes){
  // Allocate all needed base memory
  layers = layerSizes.size();
  network = new Neuron*[layers];
  layerSize = new int[layers];
  // allocate the specific layers
  for(int i = 0;i < layers;i++){
    layerSize[i] = layerSizes[i];
    network[i] = new Neuron[layerSizes[i]];
  }
  // connect the feed forward network
  // with random connection strengths
  for(int i = 0;i < layers-1;i++){
    for(int j = 0;j < layerSize[i];j++){
      network[i][j].bias = randomStrength();
      for(int k = 0;k < layerSize[i+1];k++){
        Dendrite temp;
        temp.strength = randomStrength();
        temp.to = k;
        network[i][j].connections.push_back(temp);
      }
    }
  }
}

NeuralNet::NeuralNet(const NeuralNet &copy){
  layers = copy.layers;
  network = new Neuron*[layers];
  for(int i = 0;i < layers;i++){
    layerSize[i] = copy.layerSize[i];
    network[i] = new Neuron[layerSize[i]];
  }
  for(int i = 0;i < layers-1;i++){
    for(int j = 0;j < layerSize[i];j++){
      network[i][j] = copy.network[i][j];
    }
  }
}

NeuralNet::~NeuralNet(){
  for(int i = 0;i < layers;i++){
    delete[] network[i];
  }
  delete[] network;
  network = 0;
  delete[] layerSize;
  layerSize = 0;
  layers = 0;
}

vector<double> NeuralNet::run(vector<double> input){
  clear(); //clears the network of any residual activation/error/input
  if(input.size() != layerSize[0])return vector<double>();
  for(int i = 0;i < layerSize[0];i++){
    network[0][i].input += input[i];
  }
  for(int i = 0;i < layers;i++){
    for(int j = 0;j < layerSize[i];j++){
      network[i][j].activation = activationFunction(network[i][j].input);
      for(int k = 0;k < network[i][j].connections.size();k++){
        int to = network[i][j].connections[k].to;
        network[i+1][to].input += network[i][j].connections[k].strength * network[i][j].activation;
      }
    }
  }
  vector<double> output = vector<double>();
  for(int i = 0;i < layerSize[layers-1];i++){
    output.push_back(network[layers-1][i].activation);
  }
  return output;
}

void NeuralNet::freezeNet(){
  freeze = true;
}

void NeuralNet::unfreezeNet(){
  freeze = false;
}

void NeuralNet::randomize(){
  for(int i = 0;i < layers;i++){
    for(int j = 0;j < layerSize[i];j++){
      network[i][j].bias = randomStrength();
      int   numConnections = network[i][j].connections.size();
      for(int k = 0;k < numConnections;k++){
        network[i][j].connections[k].strength = randomStrength();
      }
    }
  }
}

void NeuralNet::train(vector<double> input, vector<double> expected, double learningRate){
  if(freeze)return;
  vector<double> output = run(input);
  if(output.size() != expected.size())return;
  for(int i = 0;i < output.size();i++){
    double diff = expected[i] - output[i];
    network[layers-1][i].error = activationDerivative(network[layers-1][i].input) * diff;
  }
  for(int i = layers-2;i >= 0;i--){
    for(int j = 0;j < layerSize[i];j++){
      network[i][j].error = activationDerivative(network[i][j].input);
      double total = 0.0;
      for(int k = 0;k < network[i][j].connections.size();k++){
        int to = network[i][j].connections[k].to;
        total += network[i][j].connections[k].strength * network[i+1][to].error;
      }
      network[i][j].error = network[i][j].error * total;
    }
  }

  for(int i = 0;i < layers;i++){
    for(int j = 0;j < layerSize[i];j++){
      for(int k = 0;k < network[i][j].connections.size();k++){
        int to = network[i][j].connections[k].to;
        network[i][j].connections[k].strength += learningRate * network[i][j].activation * network[i+1][to].error;
      }
      network[i][j].bias += learningRate * network[i][j].error;
    }
  }
}

void NeuralNet::clear(){
  for(int i = 0;i < layers;i++){
    for(int j = 0;j < layerSize[i];j++){
      network[i][j].activation = 0;
      network[i][j].error = 0;
      network[i][j].input = network[i][j].bias;
    }
  }
}

double NeuralNet::activationFunction(double input){
  return 1.0/(1.0+exp(-1.0*input));
}

double NeuralNet::activationDerivative(double input){
  return activationFunction(input) * (1 - activationFunction(input)); // for sigmoid function(it's derivative)
}

double NeuralNet::randomStrength(){
  double temp = rand()%100000;
  return temp/100000;
}
