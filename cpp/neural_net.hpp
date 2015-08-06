/*FILE: neural_net.hpp
 *Author: Michael Huelsman
 *Start Date: 4 Aug 2015 12:35 EDT
 *Last Revision:
 *Notes:
 *
*/

#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
using std::vector;

class NeuralNet{
public:
  NeuralNet(vector<int> layerSizes);
  NeuralNet(const NeuralNet &copy);
  ~NeuralNet();

  vector<double> run(vector<double> input);

  void freezeNet(); //prevents further learning from taking place(even if called)
  void unfreezeNet(); //allows for further learning of the network
  void randomize();

  void train(vector<double> input, vector<double> expected, double learningRate); //performs training based on the expected output
protected:
  struct Dendrite;
  struct Neuron;

  void clear();

  double activationFunction(double input);
  double activationDerivative(double input);

  double randomStrength();
private:
  Neuron **network;
  int layers;
  int *layerSize;
  bool freeze;
};

#endif
