#include "neural_net.hpp"
#include <cmath>

#include <iostream>
using std::cout;
using std::endl;
using std::round;

#include <cstdlib>
#include <ctime>
using std::srand;
using std::time;

#define MAJORITY_CASES 8

struct TestCase{
  vector<double> input;
  vector<double> expectation;
};

void setupMajorityTest(TestCase majority[MAJORITY_CASES]){
  int count = 0;
  for(int i = 0;i < 2;i++){
    for(int j = 0;j < 2;j++){
      for(int k = 0;k < 2;k++){
        int oneCount = 0;
        majority[count].input = vector<double>();
        majority[count].expectation = vector<double>();
        majority[count].input.push_back(i);
        majority[count].input.push_back(j);
        majority[count].input.push_back(k);
        if(i==1)oneCount++;
        if(j==1)oneCount++;
        if(k==1)oneCount++;
        if(oneCount >= 2){
          majority[count].expectation.push_back(1);
        }
        else{
          majority[count].expectation.push_back(0);
        }
        count++;
      }
    }
  }
}

bool match(vector<double> output, vector<double> expected){
  if(output.size() != expected.size())return false;
  for(int i = 0;i < output.size();i++){
    if(round(output[i]) != expected[i])return false;
  }
  return true;
}

int main(int argc, char **argv){
  TestCase majority[MAJORITY_CASES];
  setupMajorityTest(majority);
  srand(time(0));
  vector<int> layerSizes = vector<int>();
  layerSizes.push_back(3);
  layerSizes.push_back(6);
  layerSizes.push_back(3);
  layerSizes.push_back(1);
  NeuralNet brain = NeuralNet(layerSizes);
  int iteration = 0;
  int correct = 0;
  int highestCorrect = 0;
  while(correct < MAJORITY_CASES){
    correct = 0;
    iteration++;
    if(iteration%10000 == 0){
      cout << "Randomize" <<endl;
      brain.randomize();
    }
    // Train
    for(int i = 0;i < MAJORITY_CASES;i++){
      if(!match(brain.run(majority[i].input), majority[i].expectation)){
        brain.train(majority[i].input,majority[i].expectation,0.01);
      }
    }
    // Check
    for(int i = 0;i < MAJORITY_CASES;i++){
      if(match(brain.run(majority[i].input), majority[i].expectation))correct++;
    }
    double performance = correct;
    performance = performance/MAJORITY_CASES;
    if(correct > highestCorrect){
      cout << "Iteration:" << iteration <<endl;
      highestCorrect = correct;
      cout << "Number Correct:" << correct <<endl;
      cout << "Percentage Correct:" << performance <<endl;
    }
  }
  cout << "Iteration:" << iteration <<endl;
  return 0;
}
