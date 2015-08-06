require_relative 'neural_net'


class TestCase
  attr_accessor :input
  attr_accessor :expectation
  def initialize input, expectation
    @input = input
    @expectation = expectation
  end

  def expectationMatched output
    return false if output.length != @expectation.length
    output.each_index do |i|
      return false if output[i] != @expectation[i]
    end
    return true
  end
end

def test
  net = NeuralNetwork.new([3,6,3,1])
  cases = []
  cases.push(TestCase.new([0,0,0],[0]))
  cases.push(TestCase.new([1,0,0],[0]))
  cases.push(TestCase.new([0,1,0],[0]))
  cases.push(TestCase.new([1,1,0],[1]))
  cases.push(TestCase.new([0,0,1],[0]))
  cases.push(TestCase.new([1,0,1],[1]))
  cases.push(TestCase.new([0,1,1],[1]))
  cases.push(TestCase.new([1,1,1],[1]))
  done = false
  trials = 0
  maxTrials = 1000
  highest = 0.0
  learningRate = 1
  puts net.toString
  retried = false
  while !done do
    learningRate = Random.rand()
    trials += 1
    count = 0.0
    trains = []
    cases.each do |c|
      net.run(c.input)
      # puts net.retrieveOutput
      if c.expectationMatched(net.retrieveOutputRounded)
        count += 1.0
      else
        trains.push(c)
        # net.train(c.expectation,learningRate)
      end
      # net.train(c.expectation,learningRate)
    end
    accuracy = (count/cases.length)
    if accuracy > highest
      highest = accuracy
      puts highest
    end
    break if count == cases.length
    cases.each do |c|
      net.train(c.expectation,learningRate)
    end
    if trials == maxTrials
      net = NeuralNetwork.new([3,6,3,1])
      trials = 0
      puts "Retry"
    end
  end
  puts trials
  puts net.toString
end

test
