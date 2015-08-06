class Neuron
  attr_reader :activation
  attr_accessor :error
  attr_accessor :connections
  attr_accessor :bias
  def initialize
    @inputs = []
    @connections = []
    @bias = 0.0
    @error = 0.0
  end

  def addInput input
    @inputs.push(input)
  end

  def calcActivation
    @activation = bias
    @inputs.each do |i|
      @activation += i
    end
    # sigmoid
    @activation = 1/(1+(Math.exp(-activation))) #Run through sigmoid function
    # hyperbolic
    # @activation = Math.tanh(activation)
  end

  def reset
    @activation = 0.0
    @inputs = []
  end

  def addConnection to, weight
    @connections.push([to,weight])
  end

  def fire
    @connections.each do |connection|
      connection[0].addInput(connection[1]*@activation)
    end
  end

  def getTotalInput
    total = @bias
    @inputs.each do |i|
      total += i
    end
    return total
  end
end


class NeuralNetwork
  # Precond:
  #   layers is an array of integers representing the size of the networks layers, in order
  # Postcond:
  #   sets up an untrained, random neural network
  def initialize layers, precision=10
    @network = []
    layers.each do |layer|
      @network.push([])
      layer.times do
        @network[-1].push(Neuron.new)
      end
    end
    @precision = precision
    # random connection weights
    @network.each_index do |i|
      if i != (@network.size - 1)
        @network[i].each do |sender|
          @network[i+1].each do |receiver|
            sender.addConnection(receiver,Random.rand().round(@precision))
            # sender.addConnection(receiver,0.0)
          end
        end
      end
    end
  end

  def run inputs
    clearNet
    @network[0].each_index do |i|
      @network[0][i].addInput(inputs[i])
    end
    @network.each do |layer|
      layer.each do |neuron|
        neuron.calcActivation
        neuron.fire
      end
    end
  end

  def retrieveOutputRounded
    output = []
    @network[-1].each do |neuron|
      output.push(neuron.activation.round)
    end
    return output
  end

  def retrieveOutputRaw
    output = []
    @network[-1].each do |neuron|
      output.push(neuron.activation)
    end
    return output
  end

  def train expected, lr
    # Calculate all the error factors
    layer = @network[-1]
    # output layer error
    layer.each_index do |i|
      layer[i].error = backpropFunction(layer[i].getTotalInput())*(expected[i]-layer[i].activation)
    end
    layers = @network[0,@network.length-1]
    layers.reverse!
    # reverse order inner and input layers error
    layers.each do |layer|
      layer.each do |neuron|
        neuron.error = backpropFunction(neuron.getTotalInput())
        total = 0.0
        neuron.connections.each do |connection|
          total += connection[1] * connection[0].error
        end
        neuron.error = neuron.error * total
      end
    end
    # Add to weights
    @network.each do |layer|
      layer.each do |neuron|
        next if neuron.connections == []
        neuron.connections.each do |connection|
          connection[1] += lr*neuron.activation*connection[0].error
        end
        neuron.bias += lr*neuron.error
      end
    end
  end

  def toString
    result = ""
    @network.each_index do |layer|
      @network[layer].each_index do |neuron|
        @network[layer][neuron].connections.each_index do |connect|
          result += "#{layer}->#{neuron}->#{connect}:#{@network[layer][neuron].connections[connect][1]}"
          result += "\n"
        end
      end
    end
    return result
  end

  private

  def backpropFunction totalInput
    # Sigmoid
    return (Math.exp(totalInput)/((1+Math.exp(totalInput))**2))
    # Hyperbolic
    # return 1-(Math.tanh(totalInput)**2)
  end

  def clearNet
    @network.each do |layer|
      layer.each do |neuron|
        neuron.reset
      end
    end
  end
end
