import tensorflow as tf
import numpy as np
import random
from typing import Callable,List #specify the function type
​
class Neuron():
    def __init__(self,act_fn: Callable[[float],float],inputs:List):
        '''
        self.weights: vector of randomly initialized floats between 0 and 1
        self.inputs: List of vector of floats (all these vector of floats are the same size)
        self.bias: scalar
        self.activation: activation function
        '''
        #initialize the random weights
        # for i in range(len(inputs[0])):
        #     weights.append(random.random())
        self.__inputs = inputs
        weights = []
        
        for i in range(len(self.__inputs) + 1): # + 1 is for bias
            weights.append(random.random())
        self.__weights = np.asarray(weights).astype("float64")
        
        
    
        self.__activation = act_fn
        self.__delta = 0
        
​
    def get_weights(self) -> tf.Tensor:
        '''
        Returns the tensor of weights
        '''
        return list(self.__weights)
    
    
    # def get_bias(self) -> tf.Tensor: #specifying return type isnt rlly doing anything here
    #     '''
    #     Returns the scalar value of the bias
    #     '''
    #     return tf.convert_to_tensor(self.__bias)
​
    def get_delta(self):
        return float(self.__delta)
​
    def set_delta(self, delta):
        """
        Set delta error of neuron
        """        
        self.__delta = delta
    
    def set_weights(self,weights: tf.Variable):
        '''
        param: vector of weights
        '''
        self.__weights = weights
    
    def set_inputs(self, inputs: List[float]):
        '''
        param: vector of floats
        '''
        self.__inputs = tf.Variable(inputs)
​
    # def set_bias(self,bias: tf.Variable):
    #     '''
    #     param: scalar for bias
    #     '''
    #     self.__bias = bias
    
    def get_output(self):
        '''
        returns the dot product of the weights and inputs and adds the bias
        '''
​
        output = float(self.__activation(float(np.dot(self.__inputs, self.__weights[:-1])) + self.__weights[-1]))
        return output
​
    
​
class NeuralNetwork():
​
    def __init__(self,num_layers: int, n_p_layer:List[int] ,act_fns: List[Callable[[float],float]],  
                loss: Callable,inputs: List[float], expected: List[float]):
        '''
        num_layers: The total number of layers for neural network (not counting the input layer)
        n_p_layer: the number of neurons per layer
        act_fns: activation functions for each layer
        '''
​
        
        # network = list()
​
        # for layer in range(num_layers):
        #     hidden_layer = []
        #     for i in range(n_p_layer[layer] + 1):
        #         hidden_layer.append(random.random())
        #     network.append(hidden_layer)
​
​
        if len(n_p_layer) != num_layers:
            raise ValueError("num_layers does not equal length of n_p_layer")
        self.__outputs = list()
        self.network: List = []
        self.__num_layers = num_layers
        self.__originalInput = inputs
        self.__loss = loss
        self.__expected = expected
        self.__n_p_layer = n_p_layer
​
        temp = []
        for _ in range(self.__n_p_layer[0]): #create the number of neurons specified in that layer
            temp.append(Neuron(act_fn=act_fns[0],inputs=self.__originalInput) ) 
        self.network.append(temp)
        for i in range(1,self.__num_layers):
            temp = []
            self.__outputs.append(self.get_layer_output(i-1))
            for x in range(self.__n_p_layer[i]):
                temp.append(Neuron(act_fn=act_fns[i],inputs=self.get_layer_output(i-1)) )
            self.network.append(temp)
        self.__outputs.append(self.get_final_output())
​
        
        # self.__optim = optimizer
    
    def feed_forward(self):
        temp = []
        self.network = []
        self.__outputs = []
        for _ in range(self.__n_p_layer[0]): #create the number of neurons specified in that layer
            lyst = [0.13436424411240122, 0.8474337369372327, 0.763774618976614]
            a = Neuron(act_fn=act_fns[0],inputs=self.__originalInput)
            a.set_weights(lyst)
            temp.append(a)
        self.network.append(temp)
        for i in range(1,self.__num_layers):
            temp = []
            self.__outputs.append(self.get_layer_output(i-1))
            for x in range(self.__n_p_layer[i]):
                a = Neuron(act_fn=act_fns[i],inputs=self.get_layer_output(i-1))
                lyst = [[0.2550690257394217, 0.49543508709194095], [0.4494910647887381, 0.651592972722763]]
                a.set_weights(lyst[x])
                temp.append(a)
            self.network.append(temp)
        self.__outputs.append(self.get_final_output())
​
        print(self.__outputs)
​
    def backward_propagate_error(self):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network)-1: #hidden layer (second to last)
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]: #next layer
                        error += neuron.get_weights()[j] * neuron.get_delta() #both are next layer
                    errors.append(error)
            else: #output layer
                for j in range(len(layer)): #for each nueron
                    neuron = layer[j]
                    errors.append(self.__expected[j] - neuron.get_output())
            for j in range(len(layer)): #for each neuron
                neuron = layer[j]
                neuron.set_delta(errors[j] * self.transfer_derivative(neuron.get_output())) #current layer
​
    def transfer_derivative(self, number):
        return number * (1.0 - number)
​
    def update_weights(self):
​
        l_rate = 0.1
        for i in range(len(self.network)):
            inputs = self.__originalInput
            if i != 0:
                inputs = self.__outputs[i - 1]
            for neuron in network[i]:
                new_weights = []
                for j in range(len(inputs)):
                    new_weights.append(neuron.get_weights()[j] + l_rate * neuron.get_delta() * inputs[j])
                new_weights.append(neuron.get_weights()[-1] + l_rate * neuron.get_delta())
                neuron.set_weights(new_weights)
            
​
    def get_layer_output(self, index):
        '''
        param: index of the layer we wish to get the output of
        '''
        layer_output:List[List[float]] = []
        for neuron in self.network[index]:
            layer_output.append(neuron.get_output())
        return np.asarray(layer_output).astype("float64")
​
    def get_final_output(self):
        #print(self.__num_layers)
        return self.get_layer_output(self.__num_layers - 1)
    
    def get_neuron_output(self,layer_index,neuron_index):
        return self.get_layer_output(layer_index)[neuron_index]
    # @tf.function
    def train(self, data, all_y_trues):
        pass
​
​
​
if __name__ == "__main__":
    # n = Neuron(lambda x: 1.0/(1.0 + tf.math.exp(-x)))
    act_fns = [tf.nn.sigmoid,tf.nn.sigmoid,tf.nn.sigmoid]
    loss = tf.keras.losses.MeanSquaredError
    network = NeuralNetwork(2, [1,2], act_fns, loss, [1, 0], [0, 1])
​
    network.feed_forward()
​
    network.backward_propagate_error()
​
    for layer in network.network:
        for neuron in layer:
            print(neuron.get_weights())
            print(neuron.get_delta())
​
   
​
    # main_input:List = [3.1,5.4,6.3,7.1,8.32,9.9]
    # main_input = np.asarray(main_input).astype('float32')
    # n = Neuron(tf.nn.relu,main_input)
    # # print(f"type: {type(tf.math.sigmoid)}")
    # print(f"output: {n.get_output()}")
    
    
    # optim = tf.optimizers.Adam
    # network = NeuralNetwork(3,[32,32,1] ,act_fns,loss,main_input)
    # print(network.get_final_output())
    # #print(network.get_neuron_output(1,0))