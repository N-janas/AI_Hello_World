##TRAINING_SET######CORRECT_RESULTS##
#|  0  |  0  |  1  |  =  | 0 |      #
#|  1  |  1  |  1  |  =  | 1 |      #
#|  1  |  0  |  1  |  =  | 1 |      #
#|  0  |  1  |  1  |  =  | 0 |      #
#####################################
import numpy as np


class NeuralNetwork:
    def __init__(self, x, y):
        np.random.seed(1)
        self.inputs = x
        self.comp_outputs = np.zeros(y.shape)
        self.outputs = y
        # Weights shape ( number_of_inputs_to_neurons, number_of_neurons_wanted_in_next_layer )
        self.weights1 = np.random.rand(self.inputs.shape[1], 4)
        self.weights2 = np.random.rand(4,1)

    # Testing functions
    def get_computed_output(self):
        return self.comp_outputs

    def get_cost(self):
        return np.mean(np.square(self.outputs - self.comp_outputs))

    # Activation functions
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return x*(1-x)

    # Training mechanism
    def feed_forward(self):
        self.layer1 = self.sigmoid(np.dot(self.inputs, self.weights1))
        self.comp_outputs = self.sigmoid(np.dot(self.layer1, self.weights2))

    def back_propagation(self):
        # Part contains: Errors, sigmoids of output layer
        part_of_propagation = 2*(self.outputs - self.comp_outputs)*self.sigmoid_derivative(self.comp_outputs)
        # So part_of_prop is multiplied by every weight on synapse between lay1 and output layer
        grad_weights2 = np.dot(self.layer1.T, part_of_propagation)
        # d_weights1 contains: Every weight on synapse between input layer and lay1, sigmoid_derivative of sums on neurons in lay1,
        # (used formula where i can use neuron's values after sigmoid was applied), 
        # partial derivatives of sums in neurons in output layer with respect to neuron's value in lay1
        # (it means weights on synapses between output layer and lay1), part_of_prop 
        grad_weights1 = np.dot(self.inputs.T, np.dot(part_of_propagation, self.weights2.T)*self.sigmoid_derivative(self.layer1))

        self.weights1 += grad_weights1
        self.weights2 += grad_weights2

    def train(self):
        self.feed_forward()
        self.back_propagation()

    def think(self, problem):
        lay1 = self.sigmoid(np.dot(problem, self.weights1))
        result_prediction = self.sigmoid(np.dot(lay1, self.weights2))
        return result_prediction


def main():
    data = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=float)
    data_results = np.array([[0, 1, 1, 0]], dtype=float).T

    nn = NeuralNetwork(data, data_results)

    for i in range(2000):
        # if i%500 == 0: 
        #     print(f'\n[{i}] Predictions:\n', nn.get_computed_output())
        #     print(f'[{i}] Cost:', nn.get_cost())
        
        nn.train()

    # Test
    print('\nTest')
    print('Output1:', nn.think(np.array([[1,0,0]], dtype=float) ))
    print('Output2:', nn.think(np.array([[0,1,0]], dtype=float) ))


if __name__=='__main__':
    main()