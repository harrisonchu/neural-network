import numpy as np


class Layer:

    def __init__(self, activation_function, num_neurons, learning_rate, previous_layer=None, next_layer=None,
                 input_layer=False, output_layer=False, debug=False):

        self.debug = debug

        # Fundamental configurations that sets up the structure of the layer
        self.num_neurons = num_neurons
        self.previous_layer = previous_layer
        self.next_layer = next_layer
        self.is_output_layer = output_layer
        self.is_input_layer = input_layer
        self.learning_rate = learning_rate

        # Only hidden layers have activation functions
        if self._is_hidden_layer():
            self.activation_function = activation_function
        self.input_signals = None

        if self.is_output_layer and self.next_layer:
            raise Exception('Output layer can not have a next layer!')

        if self.is_input_layer and self.previous_layer:
            raise Exception('Input layer can not have a previous layer!')

        # The below are states that can be changed during forward or back propagation of the network
        self.input_signals = None

        # Only input and hidden layers have "synapses" and so only they have weight matrices
        if not self.is_output_layer:
            self.weight_matrix = np.random.rand(self.num_neurons, next_layer.num_neurons)

        self.derivative_activation_function_at_input_signals = None
        self.output_signals = None

        # Basically the derivative of the error function partial the input_signals.  This is what gets back propagated
        # See http://briandolhansky.com/blog/2013/9/27/artificial-neural-networks-backpropagation-part-4
        self.deltas = None
        self.debug_field = None

    def update_weights_with_deltas(self):
        updates = np.transpose(self.input_signals) @ self.next_layer.deltas
        updates *= self.learning_rate
        updates *= 1.0 / len(self.input_signals)
        self.weight_matrix -= updates

    def send_input_signals(self, input_signals):
        self.input_signals = input_signals
        if self.debug:
            print("received input signals = %s" % self.input_signals)

    def forward_propagate(self):
        # Send signals to next layer
        if self._is_hidden_layer():
            # Apply activation function to incoming signals
            self.output_signals = self.activation_function(self.input_signals)
            if self.debug:
                print("output_signals = %s" % self.output_signals)

            # Also apply the derivative of the activation function to incoming signals for the back propagation step
            self.derivative_activation_function_at_input_signals = self.activation_function(self.output_signals,
                                                                                            derivative=True)
            input_for_next_layer = self.output_signals @ self.weight_matrix
            self.next_layer.send_input_signals(input_for_next_layer)
            self.next_layer.forward_propagate()

        # Input layer simply takes the input and send them to the first hidden layer with some weights
        if self.is_input_layer:
            self.output_signals = self.input_signals
            input_for_next_layer = self.output_signals @ self.weight_matrix
            self.next_layer.send_input_signals(input_for_next_layer)
            self.next_layer.forward_propagate()

    def back_propagate(self, truth_set=None):
        # Do not back propagate to input_layer since it doesn't have activation functions
        if self.previous_layer.is_input_layer:
            return

        if self.is_output_layer:
            self.deltas = error_function(self.input_signals, truth_set, derivative=True)

        '''
        Here's the tricky part -- we are transposing the weight matrix because we are "going the other way".
        In forward propagation we are propagating our training data from the input layer to the output layer.
        In back propagation we are propagating our deltas from the output layer to the input layer.
        And so, consider the weight matrix in the forward propagation step, each column of that matrix represents
        How every neuron of that current layer N is going to send signals to one signal neuron to layer N+1.
        Conversely, every row of that same matrix represents how a single neuron of that layer is going to send
        a signal to every neuron to the layer N + 1.  And so thus, when we transpose the matrix, we are getting a
        new matrix whose columns now represent how each neuron of the layer N + 1 is going to send deltas from the
        layer N + 1 to individual neurons of the current layer N.
        '''

        transpose_of_weight_matrix = np.transpose(self.previous_layer.weight_matrix)
        self.previous_layer.deltas = self.previous_layer.derivative_activation_function_at_input_signals \
                                     * (self.deltas @ transpose_of_weight_matrix)

        self.previous_layer.back_propagate()

    def _is_hidden_layer(self):
        return not self.is_output_layer and not self.is_input_layer


def error_function(prediction, target, derivative=False):
    error = prediction - target
    if derivative:
        return error
    return 0.5 * error.dot(error)


def sigmoid(n, derivative=False):
    if derivative:
        return sigmoid(n) * (1 - sigmoid(n))
    return 1 / (1 + np.exp(-n))

