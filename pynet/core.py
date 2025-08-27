import numpy as np

# Dense Layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0., bias_regularizer_l1=0., weight_regularizer_l2=0., bias_regularizer_l2=0.):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1 # lambda
        self.bias_regularizer_l1 = bias_regularizer_l1 # lambda
        self.weight_regularizer_l2 = weight_regularizer_l2 # lambda
        self.bias_regularizer_l2 = bias_regularizer_l2 # lambda

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, dL_dz):
        self.dL_dw = np.dot(self.inputs.T, dL_dz)
        self.dL_db = np.sum(dL_dz, axis=0, keepdims=True)
        self.dL_dX = np.dot(dL_dz, self.weights.T)

        if self.weight_regularizer_l1>0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dL_dw += self.weight_regularizer_l1 * dL1

        if self.bias_regularizer_l1>0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dL_db += self.bias_regularizer_l1 * dL1

        if self.weight_regularizer_l2>0:
            dL2 = 2 * self.weight_regularizer_l2 * self.weights
            self.dL_dw += dL2

        if self.bias_regularizer_l2>0:
            dL2 = 2 * self.bias_regularizer_l2 * self.biases
            self.dL_db += dL2

# ReLU Activation Function
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, self.inputs)

    def backward(self, dL_da):
        self.dL_dz = dL_da.copy()
        self.dL_dz[self.inputs <= 0] = 0

# Softmax Activation Function
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(inputs, axis=1, keepdims=True)
        self.outputs = probabilities

# General Loss and Categorical Cross Entropy Function
class Loss:
    def regularization_loss(self, layer):
        regularization_loss = 0
        
        # L1 Regularization
        if layer.weight_regularizer_l1>0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.bias_regularizer_l1>0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        # L2 Regularization
        if layer.weight_regularizer_l2>0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l2>0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss
    
    def calculate(self, y_pred, y_true):
        neg_log_likelihoods = self.forward(y_pred, y_true)
        avg_loss = np.mean(neg_log_likelihoods)
        return avg_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_predictions = y_pred_clipped[range(len(y_pred_clipped)), y_true]

        if len(y_true.shape) == 2:
            correct_predictions = np.sum(y_pred_clipped*y_true, axis=1)

        neg_log_likelihoods = -np.log(correct_predictions)
        return neg_log_likelihoods

# Combining the Softmax Activation Function and the Categorical Cross Entropy Function
class Actication_Softmax_Loss_Categorical_Cross_Entropy:
    def __init__(self):
        self.activation_softmax = Activation_Softmax()
        self.loss_function = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation_softmax.forward(inputs)
        self.softmax_outputs = self.activation_softmax.outputs
        print(self.softmax_outputs)
        # self.loss = self.loss_function.calculate(self.softmax_outputs, y_true)

        # return self.loss

    def backward(self, y_pred, y_true):
        self.no_of_batches = len(y_pred)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dL_dz = self.softmax_outputs.copy()
        self.dL_dz[range(self.no_of_batches), y_true] -= 1
        self.dL_dz = self.dL_dz/self.no_of_batches

# ADAM Optimizer
class Optimizer_ADAM:
    def __init__(self, learning_rate=.001, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate # alpha
        self.current_learning_rate = learning_rate
        self.decay = decay # learning rate decay factor
        self.epsilon = epsilon # To avoid division by zero
        self.beta_1 = beta_1 # momentum factor
        self.beta_2 = beta_2 # rho: cache memory decay rate
        self.epoch = 0

    def pre_update_params(self):
        self.current_learning_rate = self.learning_rate / (1. + (self.decay * self.epoch))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentum = self.beta_1*layer.weight_momentum + (1-self.beta_1)*layer.dL_dw
        layer.bias_momentum = self.beta_1*layer.bias_momentum + (1-self.beta_1)*layer.dL_db

        layer.weight_cache = self.beta_2*layer.weight_cache + (1-self.beta_2)*(layer.dL_dw**2)
        layer.bias_cache = self.beta_2*layer.bias_cache + (1-self.beta_2)*(layer.dL_db**2)

        layer.weights += -self.current_learning_rate*(layer.weight_momentum/(1-self.beta_1**(self.epoch+1)))/\
                                                   (np.sqrt(layer.weight_cache/(1-self.beta_2**(self.epoch+1))) + self.epsilon)
        layer.biases += -self.current_learning_rate*(layer.bias_momentum/(1-self.beta_1**(self.epoch+1)))/\
                                                   (np.sqrt(layer.bias_cache/(1-self.beta_2**(self.epoch+1))) + self.epsilon)

    def post_update_params(self):
        self.epoch += 1