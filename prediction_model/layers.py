import numpy as np
import prediction_model.activations as act
import prediction_model.optimizers as opti

class Layer:
    def __init__(self, input_size: int, output_size: int, activation: act.Activation, optimizer: opti.Optimizer):
        self.Input_size: int = input_size
        self.Output_size: int = output_size
        self.Activation: act.Activation = activation
        self.Optimizer: opti.Optimizer = optimizer


    def build(self) -> None:
        self.Weights: np.ndarray = np.random.uniform(0., 10., (self.Input_size, self.Output_size)) * .1
        self.Bias: np.ndarray = np.random.uniform(0., 10., (1, self.Output_size)) * .1

        if isinstance(self.Optimizer, opti.MomentumGradientDescent):
            self.Previous_weights_gradient: np.ndarray = np.zeros_like(self.Weights)
            self.Previous_bias_gradient: np.ndarray = np.zeros_like(self.Bias)
        
        elif isinstance(self.Optimizer, opti.ADAM):
            self.M_weights: np.ndarray = 0
            self.V_weights: np.ndarray = 0
            self.M_bias: np.ndarray = 0
            self.V_bias: np.ndarray = 0


    def forward(self, input_tensor: np.ndarray) -> None:
        self.Input:np.ndarray = input_tensor

        assert self.Input.shape[1] == self.Input_size, "Data shape {} does not match given shape {}".format(self.Input.shape, self.Input_size)

        self.Transfer: np.ndarray = self.Input @ self.Weights + self.Bias
        self.Activated: np.ndarray = self.Activation.function(self.Transfer)


    def back(self, output_gradient: np.ndarray) -> None:
        self.Transfer_gradient: np.ndarray = self.Activation.derivative(self.Transfer) * output_gradient
        self.Weights_gradient: np.ndarray = self.Input.T @ self.Transfer_gradient
        self.Bias_gradient: np.ndarray = np.sum(self.Transfer_gradient, axis = 0)
        self.Input_gradient: np.ndarray = self.Transfer_gradient @ self.Weights.T


    def update(self) -> None:
        if isinstance(self.Optimizer, opti.GradientDescent):
            self.Optimizer: opti.GradientDescent = self.Optimizer
            self.Weights = self.Optimizer.build(self.Weights, self.Weights_gradient)
            self.Bias = self.Optimizer.build(self.Bias, self.Bias_gradient)
        
        elif isinstance(self.Optimizer, opti.MomentumGradientDescent):
            self.Optimizer: opti.MomentumGradientDescent = self.Optimizer
            self.Weights = self.Optimizer.build(self.Weights, self.Weights_gradient, self.Previous_weights_gradient)
            self.Bias = self.Optimizer.build(self.Bias, self.Bias_gradient, self.Previous_bias_gradient)

        elif isinstance(self.Optimizer, opti.ADAM):
            self.Optimizer: opti.ADAM = self.Optimizer
            self.Weights, self.M_weights, self.V_weights = self.Optimizer.build(self.Weights, self.Weights_gradient, self.M_weights, self.V_weights)
            self.Bias, self.M_bias, self.V_bias = self.Optimizer.build(self.Bias, self.Bias_gradient, self.M_bias, self.V_bias)


    def load_params(self, weights: np.ndarray, bias: np.ndarray) -> None:
        self.Weights = weights
        self.Bias = bias


    def __repr__(self):
        print("Input size: {}\nOutput size: {}\nActivation function: {}\nOptimizer: {}\n"
        .format(self.Input_size, self.Output_size,self.Activation.__class__.__name__, self.Optimizer.__class__.__name__))


class _OutputLayer(Layer):
    def __init__(self, input_size: int, output_size: int, optimizer: opti.Optimizer):
        super().__init__(input_size, output_size, act.Sigmoid(), optimizer)

    
    def Set_labels(self, labels: np.ndarray):
        self.Labels: np.ndarray = labels


    def back(self):
        output_gradient = (self.Activated - self.Labels) / self.Labels.shape[0]
        super().back(output_gradient)
