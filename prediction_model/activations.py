import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    @abstractmethod
    def function(self, tensor: np.ndarray):
        pass


    @abstractmethod
    def derivative(self, tensor: np.ndarray):
        pass


class ReLU(Activation):
    def function(self, tensor: np.ndarray) -> np.ndarray:
        return tensor * (tensor > 0)


    def derivative(self, tensor: np.ndarray) -> np.ndarray:
        return 1 * (tensor > 0)


class Tanh(Activation):
    def function(self, tensor: np.ndarray) -> np.ndarray:
        return np.tanh(tensor)


    def derivative(self, tensor: np.ndarray) -> np.ndarray:
        return 1 - self.function(tensor)**2


class Sigmoid(Activation):
    def function(self, tensor: np.ndarray) -> np.ndarray:
        return np.where(tensor >= 0, 1 / (1 + np.exp(-tensor)), np.exp(tensor) / (1 + np.exp(tensor)))


    def derivative(self, tensor: np.ndarray) -> np.ndarray:
        return self.function(tensor) * (1 - self.function(tensor))