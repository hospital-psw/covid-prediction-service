import numpy as np
from typing import Tuple
from abc import abstractmethod, ABC


class Optimizer(ABC):
    def __init__(self, eta: float) -> None:
        self.Eta = eta


    @abstractmethod
    def build(self, current: np.ndarray, gradient: np.ndarray):
        pass


class GradientDescent(Optimizer):
    def __init__(self, eta: float) -> None:
        super().__init__(eta)


    def build(self, current, gradient):
        return current - self.Eta*gradient    


class MomentumGradientDescent(Optimizer):
    def __init__(self, eta: float, gamma: float) -> None:
        super().__init__(eta)
        self.Gamma = gamma


    def build(self, current: np.ndarray, gradient: np.ndarray, prev_gradient: np.ndarray) -> np.ndarray:
        return current - (self.Eta*gradient + self.Gamma*prev_gradient)


class ADAM(Optimizer):
    def __init__(self, eta: float, omega1: float, omega2: float) -> None:
        super().__init__(eta)
        self.Omega1 = omega1
        self.Omega2 = omega2

    
    def build(self, current, gradient, prev_m, prev_v) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        new_m = self.Omega1 * prev_m + (1-self.Omega1) * gradient
        new_v = self.Omega2 * prev_v + (1-self.Omega2) * (gradient**2)

        mhat = new_m / (1 - self.Omega1)
        vhat = np.abs(new_v) / (1 - self.Omega2)

        new_param = current - (self.Eta / (vhat + 1e-8)**0.5) * mhat

        return new_param, new_m, new_v