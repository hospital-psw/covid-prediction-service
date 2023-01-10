import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pickle
from typing import List
from time import time

import prediction_model.layers as l
import prediction_model.activations as act


class Model:
    def __init__(self, input_data: np.ndarray, labels: np.ndarray, epochs: int, batch_size: int) -> None:
        self.Input_data: np.ndarray = input_data
        self.Labels: np.ndarray = labels
        self.Epochs: int = epochs
        self.Batch_size: int = batch_size
        
        self.Layers: List[l.Layer] = []
        self.Loss: List[float] = []


    def add_layer(self, layer: l.Layer) -> None:
        self.Layers.append(layer)


    def remove_layer(self, index: int) -> None:
        self.Layers.pop(index)
    

    def model_information(self):
        for i, layer in enumerate(self.Layers):
            print(f"Layer {i}")
            layer.__repr__()
    

    def __transform_output_layer(self) -> None:
        output_layer = self.Layers[-1]

        if not isinstance(output_layer.Activation, act.Sigmoid):
            print("INFO: Output layer activation function not Sigmoid, changing to Sigmoid automaticly")

        self.Layers[-1] = l._OutputLayer(output_layer.Input_size, output_layer.Output_size, output_layer.Optimizer)


    def __calculate_loss(self, labels: np.ndarray) -> float:
        return 1 / (2*labels.shape[0]) * np.sum((labels - self.Layers[-1].Activated)**2)


    def __batch(self) -> None:
        indexes: List[int] = list(range(self.Input_data.shape[0]))
        np.random.shuffle(indexes)

        _input = self.Input_data[indexes]
        _labels = self.Labels[indexes]

        self.Input_batches: List[np.ndarray] = []
        self.Label_batches: List[np.ndarray] = []

        for i in range(0, len(_input), self.Batch_size):
            self.Input_batches.append(_input[i:i+self.Batch_size])
            self.Label_batches.append(_labels[i:i+self.Batch_size])


    def __forward_all(self, inputs: np.ndarray) -> None:
        self.Layers[0].forward(inputs)
        for i in range(1, len(self.Layers)):
            self.Layers[i].forward(self.Layers[i-1].Activated)

    
    def __back_all(self) -> None:
        self.Layers[-1].back()
        for i in range(len(self.Layers)-2, -1, -1):
            self.Layers[i].back(self.Layers[i+1].Input_gradient)

    
    def __update_all(self) -> None:
        for layer in self.Layers:
            layer.update()


    def __epoch(self) -> float:
        self.__batch()
        batch_Loss = []

        for inputs, labels in zip(self.Input_batches, self.Label_batches):
            # set labels to output for this batch
            self.Layers[-1].Set_labels(labels)
    
            self.__forward_all(inputs)
            batch_Loss.append(self.__calculate_loss(labels))
            self.__back_all()
            self.__update_all()

        return sum(batch_Loss)


    def train(self) -> None:
        self.__transform_output_layer()

        for layer in self.Layers:
            layer.build()
            self.Layers_built = True
        
        start = time()
        for i in tqdm.tqdm(range(self.Epochs)):
            batch_Loss = self.__epoch()
            self.Loss.append(batch_Loss)
        
        self.Train_time = time() - start

  
    def predict(self, datapoint: np.ndarray) -> int:
        assert datapoint.shape[0] == 1, "Input must be of shape (1, n)"
        assert self.Layers_built, "Network not trained yet, run obj.train() first"

        self.__forward_all(datapoint)
        res = 1 if self.Layers[-1].Activated >= 0.5 else 0
        print(f"Class for datapoint\n{datapoint}\nis {res}")

        return res


    def score(self, test_inputs: np.ndarray, test_labels: np.ndarray) -> None:
        self.__forward_all(test_inputs)

        true_nb: int = test_labels[test_labels.ravel() == 1].shape[0]
        false_nb: int = test_labels[test_labels.ravel() == 0].shape[0]

        predictions: np.ndarray = np.where(self.Layers[-1].Activated >= 0.5, 1, 0)
        predictions = predictions == test_labels
        true_positive_nb: int = int(sum( (predictions == True) & (test_labels == 1) ))
        false_negative_nb: int = int(sum( (predictions == False) & (test_labels == 1) ))
        true_negative_nb: int = int(sum( (predictions == True) & (test_labels == 0) ))
        false_positive_nb: int = int(sum( (predictions == False) & (test_labels == 0) ))

        print()
        print(f"{'True Positive':15}{'False Positive':15}{'Total Positive':15}\n{true_positive_nb:<15}{false_positive_nb:<15}{true_nb:<15}")
        print("-"*45)
        print(f"{'True Negative':15}{'False Negative':15}{'Total Negative':15}\n{true_negative_nb:<15}{false_negative_nb:<15}{false_nb:<15}")
        print()

        accuracy = float(sum(predictions) / len(predictions) * 100)
        precision = true_positive_nb / (true_positive_nb + false_positive_nb)
        recall = true_positive_nb / (true_positive_nb + false_negative_nb)
        f1 = 2 * precision * recall / (precision + recall)

        print(f"Accuracy: {accuracy}% out of 100%")
        print(f"Precision {precision} out of 1.0")
        print(f"Recall {recall} out of 1.0")
        print(f"F1 Score {f1} out of 1.0")
        print(f"Training time {self.Train_time:.2}s")


    def plot_loss(self) -> None:
        plt.plot(self.Loss, 'r-')
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

    
def _serialize_model(model: Model, path: str) -> None:
    with open(path, "wb") as out:
        pickle.dump(model, out)


def _deserialize_model(model_path: str) -> Model:
    with open(model_path, "rb") as inp:
        model = pickle.load(inp)
    
    return model