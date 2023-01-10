class PredictionResponse:
    def __init__(self, prediction: int, confidence: float):
        self.prediction = prediction
        self.confidence = confidence
        self.prediction_str = 'Negative' if self.prediction == 0 else 'Positive'
        