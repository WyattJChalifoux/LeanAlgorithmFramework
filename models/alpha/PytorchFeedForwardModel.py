from datetime import timedelta

from AlgorithmImports import *
import torch
from torch import nn
import joblib


class PytorchFeedForwardModel(AlphaModel):
    Name = "PytorchFeedForwardModel"
    symbols = []

    def __init__(self, algorithm: QCAlgorithm, starting_security_symbol):
        self.algorithm = algorithm
        training_length = 252 * 2
        self.algorithm.training_data = RollingWindow[float](training_length)
        self.symbols.append(self.algorithm.AddEquity(starting_security_symbol, Resolution.Daily).Symbol)
        history = self.algorithm.History[TradeBar](self.symbols[0], training_length, Resolution.Daily)
        for trade_bar in history:
            self.algorithm.training_data.Add(trade_bar.Close)

        if self.algorithm.ObjectStore.ContainsKey(self.Name + "model"):
            file_name = self.algorithm.ObjectStore.GetFilePath(self.Name + "model")
            self.algorithm.model = joblib.load(file_name)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.algorithm.model = NeuralNetwork().to(device)
            self.algorithm.Train(self.my_training_method)

        self.algorithm.Train(self.algorithm.DateRules.Every(DayOfWeek.Sunday),
                             self.algorithm.TimeRules.At(8, 0),
                             self.my_training_method)

    def get_features_and_labels(self, n_steps=5):
        close_prices = list(self.algorithm.training_data)[::-1]

        features = []
        labels = []
        for i in range(len(close_prices) - n_steps):
            features.append(close_prices[i:i + n_steps])
            labels.append(close_prices[i + n_steps])
        features = np.array(features)
        labels = np.array(labels)

        return features, labels

    def my_training_method(self):
        self.algorithm.Debug("Beginning training cycle")
        features, labels = self.get_features_and_labels()

        # Set the loss and optimization functions
        # In this example, use the mean squared error as the loss function and stochastic gradient descent as the optimizer
        loss_fn = nn.MSELoss()
        learning_rate = 0.001
        optimizer = torch.optim.SGD(self.algorithm.model.parameters(), lr=learning_rate)

        # Create a for-loop to train for preset number of epoch
        epochs = 5
        for t in range(epochs):
            # Create a for-loop to fit the model per batch
            for batch, (feature, label) in enumerate(zip(features, labels)):
                # Compute prediction and loss
                pred = self.algorithm.model(feature)
                real = torch.from_numpy(np.array(label).flatten()).float()
                loss = loss_fn(pred, real)

                # Perform backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.algorithm.Debug("Ending training cycle")

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:

        insights = []

        self.algorithm.Debug("Updating")

        for symbol in self.symbols:
            self.algorithm.Debug("Updating with symbol " + str(symbol))
            if symbol in data.Bars:
                self.algorithm.Debug("Test")
                self.algorithm.training_data.Add(data.Bars[symbol].Close)

                features, __ = self.get_features_and_labels()
                prediction = self.algorithm.model(features[-1].reshape(1, -1))
                prediction = float(prediction.detach().numpy()[-1])

                """
                        Initializes a new instance of the Insight class

                        :param symbol: The symbol this insight is for
                        :param period: The period over which the prediction will come true
                        :param type: The type of insight, price/volatility
                        :param direction: The predicted direction
                        :param magnitude: The predicted magnitude as a percentage change
                        :param confidence: The confidence in this insight
                        :param sourceModel: An identifier defining the model that generated this insight
                        :param weight: The portfolio weight of this insight
                        :param tag: The insight's tag containing additional information
                        """

                if prediction > data[symbol].Price:
                    insights.append(
                        Insight(symbol=symbol,
                                period=timedelta(days=1),
                                type=InsightType.Price,
                                direction=InsightDirection.Up,
                                magnitude=prediction / data[symbol].Price,
                                confidence=1,
                                sourceModel=self.Name,
                                weight=1,
                                tag=""
                                )
                    )
                elif prediction < data[symbol].Price:
                    insights.append(
                        Insight(symbol=symbol,
                                period=timedelta(days=1),
                                type=InsightType.Price,
                                direction=InsightDirection.Down,
                                magnitude=prediction / data[symbol].Price,
                                confidence=1,
                                sourceModel=self.Name,
                                weight=1,
                                tag=""
                                )
                    )

        return insights

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        for security in changes.AddedSecurities:
            self.symbols.append(security.Symbol)

        for security in changes.RemovedSecurities:
            if security.Symbol in self.symbols:
                self.symbols.remove(security.Symbol)


class NeuralNetwork(nn.Module):
    # Model Structure
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 128),  # input size, output size of the layer
            nn.ReLU(),  # Relu non-linear transformation
            nn.Linear(128, 128),
            nn.ReLU(),  # Relu non-linear transformation
            nn.Linear(128, 5),
            nn.ReLU(),
            nn.Linear(5, 1),  # Output size = 1 for regression
        )

    # Feed-forward training/prediction
    def forward(self, x):
        x = torch.from_numpy(x).float()  # Convert to tensor in type float
        result = self.linear_relu_stack(x)
        return result
