# region imports
from AlgorithmImports import *

from models.alpha.DemoAlphaModel import DemoAlphaModel
from models.alpha.PytorchFeedForwardModel import PytorchFeedForwardModel
from models.universe.DemoUniverseSelectionModel import DemoUniverseSelectionModel


# endregion

class AlgorithmFrameworkDemo(QCAlgorithm):

    def Initialize(self):
        self.UniverseSettings.Resolution = Resolution.Daily

        self.SetStartDate(2010, 10, 1)  # Set Start Date
        self.SetEndDate(2011, 10, 5)  # Set End Date
        self.SetCash(30000)  # Set Strategy Cash

        symbols = [
            Symbol.Create("SPY", SecurityType.Equity, Market.USA),
            Symbol.Create("AAPL", SecurityType.Equity, Market.USA),
            Symbol.Create("GOOG", SecurityType.Equity, Market.USA),
            Symbol.Create("USO", SecurityType.Equity, Market.USA),
            Symbol.Create("WMI", SecurityType.Equity, Market.USA),
            Symbol.Create("AAA", SecurityType.Equity, Market.USA),
            Symbol.Create("AIG", SecurityType.Equity, Market.USA),
            Symbol.Create("BAC", SecurityType.Equity, Market.USA),
            Symbol.Create("BNO", SecurityType.Equity, Market.USA),
            Symbol.Create("EEM", SecurityType.Equity, Market.USA),
            Symbol.Create("FB", SecurityType.Equity, Market.USA),
            Symbol.Create("FOXA", SecurityType.Equity, Market.USA),
            Symbol.Create("IBM", SecurityType.Equity, Market.USA),
            Symbol.Create("IWM", SecurityType.Equity, Market.USA),
            Symbol.Create("NWSA", SecurityType.Equity, Market.USA),
            Symbol.Create("QQQ", SecurityType.Equity, Market.USA),
            Symbol.Create("UW", SecurityType.Equity, Market.USA),
            Symbol.Create("WM", SecurityType.Equity, Market.USA),
            Symbol.Create("WMI", SecurityType.Equity, Market.USA),
        ]

        self.SetUniverseSelection(DemoUniverseSelectionModel(symbols))
        # self.AddAlpha(DemoAlphaModel())
        self.AddAlpha(PytorchFeedForwardModel(self, "SPY"))
        # DemoPortfolioModel is WIP and currently unused
        self.SetPortfolioConstruction(MeanVarianceOptimizationPortfolioConstructionModel())
        self.SetExecution(StandardDeviationExecutionModel(resolution=Resolution.Daily))
        self.AddRiskManagement(NullRiskManagementModel())

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status == OrderStatus.Filled:

            order_direction = "Buy"
            if orderEvent.Direction == OrderDirection.Sell:
                order_direction = "Sell"

            self.Debug("{order_direction} Stock: {symbol}".format(
                order_direction=order_direction,
                symbol=orderEvent.Symbol
            ))
