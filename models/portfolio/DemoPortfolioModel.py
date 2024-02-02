import datetime
import typing

from AlgorithmImports import *


# Portfolio construction scaffolding class; basic method args.
class DemoPortfolioModel(PortfolioConstructionModel):

    def CreateTargets(self, algorithm: QCAlgorithm, insights: List[Insight]) -> List[PortfolioTarget]:
        self.algorithm = algorithm
        return super().CreateTargets(algorithm, insights)

    # Determines if the portfolio should rebalance based on the provided rebalancing func
    def IsRebalanceDue(self, insights: List[Insight], algorithmUtc: datetime) -> bool:
        isRebalanceDue = super().IsRebalanceDue(insights, algorithmUtc)
        if isRebalanceDue:
            self.algorithm.Debug("PORTFOLIO:: ============ Rebalance due! ============")
        else:
            self.algorithm.Debug("PORTFOLIO:: xxxxxxxxxxxx Rebalance NOT due xxxxxxxxxxxx")
        return super().IsRebalanceDue(insights, algorithmUtc)

    # Determines the target percent for each insight
    def DetermineTargetPercent(self, activeInsights: List[Insight]) -> Dict[Insight, float]:
        targets = {}
        for insight in activeInsights:
            self.algorithm.Debug("PORTFOLIO:: Created target for insight " + str(insight))
            quantity = 10
            if insight.Direction == InsightDirection.Down:
                quantity = -10
            targets[insight] = (PortfolioTarget(insight.Symbol, quantity))
        return targets

    # Gets the target insights to calculate a portfolio target percent for, they will be piped to DetermineTargetPercent()
    def GetTargetInsights(self) -> List[Insight]:
        return None
        # return super().GetTargetInsights()

    # Determine if the portfolio construction model should create a target for this insight
    def ShouldCreateTargetForInsight(self, insight: Insight) -> bool:
        return None
        # return super().ShouldCreateTargetForInsight(insight)

    # Security change details
    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        return None
        # super().OnSecuritiesChanged(algorithm, changes)