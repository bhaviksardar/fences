class FencesError(Exception):
    """Base exception for all Fences policy violations."""
    pass


class BudgetExceeded(FencesError):
    """Raised when a governed run exceeds its allotted spend."""
    def __init__(self, spent_usd: float, budget_usd: float):
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"Run exceeded budget: spent ${spent_usd:.4f} of ${budget_usd:.4f} limit"
        )