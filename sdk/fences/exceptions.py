class FencesError(Exception):
    pass


class BudgetExceeded(FencesError):
    def __init__(self, spent_usd: float, budget_usd: float):
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"Run exceeded budget: spent ${spent_usd:.4f} of ${budget_usd:.4f} limit"
        )


class IterationLimitReached(FencesError):
    def __init__(self, iterations: int, max_iterations: int):
        self.iterations = iterations
        self.max_iterations = max_iterations
        super().__init__(
            f"Run exceeded iteration limit: {iterations} of {max_iterations} allowed"
        )


class TimeLimitReached(FencesError):
    def __init__(self, duration_ms: int, max_duration_ms: int):
        self.duration_ms = duration_ms
        self.max_duration_ms = max_duration_ms
        super().__init__(
            f"Run exceeded time limit: {duration_ms}ms of {max_duration_ms}ms allowed"
        )


class TokenLimitReached(FencesError):
    def __init__(self, tokens_used: int, max_tokens: int):
        self.tokens_used = tokens_used
        self.max_tokens = max_tokens
        super().__init__(
            f"Run exceeded token limit: {tokens_used} of {max_tokens} tokens allowed"
        )