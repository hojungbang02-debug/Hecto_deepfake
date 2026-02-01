class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",  # "min" or "max"
        verbose: bool = True
    ):
        assert mode in ["min", "max"]

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True

        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def step(self, score: float) -> bool:
        """
        Returns:
            True if training should stop
        """
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Improvement detected: {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
