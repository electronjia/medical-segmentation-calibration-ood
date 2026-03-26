class EarlyStopping:
    def __init__(self, patience, min_delta, mode):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        is_best = False  # Add this flag

        if self.best_score is None:
            self.best_score = current_score
            is_best = True  # First time always best
        else:
            improvement = (current_score - self.best_score) if self.mode == 'max' else (self.best_score - current_score)

            if improvement < self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0
                is_best = True

        return self.early_stop, is_best
