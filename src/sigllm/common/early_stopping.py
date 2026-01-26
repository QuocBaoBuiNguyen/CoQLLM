class EarlyStopping:
    def __init__(self, ref_metric='valid_auc', monitor_mode='max', patience=20):
        self.ref_metric = ref_metric
        self.mode = monitor_mode
        self.patience = patience
 
        self.best_metric_val = float('-inf') if monitor_mode == 'max' else float('inf')
        self.best_full_metric = None
        self.counter = 0
        self.early_stop = False

    def update(self, metrics):
        current_val = metrics[self.ref_metric]

        is_improved = (current_val > self.best_metric_val) if self.mode == 'max' else (current_val < self.best_metric_val)

        if is_improved:
            self.best_metric_val = current_val
            self.best_full_metric = metrics
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        
    @property
    def should_stop(self):
        return self.early_stop
    
    def __repr__(self):
        return (f"EarlyStopper(metric={self.ref_metric}, mode={self.mode}, "
                f"patience={self.patience}, best={self.best_metric_val:.6f})")