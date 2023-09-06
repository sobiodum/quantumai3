import numpy as np
import pandas as pd


class WorkerPerformanceStats():
    def __init__(self, env):
        super(WorkerPerformanceStats, self).__init__()
        self.env = env  

    def return_per_volatility(self):
        if len(self.env.return_memory) > 0:
            
            last_30_days_returns = self.env.return_memory[-30:]
            annual_volatility = np.std(last_30_days_returns) * np.sqrt(252)
            if annual_volatility != 0:
                return_per_volatility = np.mean(last_30_days_returns) / (annual_volatility + 1e-9)
            else:
                return_per_volatility = 0
        else:
            return_per_volatility = 0.0  # or some other default value
        return np.float32(return_per_volatility)
        
