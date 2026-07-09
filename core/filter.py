import numpy as np
from core.math import ekf_predict, ekf_update

class OnlineEKF:
    """
    Persistent Extended Kalman Filter to act as a dynamic
    post-processing shock-absorber for Mamba forecasts.
    """
    def __init__(self, initial_price: float, q_var=1e-2, r_var=1.0):
        # State vector: [Price, Momentum]^T
        self.x = np.array([[initial_price], [0.0]], dtype=np.float64)
        
        # Initial Error Covariance Matrix P
        self.P = np.eye(2, dtype=np.float64)
        
        # Process Noise Covariance (How much do we trust the system model?)
        self.Q = np.array([
            [q_var, 0.0],
            [0.0, q_var]
        ], dtype=np.float64)
        
        # Measurement Noise Covariance (How much do we trust the raw observation?)
        self.R = np.array([[r_var]], dtype=np.float64)

    def process(self, mamba_forecast: float, actual_price: float, volatility: float) -> float:
        """
        Executes a full Predict-Update EKF cycle.
        Returns the new a posteriori (corrected) price forecast.
        """
        # 1. Predict Step (Time Update)
        # We push the Mamba forecast as the control input to predict the new state
        self.x, self.P = ekf_predict(
            x_prev=self.x,
            P_prev=self.P,
            u_k=mamba_forecast,
            vol=volatility,
            Q=self.Q
        )
        
        # 2. Update Step (Measurement Update)
        # We observe the actual realized market price and correct the state
        self.x, self.P = ekf_update(
            x_pred=self.x,
            P_pred=self.P,
            z_k=actual_price,
            R=self.R
        )
        
        # Return the corrected price estimate (element [0, 0] of the state vector)
        return float(self.x[0, 0])
