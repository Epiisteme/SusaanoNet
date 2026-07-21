import numpy as np

def ekf_predict(x_prev, P_prev, u_k, vol, Q):
    """
    EKF Time Update (Predict Step)
    x_prev: [Price, Momentum]^T
    P_prev: 2x2 Covariance matrix
    u_k: Mamba predicted price
    vol: current volatility factor
    Q: Process noise covariance matrix
    """
    # Extract previous state
    x1_prev = x_prev[0, 0]  # Price
    x2_prev = x_prev[1, 0]  # Momentum
    
    # Non-linear state transition f(x, u)
    x1_pred = u_k + x2_prev
    x2_pred = np.tanh(x2_prev) * vol
    
    x_pred = np.array([[x1_pred], [x2_pred]], dtype=np.float64)
    
    # Calculate Jacobian F_k
    # df1/dx1 = 0, df1/dx2 = 1
    # df2/dx1 = 0, df2/dx2 = (1 - tanh^2(x2)) * vol
    df2_dx2 = (1.0 - np.tanh(x2_prev)**2) * vol
    F_k = np.array([
        [0.0, 1.0],
        [0.0, df2_dx2]
    ], dtype=np.float64)
    
    # Covariance update: P = F * P_prev * F^T + Q
    P_pred = F_k @ P_prev @ F_k.T + Q
    
    # Ensure symmetry
    P_pred = (P_pred + P_pred.T) / 2.0
    
    return x_pred, P_pred


def ekf_update(x_pred, P_pred, z_k, R):
    """
    EKF Measurement Update (Update Step)
    x_pred: a priori state estimate
    P_pred: a priori covariance estimate
    z_k: measurement (true price)
    R: Measurement noise covariance matrix (scalar for 1D measurement)
    """
    # Observation model h(x) = x1
    H_k = np.array([[1.0, 0.0]], dtype=np.float64)
    
    # Innovation (residual)
    y_k = z_k - x_pred[0, 0]
    
    # Innovation covariance: S = H * P_pred * H^T + R
    S_k = H_k @ P_pred @ H_k.T + R
    
    # Kalman Gain: K = P_pred * H^T * S^-1
    # Since S is 1x1 for a 1D measurement, we can just divide
    K_k = (P_pred @ H_k.T) / S_k[0, 0]
    
    # State update: x = x_pred + K * y
    x_upd = x_pred + K_k * y_k
    
    # Covariance update: P = (I - K * H) * P_pred
    I = np.eye(2, dtype=np.float64)
    P_upd = (I - K_k @ H_k) @ P_pred
    
    # Ensure symmetry to prevent numerical drift
    P_upd = (P_upd + P_upd.T) / 2.0
    
    return x_upd, P_upd
