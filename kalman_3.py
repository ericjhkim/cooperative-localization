"""
Describes Kalman filter for internal GNSS and collaborative position integration.
"""

import numpy as np
import copy as copy

class KalmanFilter():
	def __init__(self, dt):
		# Data
		self.dt = dt
		
		# Initial state vector
		self.X_updated = np.zeros(6)

		# Estimation error covariance matrix
		self.P_plus = np.eye(6)

		# State transition matrix
		self.F = np.eye(6)
		self.F[0, 3] = self.dt  # ddx/dt
		self.F[1, 4] = self.dt  # ddy/dt
		self.F[2, 5] = self.dt  # ddz/dt

		self.Q = np.eye(6)*1e-1

		# Measurement model matrix H
		self.H = np.eye(6)

		self.ema_var = np.ones(6) * 1.0     # Initial guess for measurement variance
		self.ema_alpha = 0.1                # EMA smoothing factor

		# Data storage
		self.history = [copy.deepcopy(self.X_updated)]
		self.ext_history = []				# External data (GNSS or collaborative position)

	def run_epoch(self, INT, EXT, agents, A, agent_i):
		# Measurement noise covariance matrix R
		R = self.get_R(agents, A, agent_i)

		# Kalman filter equations
		X_predicted = self.F @ self.X_updated
		P_minus = self.F @ self.P_plus @ self.F.T + self.Q

		# Compute Kalman gain
		S = self.H @ P_minus @ self.H.T + R
		K = P_minus @ self.H.T @ np.linalg.inv(S)

		# Get IMU and GNSS difference
		Z = INT - EXT

		# Update state and covariance
		self.X_updated = X_predicted + K @ (Z - self.H @ X_predicted)
		self.P_plus = P_minus - K @ self.H @ P_minus
		self.P_plus = (self.P_plus + self.P_plus.T)/2 # enforcing symmetry
		
		self.history.append(copy.deepcopy(self.X_updated))

		return self.X_updated
	
	def get_R(self, agents, A, agent_i):
		"""Measurement noise covariance matrix R depends on standard deviation of collaborative position."""
		gnss_std = []
		for j in range(len(agents)):
			if A[agent_i][j] == 1:
				if agents[j].in_blockage or agents[j].in_outage:
					gnss_std.append(agents[j].gnss_pos_noise * agents[j].outage_multiplier)
				else:
					gnss_std.append(agents[j].gnss_pos_noise)
		
		R = 1e-3 * np.ones(6) * np.mean(np.array(gnss_std)**2)
				
		return R