"""
Describes Kalman filter for IMU and GNSS/Collaborative GNSS integration.
"""

import numpy as np
import copy as copy
import matplotlib.pyplot as plt

class KalmanFilter():
	def __init__(self, dt):
		# Data
		self.dt = dt
		self.k_R = 1e0

		# Initial state vector
		self.X_updated = np.zeros(6)

		# Estimation error covariance matrix
		self.P_plus = np.eye(6)

		# State transition matrix
		self.F = np.eye(6)
		self.F[0, 3] = self.dt  # ddx/dt
		self.F[1, 4] = self.dt  # ddy/dt
		self.F[2, 5] = self.dt  # ddz/dt

		self.Q = np.eye(6)*1e-2

		# Measurement model matrix H
		self.H = np.eye(6)

		self.ema_var = np.ones(6) * 1.0     # Initial guess for measurement variance
		self.ema_alpha = 0.1                # EMA smoothing factor

		# Measurement noise covariance matrix R

		# Data storage
		self.history = [copy.deepcopy(self.X_updated)]
		self.ext_history = []				# External data (GNSS or collaborative position)
		self.convergence = []

	def run_epoch(self, INT, EXT, agent_i, t, in_outage=False, in_blockage=False):
		# Check of GNSS outage:
		if in_outage:
			X_predicted = self.F @ self.X_updated
			P_minus = self.F @ self.P_plus @ self.F.T + self.Q

			self.X_updated = X_predicted
			self.P_plus = P_minus

			self.history.append(copy.deepcopy(self.X_updated))
			return self.X_updated

		# Measurement noise covariance matrix R
		R = self.k_R * self.get_R(EXT)

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

		# Convergence check
		self.convergence.append([np.trace(self.P_plus), np.linalg.norm(self.X_updated)])

		return self.X_updated
	
	def get_R(self, EXT):
		"""Measurement noise covariance matrix R using exponential moving average of squared residuals."""
		z = np.array(EXT)

		# Compute squared difference from moving mean
		if len(self.ext_history) >= 10:
			ref = np.mean(self.ext_history[-10:], axis=0)
		else:
			ref = z
		residual = z - ref
		squared_error = residual**2

		# Exponential moving average update
		self.ema_var = self.ema_alpha * squared_error + (1 - self.ema_alpha) * self.ema_var

		# Ensure minimum noise level to avoid singular R
		R = np.diag(np.maximum(self.ema_var, 1e-4))

		self.ext_history.append(copy.deepcopy(EXT))
		return R
	
	def plot_convergence(self):
		plt.figure(figsize=(12,5))
		t = np.arange(0, int(len(self.convergence)*self.dt), self.dt)
		plt.plot(t, np.array(self.convergence)[:,0], label='Trace of P_plus')
		plt.plot(t, np.array(self.convergence)[:,1], label='Norm of X_updated')
		plt.legend()
		plt.grid()
		plt.show()