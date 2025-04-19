"""
Describes Kalman filter for collaborative GNSS position estimate.
"""

import numpy as np
import copy as copy

class KalmanFilter():
	def __init__(self, pos_est, vel_est, dt, noise_level):
		# Data
		self.dt = dt

		self.noise_level = noise_level
		
		# Initial state vector
		self.X_updated = np.zeros(6)
		self.pos_est = copy.copy(pos_est)
		self.vel_est = copy.copy(vel_est)

		# Estimation error covariance matrix
		self.P_plus = 1e2*np.eye(6)  			        				# Large initial uncertainty

		# State transition matrix
		self.F = np.eye(6)
		self.F[0, 3] = self.dt  # ddx/dt
		self.F[1, 4] = self.dt  # ddy/dt
		self.F[2, 5] = self.dt  # ddz/dt

		self.Q = np.eye(6)*1e0
		
		# Data storage
		self.history = [np.concatenate((copy.copy(self.pos_est), copy.copy(self.vel_est)))]

	def in_outage(self, outage, t):
		for start,end in outage:
			if start <= t <= end:
				return True
		return False

	def run_epoch(self, agents, outages, A, agent_i, t):
		M = np.count_nonzero(A[agent_i])

		if M < 4:
			X_predicted = self.F @ self.X_updated
			P_minus = self.F @ self.P_plus @ self.F.T + self.Q

			self.X_updated = X_predicted
			self.P_plus = P_minus

			self.pos_est += self.X_updated[:3]
			self.vel_est += self.X_updated[3:]

			self.history.append(np.concatenate((copy.copy(self.pos_est), copy.copy(self.vel_est))))
			return False

		# Retrieve neighbouring positions and velocities
		pos_js = np.array([agents[j].ksol1[-1][:3] for j in range(len(agents)) if A[agent_i][j] == 1])			# Mx3
		vel_js = np.array([agents[j].ksol1[-1][3:] for j in range(len(agents)) if A[agent_i][j] == 1])			# Mx3

		one_m_estimates = np.array([self.get_1m_est(self.pos_est, pos_js[m]) for m in range(M)])    # Mx3

		h_zeros = np.zeros((M, 3))                              									# Filler matrix for H
		H = np.vstack((   np.hstack((one_m_estimates, h_zeros)),
						  np.hstack((h_zeros, one_m_estimates))   ))    							# 2Mx6 (measurement model matrix)
		
		# Measurement vector Z (combining simulated ranges and velocities)
		# Retrieve simulated range values
		pos_rho = self.get_simulated_range(agents, agent_i, A)  		# 1xM
		# Calculate simulated velocity values
		vel_rho = self.get_simulated_velocity(agents, agent_i, A) 		# 1xM

		# Estimate pseudo position and velocities
		ppos_est = np.array([np.linalg.norm(self.pos_est - pos_js[m]) for m in range(M)])
		pvel_est = np.array([self.vel_est - vel_js[m] for m in range(M)])                    # Mx3
		pvel_est = np.array([np.dot(one_m_estimates[m], pvel_est[m]) for m in range(M)])

		# Compute difference between pseudoranges
		del_rho = pos_rho - ppos_est                	# 1xM
		del_rho_dot = vel_rho - pvel_est    			# 1xM

		# Combine position and velocity pseudos
		Z = np.hstack((del_rho, del_rho_dot))

		# Measurement noise covariance matrix R
		# This function calculates the noise covariance based on how noisy the INS/GNSS KF is
		R = self.get_R(agents, A, agent_i)
		
		# Kalman filter equations
		X_predicted = self.F @ self.X_updated
		P_minus = self.F @ self.P_plus @ self.F.T + self.Q

		# Compute Kalman gain
		S = H @ P_minus @ H.T + R
		S += np.eye(S.shape[0]) * 1e-6  # regularization
		K = P_minus @ H.T @ np.linalg.inv(S)

		# Update state and covariance
		self.X_updated = X_predicted + K @ (Z - H @ X_predicted)
		self.P_plus = P_minus - K @ H @ P_minus
		self.P_plus = (self.P_plus + self.P_plus.T)/2 # enforcing symmetry
		
		self.pos_est += self.X_updated[:3]
		self.vel_est += self.X_updated[3:]

		self.history.append(np.concatenate((copy.copy(self.pos_est), copy.copy(self.vel_est))))

		return self.X_updated
	
	def get_collab(self):
		return np.concatenate((copy.copy(self.pos_est), copy.copy(self.vel_est)))
	
	def get_1m_est(self, pos_est, pos_js):
		vec = pos_est - pos_js  			# Vector from estimated position to neighbours
		return vec / np.linalg.norm(vec)  	# Normalize to unit vector

	def get_R(self, agents, A, agent_i):
		"""
		Generates the noise covariance matrix.
		Increases R if the agent is in outage or if the agent is blocked.
		"""
		R_pos = []
		R_vel = []
		for j in range(len(agents)):
			if A[agent_i][j] == 1 and agents[j].in_outage:
				R_pos.append( 1e4 )
				R_vel.append( 1e4 )
			elif A[agent_i][j] == 1:
				R_pos.append( 1e0 * agents[j].gnss_pos_noise )
				R_vel.append( 1e-2 * agents[j].gnss_vel_noise )

		return np.diag(np.concatenate((R_pos, R_vel)))
	
	def get_simulated_range(self, agents, tgt_i, A):
		"""Generates distance measurements from an unknown UAV position."""
		anchors = np.array([agents[j].pos for j in range(len(agents)) if j != tgt_i and A[tgt_i][j] == 1])
		true_pos = np.array(agents[tgt_i].pos)
		true_distances = np.linalg.norm(anchors - true_pos, axis=1)
		noisy_distances = true_distances + np.random.normal(0, self.noise_level, size=true_distances.shape)
		return noisy_distances

	def get_simulated_velocity(self, agents, tgt_i, A):
		"""Generates velocity measurements from an unknown UAV velocity."""
		anchors = np.array([agents[j].vel for j in range(len(agents)) if j != tgt_i and A[tgt_i][j] == 1])
		true_vel = np.array(agents[tgt_i].vel)
		true_velocities = np.linalg.norm(anchors - true_vel, axis=1)
		noisy_velocities = true_velocities + np.random.normal(0, self.noise_level/10.0, size=true_velocities.shape)
		return noisy_velocities