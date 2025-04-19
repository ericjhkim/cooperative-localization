"""
Describes individual agents.
"""

import numpy as np
import copy
from kalman_1 import KalmanFilter as kf1
from kalman_3 import KalmanFilter as kf3
from kalman_4 import KalmanFilter as kf4
from kalman_5 import KalmanFilter as kf5

class Agent():
	def __init__(self, buftime, pos, outages, blockages, imu_xl_drift, imu_xl_noise, gnss_pos_noise, gnss_vel_noise, outage_multiplier, collab_noise, gust_amp, vel_d):
		self.dt = 0.1
		self.outages = outages						# GNSS outages
		self.blockages = blockages					# GNSS blockages

		self.gust_amp = gust_amp					# Amplitude of gusts
		self.gust_prob = 0.05						# Probability of gust
		self.max_gust_time = 5.0					# Maximum gust duration
		self.gust = np.array([])

		# Controls
		self.u_max = 2.0							# Maximum control input
		self.Kp = 1.0								# Proportional gain
		self.Kd = 2.0								# Derivative gain
		self.Kv = 1.0  								# Tracking gain for velocity

		# Truth
		self.pos = pos
		self.vel = np.zeros(3)
		self.vel_d = np.array(vel_d)				# Desired velocity (x, y, z)

		# IMU
		self.imu_pos = copy.deepcopy(self.pos)
		self.imu_vel = copy.deepcopy(self.vel)
		self.imu_noise = imu_xl_noise
		self.imu_drift = np.zeros(3)
		self.imu_drift_rate = np.random.normal(0.0, imu_xl_drift, size=3)

		# GNSS
		self.gnss_pos = copy.deepcopy(self.pos)
		self.gnss_vel = copy.deepcopy(self.vel)
		self.gnss_pos_noise = gnss_pos_noise
		self.gnss_vel_noise = gnss_vel_noise
		self.outage_multiplier = outage_multiplier
		self.in_outage = False
		self.in_blockage = False

		# Kalman filters
		self.kf1 = kf1(self.dt)									# Filter for GNSS/IMU
		self.alpha = 0.01 										# Controlled correction (damping factor for closed loop)
        
		self.kf2 = kf1(self.dt)									# Filter for Collab/IMU
		self.kf3 = kf3(self.dt)									# Filter for GNSS/Collab
		self.kf4 = kf4(self.dt)									# Filter for GNSS/Collab + IMU

		self.collab_noise = collab_noise
		self.kf5 = kf5(copy.deepcopy(self.pos), copy.deepcopy(self.vel), self.dt, collab_noise)	# Filter to calculate collaborative position

		# Data saving
		self.states = [np.concatenate((copy.deepcopy(self.pos), copy.deepcopy(self.vel)))]
		self.imu = [np.concatenate((copy.deepcopy(self.imu_pos), copy.deepcopy(self.imu_vel)))]
		self.gnss = [np.concatenate((copy.deepcopy(self.gnss_pos), copy.deepcopy(self.gnss_vel)))]
		self.ksol1 = [np.concatenate((copy.deepcopy(self.pos), copy.deepcopy(self.vel)))]
		self.collab = [np.concatenate((copy.deepcopy(self.pos), copy.deepcopy(self.vel)))]
		self.ksol2 = [np.concatenate((copy.deepcopy(self.pos), copy.deepcopy(self.vel)))]
		self.extpos = [np.concatenate((copy.deepcopy(self.pos), copy.deepcopy(self.vel)))]
		self.final = [np.concatenate((copy.deepcopy(self.pos), copy.deepcopy(self.vel)))]

		self.gust_history = [np.zeros(3)]

		self.t = [-buftime]

	def get_control(self, agents, A, agent_i, t):
		pos_i = self.ksol1[-1][:3] if t <= 1e20 else self.final[-1][:3]
		vel_i = self.ksol1[-1][3:] if t <= 1e20 else self.final[-1][3:]
		pos0_i = self.ksol1[0][:3] if t <= 1e20 else self.final[0][:3]

		u = np.zeros(3)
		num_neighbors = 0

		for j in range(len(agents)):
			if A[agent_i, j] == 1:
				pos_j = agents[j].ksol1[-1][:3] if t <= 1e20 else self.final[-1][:3]
				vel_j = agents[j].ksol1[-1][3:] if t <= 1e20 else self.final[-1][3:]
				pos0_j = agents[j].ksol1[0][:3] if t <= 1e20 else self.final[0][:3]

				# Formation control terms
				rel_pos_error = (pos_i - pos_j) - (pos0_i - pos0_j)
				rel_vel_error = (vel_i - vel_j)

				u += -self.Kp * rel_pos_error - self.Kd * rel_vel_error
				num_neighbors += 1

		if num_neighbors > 0:
			u = u / num_neighbors

		# Add tracking component to global velocity
		u += -self.Kv * (vel_i - self.vel_d)

		# u = np.clip(u, -self.u_max, self.u_max)			# Limit control input
		return u

	def update(self, u, agents, outages, A, agent_i, t):
		# Compute control input (add gust and correction)
		u += self.get_gust()
		u += self.get_control(agents, A, agent_i, t)

		# Advance truth and sensor simulation
		self.update_truth(u)
		self.update_imu(u)
		self.update_gnss(t)

		self.update_kf1()											# GNSS + IMU

		collab = self.update_kf5(agents, outages, A, agent_i, t)	# Collaborative position

		self.update_kf2(collab)										# Collab + IMU
		self.update_kf3(collab, agents, A, agent_i)					# GNSS + Collab
		self.update_kf4(agent_i, t)									# GNSS/Collab + IMU
		self.collab.append(copy.deepcopy(collab))
		
		self.t.append(self.t[-1] + self.dt)

	def in_period(self, period, t):
		"""Check if t is in outage/blockage period."""
		for start,end in period:
			if start <= t <= end:
				return True
		return False

	def get_gust(self):
		if len(self.gust) != 0:
			g = list(self.gust).pop(0)
			self.gust = self.gust[1:,:]
		elif np.random.rand() < self.gust_prob:
			T = np.random.uniform(0.1, self.max_gust_time)
			tau = T/-np.log(0.01)
			t = np.arange(0, T, 0.1)
			amps = np.random.uniform(-self.gust_amp, self.gust_amp, size=3)
			self.gust = np.array([amps*x for x in np.exp(-t/tau)])
			g = list(self.gust).pop(0)
			self.gust = self.gust[1:,:]
		else:
			g = np.zeros(3)

		self.gust_history.append(g)
		return g

	def update_kf1(self):
		"""Kalman filter for IMU/GNSS integration."""
		corrections = self.kf1.run_epoch(self.imu[-1], self.gnss[-1], self.in_outage, self.in_blockage)
		state = self.imu[-1] - corrections
		self.ksol1.append(copy.deepcopy(state))

		# Closed loop feedback
		# self.imu_pos -= self.alpha * corrections[:3]
		# self.imu_vel -= self.alpha * corrections[3:]

	def update_kf2(self, collab):
		"""Kalman filter for IMU/Collaborative position integration."""
		corrections = self.kf2.run_epoch(self.imu[-1], collab)
		state = self.imu[-1] - corrections
		self.ksol2.append(copy.deepcopy(state))

	def update_kf3(self, collab, agents, A, agent_i):
		"""Kalman filter for GNSS/Collaborative position integration."""
		corrections = self.kf3.run_epoch(self.gnss[-1], collab, agents, A, agent_i)
		state = self.gnss[-1] - corrections
		self.extpos.append(copy.deepcopy(state))

	def update_kf4(self, agent_i, t):
		"""Kalman filter for IMU + Collab/GNSS integration."""
		corrections = self.kf4.run_epoch(self.imu[-1], self.extpos[-1], agent_i, t)
		state = self.imu[-1] - corrections
		self.final.append(copy.deepcopy(state))

	def update_kf5(self, agents, outages, A, agent_i, t):
		"""Kalman filter for collaborative position."""
		self.kf5.run_epoch(agents, outages, A, agent_i, t)
		collab = self.kf5.get_collab()
		return collab

	def update_truth(self, u):
		"""Truth simulation."""
		self.vel += u * self.dt
		self.pos += self.vel * self.dt
		self.states.append(np.concatenate((copy.deepcopy(self.pos), copy.deepcopy(self.vel))))

	def update_imu(self, u):
		"""IMU simulation."""
		# Update drift state
		self.imu_drift += self.imu_drift_rate * self.dt
		self.imu_drift = np.clip(self.imu_drift, -0.1, 0.1)

		imu_noise = np.random.normal(0.0, self.imu_noise, size=3)
		self.imu_vel += (u + self.imu_drift + imu_noise) * self.dt
		self.imu_pos += self.imu_vel * self.dt
		self.imu.append(np.concatenate((copy.deepcopy(self.imu_pos), copy.deepcopy(self.imu_vel))))

	def update_gnss(self, t):
		"""GNSS simulation."""
		self.in_outage = False
		self.in_blockage = False

		if self.in_period(self.blockages, t):
			pos_noise = self.gnss_pos_noise * self.outage_multiplier
			vel_noise = self.gnss_vel_noise * self.outage_multiplier
			self.in_blockage = True
		else:
			pos_noise = self.gnss_pos_noise
			vel_noise = self.gnss_vel_noise

		if not self.in_period(self.outages, t):
			self.gnss_vel = copy.deepcopy(self.vel) + np.random.normal(0, vel_noise, size=3)
			self.gnss_pos = copy.deepcopy(self.pos) + np.random.normal(0, pos_noise, size=3)
		else:
			# If in outage, use last known position
			self.in_outage = True

		self.gnss.append(np.concatenate((copy.deepcopy(self.gnss_pos), copy.deepcopy(self.gnss_vel))))