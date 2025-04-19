"""
Main file for running simulations. Collaborative positioning uses a Kalman Filter.
"""

import numpy as np
from agent_kf import Agent
import tools as tools
import copy as copy

N_AGENTS = 5
R_MAX = 30                      # Distance at which agents can sense each other
D_MIN = 5                       # Minimum distance between agents
D_MAX = R_MAX*0.8               # Maximum distance between agents
DRIFT = 0.005                   # Real world drift due to steady wind (m/s^2)
IMU_XL_DRIFT = 0.7              # Simulated IMU acceleration drift (std, m/s^2)
IMU_XL_NOISE = 0.05             # Simulated IMU acceleration noise (std, m/s^2)
GNSS_POS_NOISE = 1.5            # Simulated GNSS position noise (std, m)
GNSS_VEL_NOISE = 0.1            # Simulated GNSS velocity noise (std, m/s)
COLLAB_NOISE = 1.0              # Simulated collaborative positioning noise (std, m, m/s)
OUTAGE_MULTIPLIER = 10          # Factor by which to multiply GNSS noise during outages
GUST_AMP = 3.0					# Maximum gust amplitude
VEL_D = [2, 1, 0.1]				# Desired velocity (x, y, z)

SAVE_GIF = False				# Save animation as GIF
SIMTIME = 60.0                  # Simulation time (s)
BUFTIME = 60.0					# Buffer time (s) for Kalman filter to stabilize

SEED = 3
np.random.seed(SEED)

outages = {
			0:[(40,50)],    # Indices and outage durations for degraded agent(s)
		   	# 1:[(30,40)],
		   }
blockages = {
			0:[(10,35)], 	# Indices and blockage durations for degraded agent(s)
		   	# 1:[(30,40)],
		   	}

def main():
	# Initialize agents
	coords = generate_3d_coordinates(N_AGENTS, D_MIN, D_MAX)
	agents = []
	for i in range(N_AGENTS):
		outage_i = outages[i] if i in outages.keys() else []
		blockage_i = blockages[i] if i in blockages.keys() else []
			
		agents.append(Agent(BUFTIME, coords[i], outage_i, blockage_i, IMU_XL_DRIFT, IMU_XL_NOISE, GNSS_POS_NOISE, GNSS_VEL_NOISE, OUTAGE_MULTIPLIER, COLLAB_NOISE, GUST_AMP, VEL_D))

	# Run simulation
	for t in np.arange(-BUFTIME, SIMTIME, 0.1):
		for i in range(N_AGENTS):
			u = np.random.normal(0, DRIFT, size=3)
			A = get_adjacency([agents[i].pos for i in range(N_AGENTS)])
			agents[i].update(u, agents, outages, A, i, t)

	# Print data
	tools.calc_errors(agents[0], start=0.0)

	# Visualize data
	tools.animate_3d_trajectories(SAVE_GIF, agents, outages, blockages)
	tools.plot_trajectory(agents[0], 0, outages, blockages)
	# tools.plot_trajectory_errors(agents[0], 0, outages, blockages)
	# tools.plot_trajectory_mini(agents[0], 0, outages, blockages)
	# tools.plot_trajectory_mini_errors(agents[0], 0, outages, blockages)
	# agents[0].kf4.plot_convergence()

def get_adjacency(X):
	"""
	Compute the adjacency matrix for a given set of agent positions.
	"""
	X = np.array(X)
	A = np.zeros((N_AGENTS,N_AGENTS))
	for i in range(N_AGENTS):
		for j in range(N_AGENTS):
			if np.linalg.norm(X[i,:3] - X[j,:3]) <= R_MAX:
				A[i,j] = 1
			if i == j:
				A[i,j] = 0
	return A

def generate_3d_coordinates(N_AGENTS, D_MIN, D_MAX):
	"""
	Generate N 3D coordinates with a minimum distance of D_MIN and a maximum distance of D_MAX.
	This is to nondeterministically initialize agents' locations.
	"""
	def is_valid_point(new_point, points):
		if len(points) == 0:
			return True
		distances = np.linalg.norm(points - new_point, axis=1)
		return np.all((distances >= D_MIN) & (distances <= D_MAX))

	coordinates = []

	# Create points iteratively
	while len(coordinates) < N_AGENTS:
		new_point = np.random.uniform(0, D_MAX, size=3)
		if is_valid_point(new_point, np.array(coordinates)):
			coordinates.append(new_point)

	return np.array(coordinates)

if __name__ == "__main__":
	main()