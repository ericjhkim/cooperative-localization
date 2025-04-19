"""
Plotting tools
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d
from datetime import datetime
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

## Constants
N_AGENTS = 5
R_MAX = 30                  # Distance at which agents can sense each other

## Functions
def calc_errors(agent, start=0):
	start_i = int(start/agent.dt)
	
	TRUTH = np.array(agent.states)[start_i:]
	imu = np.array(agent.imu)[start_i:]
	gnss = np.array(agent.gnss)[start_i:]
	ksol1 = np.array(agent.ksol1)[start_i:]
	collab = np.array(agent.collab)[start_i:]
	ksol2 = np.array(agent.ksol2)[start_i:]
	extpos = np.array(agent.extpos)[start_i:]
	final = np.array(agent.final)[start_i:]
	datasets = [imu,gnss,collab,extpos,ksol1,final]

	E = {"IMU":{},
		 "GNSS":{},
		 "KF5: Collab":{},
		#  "KF2: IMU/Collab":{},
		 "KF3: GNSS/Collab":{},
		 "KF1: IMU/GNSS":{},
		 "KF4: Final":{},
		 }
	
	print(f"Errors starting at {start} seconds----------------------------------")
	for dataset, DATA in [(list(E.keys())[i],D) for i,D in enumerate(datasets)]:
		print(f"{dataset}-------------------")
		E[dataset]["ERR"] = {}
		E[dataset]["ERR"]["x"] = DATA[:,0] - TRUTH[:,0]
		E[dataset]["ERR"]["y"] = DATA[:,1] - TRUTH[:,1]
		E[dataset]["ERR"]["z"] = DATA[:,2] - TRUTH[:,2]
		E[dataset]["ERR"]["3d"] = np.sqrt(E[dataset]["ERR"]["x"]**2 + E[dataset]["ERR"]["y"]**2 + E[dataset]["ERR"]["z"]**2)

		E[dataset]["ERR"]["vx"] = DATA[:,3] - TRUTH[:,3]
		E[dataset]["ERR"]["vy"] = DATA[:,4] - TRUTH[:,4]
		E[dataset]["ERR"]["vz"] = DATA[:,5] - TRUTH[:,5]
		E[dataset]["ERR"]["v3d"] = np.sqrt(E[dataset]["ERR"]["vx"]**2 + E[dataset]["ERR"]["vy"]**2 + E[dataset]["ERR"]["vz"]**2)

		# RMS errors
		E[dataset]["RMS"] = {}
		# for key in E[dataset]["ERR"].keys():
		for key in ["3d","v3d"]:
			E[dataset]["RMS"][key] = np.sqrt(np.mean(E[dataset]["ERR"][key]**2))
			print(f'RMS Error {key}:\t{np.round(E[dataset]["RMS"][key],4)}')
		print("")

		# Average errors
		E[dataset]["AVG"] = {}
		for key in E[dataset]["ERR"].keys():
			E[dataset]["AVG"][key] = np.mean(E[dataset]["ERR"][key])
		#     print(f'Avg Error {key}:\t{np.round(E[dataset]["AVG"][key],2)}')
		# print("")

		# Max errors
		E[dataset]["MAX"] = {}
		for key in E[dataset]["ERR"].keys():
			E[dataset]["MAX"][key] = max(E[dataset]["ERR"][key])
		#     print(f'Max Error {key}:\t{np.round(E[dataset]["MAX"][key],2)}')
		# print("")

def visualize_gusts():
	max_gust_time = 5.0
	gust_amp = 3.0

	T = np.random.uniform(0.1, max_gust_time)
	tau = T/-np.log(0.01)
	t = np.arange(0, T, 0.1)
	amps = np.random.uniform(-gust_amp, gust_amp, size=3)
	gust = np.array([amps*x for x in np.exp(-t/tau)])

	plt.figure(figsize=(6,4),dpi=200)
	plt.plot(t, gust[:,0], label='x')
	plt.plot(t, gust[:,1], label='y')
	plt.plot(t, gust[:,2], label='z')
	plt.legend()
	plt.xlabel('Time (s)')
	plt.ylabel('Gust Amplitude ($m/s^2$)')
	plt.grid()
	plt.show()

def animate_3d_trajectories(save_gif, agents, outages, blockages, start=0.0, interval=100, gust_length=1.0, ALPHA=0.7):
	fig = plt.figure(dpi=200)
	ax = fig.add_subplot(111, projection='3d')

	# Get time vector and start index
	t = np.array(agents[0].t)
	idx = np.where(t >= start)[0][0]
	t = t[idx:]

	# Determine 3D limits
	all_positions = np.concatenate([np.array(agent.states)[idx:, :3] for agent in agents], axis=0)
	x_min, y_min, z_min = np.min(all_positions, axis=0)
	x_max, y_max, z_max = np.max(all_positions, axis=0)

	max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
	mid_x = (x_max + x_min) * 0.5
	mid_y = (y_max + y_min) * 0.5
	mid_z = (z_max + z_min) * 0.5

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.grid(True)

	for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
		axis._axinfo['grid']['color'] = (0, 0, 0, 0.1)

	lines = [ax.plot([], [], [], color='b', alpha=ALPHA)[0] for _ in agents]
	agent_points = [ax.plot([], [], [], 'o', color='black', markersize=4, alpha=ALPHA)[0] for _ in agents]
	gust_lines = [Line3D([], [], [], color='green', linewidth=1, alpha=ALPHA) for _ in agents]
	for line in gust_lines:
		ax.add_line(line)
	estimation_lines = [ax.plot([], [], [], color='orange', alpha=ALPHA)[0] for _ in agents]
	connection_lines = []
	time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
	disks = []
	status_rings = []

	def init():
		for line, point, gust_line, est_line in zip(lines, agent_points, gust_lines, estimation_lines):
			line.set_data([], [])
			line.set_3d_properties([])
			point.set_data([], [])
			point.set_3d_properties([])
			gust_line.set_data([], [])
			gust_line.set_3d_properties([])
			est_line.set_data([], [])
			est_line.set_3d_properties([])
		time_text.set_text('')

		for _ in agents:
			circle = Circle((0, 0), radius=0.01, color='red', alpha=0.07)
			ax.add_patch(circle)
			art3d.pathpatch_2d_to_3d(circle, z=0, zdir="z")
			disks.append(circle)

			ring = Circle((0, 0), radius=2.0, fill=False, linestyle='dotted', linewidth=1.5)
			ax.add_patch(ring)
			art3d.pathpatch_2d_to_3d(ring, z=0, zdir="z")
			ring.set_visible(False)
			status_rings.append(ring)

		return lines + agent_points + gust_lines + estimation_lines + disks + status_rings + [time_text]

	def update(frame):
		nonlocal connection_lines, disks, status_rings
		for conn in connection_lines:
			try:
				conn.remove()
			except Exception:
				pass
		connection_lines = []

		current_time = t[frame]
		current_positions = [agent.states[idx + frame][:3] for agent in agents]
		A = get_adjacency(current_positions)

		pos_array = np.array(current_positions)
		x_min, y_min, z_min = np.min(pos_array, axis=0)
		x_max, y_max, z_max = np.max(pos_array, axis=0)
		max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min, 1.0]) / 2.0
		padding = 2.0
		mid_x = (x_max + x_min) / 2.0
		mid_y = (y_max + y_min) / 2.0
		mid_z = (z_max + z_min) / 2.0
		ax.set_xlim(mid_x - max_range - padding, mid_x + max_range + padding)
		ax.set_ylim(mid_y - max_range - padding, mid_y + max_range + padding)
		ax.set_zlim(mid_z - max_range - padding, mid_z + max_range + padding)

		for i, agent in enumerate(agents):
			positions = np.array(agent.states)[idx:]
			gusts = np.array(agent.gust_history)[idx:]
			x, y, z = positions[:frame + 1, 0], positions[:frame + 1, 1], positions[:frame + 1, 2]
			lines[i].set_data(x, y)
			lines[i].set_3d_properties(z)

			curr_pos = positions[frame][:3]
			is_outage = any(start <= current_time <= end for start, end in outages.get(i, []))
			is_blockage = any(start <= current_time <= end for start, end in blockages.get(i, []))
			agent_points[i].set_color('red' if is_outage else 'orange' if is_blockage else 'black')
			agent_points[i].set_data([curr_pos[0]], [curr_pos[1]])
			agent_points[i].set_3d_properties([curr_pos[2]])

			gust_vec = -gusts[frame]
			scaled_vec = gust_vec * gust_length
			gust_end = curr_pos + scaled_vec
			gust_lines[i].set_data([curr_pos[0], gust_end[0]], [curr_pos[1], gust_end[1]])
			gust_lines[i].set_3d_properties([curr_pos[2], gust_end[2]])

			est = np.array(agent.final)[idx:][:frame + 1, :3]
			estimation_lines[i].set_data(est[:, 0], est[:, 1])
			estimation_lines[i].set_3d_properties(est[:, 2])

			valid_start = max(0, frame - 9)
			true_pos = positions[valid_start:frame + 1, :3]
			est_pos = np.array(agent.final)[idx:][valid_start:frame + 1, :3]
			radius = np.mean(np.linalg.norm(true_pos - est_pos, axis=1)) if len(true_pos) > 0 else 0.01
			center = curr_pos

			disks[i].remove()
			new_circle = Circle((center[0], center[1]), radius=radius, color='red', alpha=0.1)
			ax.add_patch(new_circle)
			art3d.pathpatch_2d_to_3d(new_circle, z=center[2], zdir="z")
			disks[i] = new_circle

			status_rings[i].remove()
			ring_color = 'red' if is_outage else 'orange' if is_blockage else None
			if ring_color:
				new_ring = Circle((center[0], center[1]), radius=2.0, fill=False, linestyle='dotted', edgecolor=ring_color, linewidth=1.5)
				ax.add_patch(new_ring)
				art3d.pathpatch_2d_to_3d(new_ring, z=center[2], zdir="z")
				status_rings[i] = new_ring
			else:
				dummy = Circle((0, 0), radius=2.0, fill=False, linestyle='dotted', edgecolor='none')
				ax.add_patch(dummy)
				art3d.pathpatch_2d_to_3d(dummy, z=0, zdir="z")
				status_rings[i] = dummy

		for i in range(len(agents)):
			for j in range(i + 1, len(agents)):
				if A[i][j] == 1:
					pi = current_positions[i]
					pj = current_positions[j]
					conn = Line3D([pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]], color='black', linestyle='dotted', linewidth=0.8, alpha=ALPHA)
					ax.add_line(conn)
					connection_lines.append(conn)

		time_text.set_text(f'Time: {current_time:.2f} s')
		return lines + agent_points + gust_lines + estimation_lines + connection_lines + disks + status_rings + [time_text]

	num_frames = len(agents[0].states) - idx
	ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=interval, blit=False)

	legend_elements = [
		Line2D([0], [0], marker='o', color='w', label='Agent',
			markerfacecolor='black', markersize=6),
		Line2D([0], [0], color='k', lw=2, linestyle=':', label='Inter-Agent Connections'),
		Line2D([0], [0], color='g', lw=2, label='Wind Gusts'),
		Line2D([0], [0], color='b', lw=2, label='True Trajectory'),
		Line2D([0], [0], color='orange', lw=2, label='Estimated Trajectory'),
		Patch(facecolor='red', alpha=0.1, label='Estimation Error Disk'),
		Line2D([0], [0], marker='o', color='w', label='Outage Agent',
			markerfacecolor='red', markersize=6),
		Line2D([0], [0], marker='o', color='w', label='Blockage Agent',
			markerfacecolor='orange', markersize=6),
		Patch(facecolor='none', edgecolor='red', linestyle='dotted', linewidth=1.5, label='Outage Ring'),
		Patch(facecolor='none', edgecolor='orange', linestyle='dotted', linewidth=1.5, label='Blockage Ring'),
	]

	# Add to the right of the plot
	ax.legend(handles=legend_elements,
		  loc='lower center',
		  bbox_to_anchor=(0.5, 1.04),  # adjust upward as needed
		  ncol=2,
		  fontsize=6,
		  fancybox=True,
		  shadow=False)

	plt.tight_layout()

	if save_gif:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		path = f"visualizations/anim_{timestamp}.gif"
		ani.save(path, writer='pillow', fps=1000/interval)

	plt.show()

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

def plot_trajectory_mini(agent, agent_i, outages, blockages):
	fig = plt.figure(figsize=(8, 6))
	# plt.suptitle(f"Agent {agent_i} Trajectory")
	handles, labels = [], []
	h1, h2 = None, None

	# Get time vector
	t = np.array(agent.t)
	idx = np.where(t >= 0)[0]
	t = t[idx]

	truth = np.array(agent.states)[idx]
	imu = np.array(agent.imu)[idx]
	gnss = np.array(agent.gnss)[idx]
	ksol1 = np.array(agent.ksol1)[idx]
	kf1_x_est = np.array(agent.kf1.history)[idx]
	collab = np.array(agent.collab)[idx]
	ksol2 = np.array(agent.ksol2)[idx]
	extpos = np.array(agent.extpos)[idx]
	final = np.array(agent.final)[idx]
	data = {
			# "IMU":imu,
		 	# "GNSS":gnss,
		 	# "Benchmark Filter\nIMU/GNSS":ksol1,
		 	# "Collaborative Filter\nCollab":collab,
		 	# "KF2\nIMU/Collab":ksol2,
		 	# "External Filter\nGNSS/Collab":extpos,
		 	"Integration Filter\nINS/External":final
			}

	# Determine outages/blockages for the current agent (if dictionaries are provided)
	outage_periods = outages.get(agent_i, []) if outages else []
	blockage_periods = blockages.get(agent_i, []) if blockages else []

	label = list(data.keys())[0]
	for j in range(3):  # X, Y, Z
		plt.subplot(3, 1, j+1)
		h1, = plt.plot(t, truth[:, j], color='k', alpha=0.7)
		h2, = plt.plot(t, data[label][:, j], alpha=0.7)

		# Add shading for outages (red) and blockages (orange)
		for start, end in outage_periods:
			h3 = plt.axvspan(start, end, color='red', alpha=0.2)
		for start, end in blockage_periods:
			h4 = plt.axvspan(start, end, color='orange', alpha=0.2)

		plt.ylabel(["X (m)", "Y (m)", "Z (m)"][j])

		plt.grid()

	fig.supxlabel("Time (s)", y=0.05)
			
	handles.extend([h1, h2, h3, h4])
	labels.extend(["Truth", "State", "Outage", "Blockage"])
	plt.figlegend(handles, labels, loc="upper center", ncol=4, fontsize=12)
	plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the legend
	plt.show()

def plot_trajectory_mini_errors(agent, agent_i, outages, blockages):
	fig = plt.figure(figsize=(8, 6))
	# plt.suptitle(f"Agent {agent_i} Trajectory")
	handles, labels = [], []
	h1, h2 = None, None

	# Get time vector
	t = np.array(agent.t)
	idx = np.where(t >= 0)[0]
	t = t[idx]

	truth = np.array(agent.states)[idx]
	imu = np.array(agent.imu)[idx]
	gnss = np.array(agent.gnss)[idx]
	ksol1 = np.array(agent.ksol1)[idx]
	kf1_x_est = np.array(agent.kf1.history)[idx]
	collab = np.array(agent.collab)[idx]
	ksol2 = np.array(agent.ksol2)[idx]
	extpos = np.array(agent.extpos)[idx]
	final = np.array(agent.final)[idx]
	data = {
			# "IMU":imu,
		 	# "GNSS":gnss,
		 	"Benchmark Filter\nIMU/GNSS":ksol1,
		 	# "Collaborative Filter\nCollab":collab,
		 	# "KF2\nIMU/Collab":ksol2,
		 	# "External Filter\nGNSS/Collab":extpos,
		 	"Integration Filter\nINS/External":final
			}

	# Determine outages/blockages for the current agent (if dictionaries are provided)
	outage_periods = outages.get(agent_i, []) if outages else []
	blockage_periods = blockages.get(agent_i, []) if blockages else []

	label = list(data.keys())[0]
	for j in range(3):  # X, Y, Z
		plt.subplot(3, 1, j+1)
		bench = np.abs(truth[:, 3+j] - ksol1[:, 3+j])
		fin = np.abs(truth[:, 3+j] - final[:, 3+j])
		
		h1, = plt.plot(t, bench, alpha=0.7, color='g')
		h2, = plt.plot(t, fin, alpha=0.7, color='b')

		# Add shading for outages (red) and blockages (orange)
		for start, end in outage_periods:
			h3 = plt.axvspan(start, end, color='red', alpha=0.2)
		for start, end in blockage_periods:
			h4 = plt.axvspan(start, end, color='orange', alpha=0.2)

		plt.ylabel(["X (m)", "Y (m)", "Z (m)"][j])

		plt.grid()

	fig.supxlabel("Time (s)", y=0.05)
			
	handles.extend([h1, h2, h3, h4])
	labels.extend(["Benchmark", "Final", "Outage", "Blockage"])
	plt.figlegend(handles, labels, loc="upper center", ncol=4, fontsize=12)
	plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the legend
	plt.show()

def plot_trajectory(agent, agent_i, outages, blockages):
	fig = plt.figure(figsize=(12, 8))
	# plt.suptitle(f"Agent {agent_i} Trajectory")
	handles, labels = [], []
	h1, h2 = None, None

	truth = np.array(agent.states)
	imu = np.array(agent.imu)
	gnss = np.array(agent.gnss)
	ksol1 = np.array(agent.ksol1)
	kf1_x_est = np.array(agent.kf1.history)
	collab = np.array(agent.collab)
	ksol2 = np.array(agent.ksol2)
	extpos = np.array(agent.extpos)
	final = np.array(agent.final)
	data = {
			"IMU":imu,
		 	"GNSS":gnss,
		 	"Benchmark Filter\nIMU/GNSS":ksol1,
		 	"Collaborative Filter\nCollab":collab,
		 	# "KF2\nIMU/Collab":ksol2,
		 	"External Filter\nGNSS/Collab":extpos,
		 	"Integration Filter\nINS/External":final
			}
	
	# Get time vector
	t = np.array(agent.t)

	# Determine outages/blockages for the current agent (if dictionaries are provided)
	outage_periods = outages.get(agent_i, []) if outages else []
	blockage_periods = blockages.get(agent_i, []) if blockages else []

	for i, label in enumerate(data.keys()):
		for j in range(3):  # X, Y, Z
			plt.subplot(len(data), 3, 3 * i + j + 1)
			plt.plot(t, truth[:, j], color='k', alpha=0.7)
			plt.plot(t, data[label][:, j], alpha=0.7)

			# Add shading for outages (red) and blockages (orange)
			for start, end in outage_periods:
				h1 = plt.axvspan(start, end, color='red', alpha=0.2)
			for start, end in blockage_periods:
				h2 = plt.axvspan(start, end, color='orange', alpha=0.2)

			if j == 0:
				plt.ylabel(label)

			plt.grid()

	fig.supxlabel("Time (s)", y=0.02)
			
	handles.extend([h1, h2])
	labels.extend(["Outage", "Blockage"])
	plt.figlegend(handles, labels, loc="upper center", ncol=2, fontsize=12)
	plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the legend
	plt.show()
	
def plot_trajectory_errors(agent, agent_i, outages, blockages):
	plt.figure(figsize=(12, 8))
	# plt.suptitle(f"Agent {agent_i} Trajectory Errors")
	handles, labels = [], []
	h1, h2 = None, None

	truth = np.array(agent.states)
	imu = np.array(agent.imu)
	gnss = np.array(agent.gnss)
	ksol1 = np.array(agent.ksol1)
	kf1_x_est = np.array(agent.kf1.history)
	collab = np.array(agent.collab)
	ksol2 = np.array(agent.ksol2)
	extpos = np.array(agent.extpos)
	final = np.array(agent.final)
	data = {
			"IMU":imu,
		 	"GNSS":gnss,
		 	"Benchmark Filter\nIMU/GNSS":ksol1,
		 	"Collaborative Filter\nCollab":collab,
		 	# "KF2\nIMU/Collab":ksol2,
		 	"External Filter\nGNSS/Collab":extpos,
		 	"Integration Filter\nINS/External":final
			}
	
	# Get time vector
	t = np.array(agent.t)

	# Determine outages/blockages for the current agent (if dictionaries are provided)
	outage_periods = outages.get(agent_i, []) if outages else []
	blockage_periods = blockages.get(agent_i, []) if blockages else []

	for i, label in enumerate(data.keys()):
		for j in range(3):  # X, Y, Z
			plt.subplot(len(data), 3, 3 * i + j + 1)
			plt.plot(t, abs(truth[:, j] - data[label][:, j]), alpha=0.7)

			# Add shading for outages (red) and blockages (orange)
			for start, end in outage_periods:
				h1 = plt.axvspan(start, end, color='red', alpha=0.2)
			for start, end in blockage_periods:
				h2 = plt.axvspan(start, end, color='orange', alpha=0.2)

			if j == 0:
				plt.ylabel(label)

			plt.grid()

	handles.extend([h1, h2])
	labels.extend(["Outage", "Blockage"])
	plt.figlegend(handles, labels, loc="upper center", ncol=2, fontsize=12)
	plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the legend
	plt.show()

def plot_velocity(agent, agent_i, outages, blockages):
	plt.figure(figsize=(12, 8))
	plt.suptitle(f"Agent {agent_i} Velocity")

	truth = np.array(agent.states)
	imu = np.array(agent.imu)
	gnss = np.array(agent.gnss)
	ksol1 = np.array(agent.ksol1)
	kf1_x_est = np.array(agent.kf1.history)
	collab = np.array(agent.collab)
	ksol2 = np.array(agent.ksol2)
	extpos = np.array(agent.extpos)
	final = np.array(agent.final)
	data = {
			"IMU":imu,
		 	"GNSS":gnss,
		 	"Benchmark Filter\nIMU/GNSS":ksol1,
		 	"Collaborative Filter\nCollab":collab,
		 	# "KF2\nIMU/Collab":ksol2,
		 	"External Filter\nGNSS/Collab":extpos,
		 	"Integration Filter\nINS/External":final
			}
	
	# Get time vector
	t = np.array(agent.t)

	# Determine outages/blockages for the current agent (if dictionaries are provided)
	outage_periods = outages.get(agent_i, []) if outages else []
	blockage_periods = blockages.get(agent_i, []) if blockages else []

	for i, label in enumerate(data.keys()):
		for j in range(3):  # X, Y, Z
			plt.subplot(len(data), 3, 3 * i + j + 1)
			plt.plot(t, truth[:, 3+j], color='k', alpha=0.7)
			plt.plot(t, data[label][:, 3+j], alpha=0.7)

			# Add shading for outages (red) and blockages (orange)
			for start, end in outage_periods:
				plt.axvspan(start, end, color='red', alpha=0.2)
			for start, end in blockage_periods:
				plt.axvspan(start, end, color='orange', alpha=0.2)

			if j == 0:
				plt.ylabel(label)

			plt.grid()

	plt.tight_layout()
	plt.show()
	
def plot_3d_trajectory(agents):
	fig = plt.figure(figsize=(12, 8))
	ax = fig.add_subplot(111, projection='3d')

	handles, labels = [], []

	for agent in agents:
		truth = np.array(agent.states)
		h1, = ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], label="Truth", color='b', alpha=0.7)
		h2 = ax.scatter(truth[:, 0][0], truth[:, 1][0], truth[:, 2][0], color='g', s=10, label="Start", alpha=0.7)  # Green dot for start
		h3 = ax.scatter(truth[:, 0][-1], truth[:, 1][-1], truth[:, 2][-1], color='r', s=10, label="End", alpha=0.7)  # Red dot for end

		imu = np.array(agent.imu)
		h4, = ax.plot(imu[:, 0], imu[:, 1], imu[:, 2], label="IMU", color='m', linestyle='--', alpha=0.7)

		gnss = np.array(agent.gnss)
		h5, = ax.plot(gnss[:, 0], gnss[:, 1], gnss[:, 2], label="GNSS", color='orange', linestyle=':', alpha=0.7)

		int = np.array(agent.int)
		h6, = ax.plot(int[:, 0], int[:, 1], int[:, 2], label="KF", color='g', alpha=0.7)

		collab = np.array(agent.collab)
		h7, = ax.plot(collab[1:, 0], collab[1:, 1], collab[1:, 2], label="CP", color='k', alpha=0.7)

	handles.extend([h2, h3, h1, h4, h5, h6, h7])
	labels.extend(["Start", "End", "Truth", "IMU", "GNSS", "KF", "CP"])

	ax.set_title("Trajectory")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")

	plt.figlegend(handles, labels, loc="upper center", ncol=5, fontsize=12)

	plt.show()

