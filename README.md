# Trilateration-Based Cooperative Localization in GNSS-Degraded Swarms

A simulation framework for decentralized UAV swarms that localize cooperatively under GNSS-degraded conditions using trilateration and a cascaded Kalman filter architecture.

## Summary

This project implements a robust, multi-agent localization and control system for aerial swarms operating in environments with degraded or unavailable GNSS signals.

The core contributions include:
- Cascaded Kalman filter architecture for integrating IMU mechanization, GNSS measurements, and cooperative trilateration-based estimates.
- Trilateration to localize an agent using relative distances to neighbours mimicking GNSS single-point positioning.
- Consensus-based decentralized control to maintain formation under drift, gusts, and sensor noise.
- Realistic environmental modeling including wind gusts, sensor bias, and signal outages.
- Extensive simulation framework with logging and visualization.

## Features

- 3-layer filter cascade:
  1. KF5: Cooperative trilateration-based GNSS estimate (from neighbouring agents)
  2. KF3: Fusion of cooperative and local GNSS
  3. KF4: Final INS+External integration

- Relative localization using noisy pseudoranges and Doppler-style velocity projections
- Agent control using formation-maintaining, consensus-based PD control
- GNSS degradation modeling: per-agent blockages and outages
- Plotting and animation of 3D trajectories and Kalman filter performance
- Modular, object-oriented Python codebase with configurable noise and filter parameters

## Repository Structure

```bash
├── main_kf.py                # Main simulation loop
├── agent_kf.py               # Agent class with Kalman filters and control
├── kalman_1.py               # Benchmark: IMU + GNSS
├── kalman_3.py               # External state: local GNSS + cooperative
├── kalman_4.py               # Final estimate: INS + kalman_3 output
├── kalman_5.py               # Cooperative trilateration Kalman filter
├── tools.py                  # Plotting utilities and animation functions
├── visualizations/           # Gif
├── Report/                   # Report files
└── README.md
```

## How It Works

1. Agents are initialized in 3D space with GNSS and IMU sensors.
2. At each timestep:
   - Agents compute relative range and LOS velocities to neighbours.
   - Cooperative estimates are obtained via trilateration-based linearization.
   - A 3-stage filter fuses cooperative, local GNSS, and IMU states.
   - A decentralized control law uses benchmark or fused estimates to compute control input.
   - Environmental disturbances (gusts, jitter) are applied.
3. Data is logged and analyzed post-simulation.

## Results

Cooperative localization improves position accuracy when an individual agent suffers GNSS outages/blockages.

<img src="https://github.com/ericjhkim/cooperative-localization/blob/main/visualizations/anim_20250418_173942.gif" style="width:75%;">

_After an initial convergence period, the position estimates converge and remain stable._

| Network Size (N) | Benchmark Pos. Error | Final Pos. Error | Benchmark Vel. Error | Final Vel. Error |
|------------------|----------------------|------------------|----------------------|------------------|
| 5                | 1.3 m                | 8.1 m            | 0.36 m/s             | 1.2 m/s          |
| 6                | 1.6 m                | 0.82 m           | 0.35 m/s             | 0.91 m/s         |
| 7                | 1.5 m                | 0.82 m           | 0.37 m/s             | 0.86 m/s         |
| 8                | 1.9 m                | 0.78 m           | 0.38 m/s             | 0.92 m/s         |

When a neighbouring agent also undergoes GNSS degradation, the technique improves state estimation when N ≥ 6, as at least 4 satellites are required for an accurate positioning fix. Velocity estimation remains limited due to geometric projection constraints.

## Technical Background

This work is inspired by:
- GNSS single-point positioning linearization
- Kalman filtering for sensor fusion
- Cooperative UWB-based localization
- Decentralized formation control in mobile robotics