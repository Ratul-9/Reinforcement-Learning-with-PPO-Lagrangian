# Vehicle Module Documentation (`vehicle.py`)

## Overview
The `vehicle.py` module provides a highly realistic `Sedan` vehicle class designed for multi-agent Reinforcement Learning (specifically intended for PPO Lagrangian). It acts as a plug-and-play component that can be injected into any 2D continuous or grid-based Gymnasium environment.

Rather than relying on simplified point-mass or standard bicycle kinematics, this model implements a **4-Wheel Dynamic Vehicle Model**. It accurately calculates mass inertia, tire slip angles, lateral cornering stiffness, and aerodynamic drag, making it perfect for training robust self-driving policies that must adhere to real-world physics and safety constraints.

---

## Features

### 1. 4-Wheel Dynamic Physics Engine
The vehicle mathematically models real-world forces:
- **Mass & Yaw Inertia**: Accounts for the weight of the sedan (1500 kg) and its resistance to rotational changes.
- **Individual Tire Slip Angles**: During cornering, the model calculates the specific slip angle for the Front-Left, Front-Right, Rear-Left, and Rear-Right wheels based on the vehicle's track width and wheelbase.
- **Friction Circle Limits**: Tire lateral forces are clamped to maximum friction ($\mu$). If the RL agent attempts a sharp turn at high speeds, the vehicle will naturally understeer or lose traction.
- **Aerodynamic Drag**: Computes drag based on frontal area and velocity.
- **Low-Speed Kinematic Fallback**: When the vehicle drops below `1.0 m/s`, it smoothly transitions back to a kinematic bicycle model. This is an industry-standard safety measure to prevent the mathematical singularities (divide-by-zero errors in slip angle calculations) that plague dynamic models at low speeds (e.g., when pulling out of a parking spot).

### 2. Drive Modes (Gears)
The vehicle respects real mechanical gears, passed through the action space:
- **`PARK` (0)**: Parking brake engaged. Velocity is forced to `0.0`. The car will not roll down slopes.
- **`REVERSE` (1)**: The throttle action applies negative longitudinal force.
- **`NEUTRAL` (2)**: The engine is decoupled. The vehicle will roll freely based on gravity/slope and momentum. Only the brake action is effective.
- **`DRIVE` (3)**: The throttle action applies positive longitudinal force.

### 3. RL-Ready Spaces (Gymnasium)
The class natively builds structured observation and action spaces using `gymnasium.spaces.Dict`.

#### Action Space
- **`steering`**: `[-1.0, 1.0]` $\rightarrow$ Mapped to maximum steering angle.
- **`throttle`**: `[0.0, 1.0]` $\rightarrow$ Mapped to maximum engine acceleration.
- **`brake`**: `[0.0, 1.0]` $\rightarrow$ Mapped to maximum braking deceleration.
- **`gear`**: Discrete `[0, 1, 2, 3]` corresponding to the modes above.

#### Observation Space
The vehicle outputs a dictionary of arrays on every `step()` and `reset()`:
- **`state`**: `[x, y, vx, vy, heading, yaw_rate, steering_angle, gear]`
- **`navigation`**: `[destination_x, destination_y, distance_to_dest, relative_heading_to_dest]`
- **`rgb_camera`**: $(64, 64, 3)$ Image Array
- **`lidar`**: $120$ Distance Rays
- **`radar`**: $5 \times 4$ Matrix (Max 5 objects, returning relative velocities and positions).

---

## Sensor Integration
The `vehicle.py` class does **not** render the 3D world on its own. It provides the arrays and properties required by the RL algorithm, but it expects the parent Environment simulator to handle the raycasting and rendering.

When calling `step()`, the environment can inject actual sensor readings into the vehicle's observation:

```python
# Assuming your friend's environment generated these readings based on the vehicle's (x,y)
external_readings = {
    'rgb_camera': env_camera_render_array,
    'lidar': env_lidar_distances
}

# The vehicle updates its physics and packages the external sensors into the obs dict
obs = sedan.step(action, slope_angle=0.0, external_sensors=external_readings)
```

---

## Usage Example
```python
from vehicle import Sedan
import numpy as np

# Initialize vehicle with spawn and destination points
sedan = Sedan(spawn_point=(10, 10), destination=(500, 500), dt=0.1)
obs = sedan.reset()

# Sample a random action from the defined space
action = sedan.action_space.sample()

# Step the vehicle forward
obs = sedan.step(action, slope_angle=0.0)

print(f"Current Velocity: {obs['state'][2]} m/s")
print(f"Distance to Goal: {obs['navigation'][2]} m")
```
