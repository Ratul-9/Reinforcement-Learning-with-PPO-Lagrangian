import numpy as np
from enum import Enum
import gymnasium as gym
from gymnasium import spaces

class Gear(Enum):
    PARK = 0
    REVERSE = 1
    NEUTRAL = 2
    DRIVE = 3

class Sedan:
    """
    Sedan vehicle model for multi-agent RL (e.g. PPO Lagrangian).
    Uses a 4-Wheel Dynamic Vehicle Model for realistic physics (slip angles, 
    lateral forces, mass inertia, and track width).
    Designed to be a plug-and-play component for any grid/continuous environment.
    """
    def __init__(self, spawn_point=(0.0, 0.0), destination=(100.0, 100.0), dt=0.1):
        # Environment properties
        self.dt = dt
        self.spawn_point = np.array(spawn_point, dtype=np.float32)
        self.destination = np.array(destination, dtype=np.float32)
        
        # Vehicle state (Global)
        self.x = self.spawn_point[0]
        self.y = self.spawn_point[1]
        self.heading = 0.0  # Orientation angle in radians
        
        # Vehicle state (Local Dynamics)
        self.vx = 0.0 # Longitudinal velocity (m/s)
        self.vy = 0.0 # Lateral velocity (m/s)
        self.yaw_rate = 0.0 # Angular velocity (rad/s)
        self.steering_angle = 0.0
        self.gear = Gear.PARK
        
        # --- Sedan Physical Parameters ---
        self.mass = 1500.0 # kg
        self.Iz = 3000.0 # Yaw moment of inertia (kg*m^2)
        self.lf = 1.2 # Distance from CG to front axle (m)
        self.lr = 1.6 # Distance from CG to rear axle (m)
        self.track_width = 1.8 # m
        self.wheelbase = self.lf + self.lr
        
        self.max_steer_angle = np.radians(35.0)
        self.max_accel = 3.0 # m/s^2 (corresponds to engine force)
        self.max_brake = 8.0 # m/s^2
        self.drag_coef = 0.3 # Aerodynamic drag coefficient
        self.frontal_area = 2.2 # m^2
        self.air_density = 1.225 # kg/m^3
        self.gravity = 9.81
        
        # Tire parameters (Linear tire model with friction circle)
        self.Cf = 100000.0 # Front cornering stiffness (N/rad) - total for axle
        self.Cr = 100000.0 # Rear cornering stiffness (N/rad) - total for axle
        self.mu = 0.9 # Friction coefficient
        
        # Sensor configurations
        self.rgb_resolution = (64, 64, 3)
        self.lidar_rays = 120
        self.radar_objects_max = 5
        
        # Define observation and action spaces
        self._build_spaces()
        
    def _build_spaces(self):
        """
        Defines the observation and action spaces for the RL agent using Gymnasium.
        """
        self.action_space = spaces.Dict({
            'steering': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'throttle': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'brake': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'gear': spaces.Discrete(4) # 0: Park, 1: Reverse, 2: Neutral, 3: Drive
        })
        
        self.observation_space = spaces.Dict({
            'state': spaces.Box(
                low=-np.inf, high=np.inf, 
                # [x, y, vx, vy, heading, yaw_rate, steering_angle, gear]
                shape=(8,), 
                dtype=np.float32
            ),
            'navigation': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(4,), # [dest_x, dest_y, distance, relative_heading]
                dtype=np.float32
            ),
            'rgb_camera': spaces.Box(
                low=0, high=255, 
                shape=self.rgb_resolution, 
                dtype=np.uint8
            ),
            'lidar': spaces.Box(
                low=0.0, high=100.0, 
                shape=(self.lidar_rays,), 
                dtype=np.float32
            ),
            'radar': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.radar_objects_max, 4), 
                dtype=np.float32
            )
        })

    def reset(self, spawn_point=None, destination=None):
        if spawn_point is not None:
            self.spawn_point = np.array(spawn_point, dtype=np.float32)
        if destination is not None:
            self.destination = np.array(destination, dtype=np.float32)
            
        self.x = self.spawn_point[0]
        self.y = self.spawn_point[1]
        self.vx = 0.0
        self.vy = 0.0
        self.heading = 0.0
        self.yaw_rate = 0.0
        self.steering_angle = 0.0
        self.gear = Gear.PARK
        
        return self.get_observation()
        
    def step(self, action, slope_angle=0.0, external_sensors=None):
        """
        Update the vehicle state using a 4-Wheel Dynamic Model.
        """
        steering_input = np.clip(action['steering'][0], -1.0, 1.0)
        throttle_input = np.clip(action['throttle'][0], 0.0, 1.0)
        brake_input = np.clip(action['brake'][0], 0.0, 1.0)
        gear_input = action['gear']
        
        # Update gear and steering
        if gear_input == 0: self.gear = Gear.PARK
        elif gear_input == 1: self.gear = Gear.REVERSE
        elif gear_input == 2: self.gear = Gear.NEUTRAL
        elif gear_input == 3: self.gear = Gear.DRIVE
            
        self.steering_angle = steering_input * self.max_steer_angle
        
        # --- Longitudinal Forces ---
        Fx = 0.0
        force_gravity = -self.mass * self.gravity * np.sin(slope_angle)
        
        if self.gear == Gear.DRIVE:
            Fx = throttle_input * self.mass * self.max_accel
        elif self.gear == Gear.REVERSE:
            Fx = -throttle_input * self.mass * self.max_accel
            
        # Braking force
        if brake_input > 0 and abs(self.vx) > 0.01:
            brake_force = np.sign(self.vx) * brake_input * self.mass * self.max_brake
            Fx -= brake_force
            
        # Aero drag
        drag_force = 0.5 * self.drag_coef * self.air_density * self.frontal_area * (self.vx**2)
        Fx -= np.sign(self.vx) * drag_force
        Fx += force_gravity
        
        # --- Kinematic vs Dynamic Physics ---
        # At very low speeds, dynamic models with slip angles become unstable (singularity in atan2).
        # We fall back to a kinematic model below 1 m/s.
        velocity_mag = np.hypot(self.vx, self.vy)
        
        if velocity_mag < 1.0:
            # Kinematic update
            acceleration = Fx / self.mass
            if self.gear == Gear.PARK:
                self.vx = 0.0
            else:
                self.vx += acceleration * self.dt
                
            self.vy = 0.0
            self.yaw_rate = (self.vx / self.wheelbase) * np.tan(self.steering_angle)
            
            # Stop completely logic
            if abs(self.vx) < 0.1 and (brake_input > 0 or self.gear == Gear.PARK):
                self.vx = 0.0
                self.yaw_rate = 0.0
                
            self.x += self.vx * np.cos(self.heading) * self.dt
            self.y += self.vx * np.sin(self.heading) * self.dt
            self.heading += self.yaw_rate * self.dt
            
        else:
            # 4-Wheel Dynamic Model update
            half_track = self.track_width / 2.0
            
            # 1. Compute local velocity at each wheel (fl, fr, rl, rr)
            vx_fl = self.vx - self.yaw_rate * half_track
            vy_fl = self.vy + self.yaw_rate * self.lf
            
            vx_fr = self.vx + self.yaw_rate * half_track
            vy_fr = self.vy + self.yaw_rate * self.lf
            
            vx_rl = self.vx - self.yaw_rate * half_track
            vy_rl = self.vy - self.yaw_rate * self.lr
            
            vx_rr = self.vx + self.yaw_rate * half_track
            vy_rr = self.vy - self.yaw_rate * self.lr
            
            # 2. Compute slip angles for each wheel
            alpha_fl = self.steering_angle - np.arctan2(vy_fl, vx_fl)
            alpha_fr = self.steering_angle - np.arctan2(vy_fr, vx_fr)
            alpha_rl = -np.arctan2(vy_rl, vx_rl)
            alpha_rr = -np.arctan2(vy_rr, vx_rr)
            
            # 3. Compute static normal forces (ignoring dynamic weight transfer for stability)
            Fz_f = (self.mass * self.gravity * self.lr) / self.wheelbase
            Fz_r = (self.mass * self.gravity * self.lf) / self.wheelbase
            Fz_wheel = Fz_f / 2.0 # simplified per-wheel normal force
            
            # 4. Compute lateral forces using linear tire model
            Fy_fl = (self.Cf / 2.0) * alpha_fl
            Fy_fr = (self.Cf / 2.0) * alpha_fr
            Fy_rl = (self.Cr / 2.0) * alpha_rl
            Fy_rr = (self.Cr / 2.0) * alpha_rr
            
            # Clip forces to friction circle limits
            max_lat_force = self.mu * Fz_wheel
            Fy_fl = np.clip(Fy_fl, -max_lat_force, max_lat_force)
            Fy_fr = np.clip(Fy_fr, -max_lat_force, max_lat_force)
            Fy_rl = np.clip(Fy_rl, -max_lat_force, max_lat_force)
            Fy_rr = np.clip(Fy_rr, -max_lat_force, max_lat_force)
            
            # Combine forces for equations of motion
            FyF = Fy_fl + Fy_fr
            FyR = Fy_rl + Fy_rr
            
            # 5. Equations of motion (Dynamic)
            # ax = Fx/m - vy*yaw_rate - (FyF * sin(delta))/m
            ax = (Fx - FyF * np.sin(self.steering_angle)) / self.mass + self.vy * self.yaw_rate
            
            # ay = (FyF * cos(delta) + FyR)/m - vx*yaw_rate
            ay = (FyF * np.cos(self.steering_angle) + FyR) / self.mass - self.vx * self.yaw_rate
            
            # yaw_accel = (lf * FyF * cos(delta) - lr * FyR) / Iz
            yaw_accel = (self.lf * FyF * np.cos(self.steering_angle) - self.lr * FyR) / self.Iz
            
            # 6. Euler Integration
            self.vx += ax * self.dt
            self.vy += ay * self.dt
            self.yaw_rate += yaw_accel * self.dt
            
            # 7. Update Global Position
            X_dot = self.vx * np.cos(self.heading) - self.vy * np.sin(self.heading)
            Y_dot = self.vx * np.sin(self.heading) + self.vy * np.cos(self.heading)
            
            self.x += X_dot * self.dt
            self.y += Y_dot * self.dt
            self.heading += self.yaw_rate * self.dt
            
        # Normalize heading to [-pi, pi]
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi
        
        return self.get_observation(external_sensors)
        
    def get_observation(self, external_sensors=None):
        """
        Assemble the observation dict.
        """
        dx = self.destination[0] - self.x
        dy = self.destination[1] - self.y
        dist = np.hypot(dx, dy)
        target_heading = np.arctan2(dy, dx)
        rel_heading = target_heading - self.heading
        rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi
        
        # State array now includes vx, vy, and yaw_rate
        state = np.array([
            self.x, self.y, self.vx, self.vy, 
            self.heading, self.yaw_rate, 
            self.steering_angle, self.gear.value
        ], dtype=np.float32)
        
        obs = {
            'state': state,
            'navigation': np.array([self.destination[0], self.destination[1], dist, rel_heading], dtype=np.float32),
            'rgb_camera': np.zeros(self.rgb_resolution, dtype=np.uint8),
            'lidar': np.zeros(self.lidar_rays, dtype=np.float32),
            'radar': np.zeros((self.radar_objects_max, 4), dtype=np.float32)
        }
        
        if external_sensors:
            if 'rgb_camera' in external_sensors: obs['rgb_camera'] = external_sensors['rgb_camera']
            if 'lidar' in external_sensors: obs['lidar'] = external_sensors['lidar']
            if 'radar' in external_sensors: obs['radar'] = external_sensors['radar']
                
        return obs
