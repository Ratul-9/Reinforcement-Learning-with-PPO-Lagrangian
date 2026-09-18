import numpy as np
from enum import Enum
import gymnasium as gym
from gymnasium import spaces

from fleet import SEDAN, VehicleSpec

# Physics sub-steps per control step. Sized so the stiff lateral dynamics
# stay stable down to the speed at which the kinematic model takes over.
PHYSICS_SUBSTEPS = 10

class Gear(Enum):
    PARK = 0
    REVERSE = 1
    NEUTRAL = 2
    DRIVE = 3

class Vehicle:
    """
    Vehicle model for multi-agent RL (e.g. PPO Lagrangian).
    Uses a 4-Wheel Dynamic Vehicle Model for realistic physics (slip angles,
    lateral forces, mass inertia, and track width).
    Designed to be a plug-and-play component for any grid/continuous environment.

    The physics is one model for the whole fleet; what differs between a
    motorcycle and a bus is the `VehicleSpec` handed in, not the equations.
    A spec that is twelve metres long with a high centre of gravity produces
    a vehicle that understeers, brakes slowly and cannot clear a junction in
    the time a car can, purely out of its own numbers.

    `Sedan` below is this class bound to the sedan spec, so every existing
    caller keeps working and keeps the behaviour it had before the fleet
    existed.
    """
    def __init__(self, spawn_point=(0.0, 0.0), destination=(100.0, 100.0), dt=0.1,
                 spec: VehicleSpec = SEDAN, actuator_lag: bool = True):
        # Environment properties
        self.dt = dt
        self.spec = spec
        # Actuators respond over time, not instantly. Off only for
        # ablations and for tests that need one exact command applied.
        self.actuator_lag = bool(actuator_lag)
        # See `step` for why this is 10 and not 1.
        self.substeps = PHYSICS_SUBSTEPS
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
        # Actuator states: what the vehicle is ACTUALLY doing, as opposed to
        # what it was last told to do.
        self.throttle = 0.0
        self.brake = 0.0
        
        # --- Physical Parameters, from the spec ---
        self.mass = spec.mass # kg
        self.Iz = spec.yaw_inertia # Yaw moment of inertia (kg*m^2)
        self.lf = spec.lf # Distance from CG to front axle (m)
        self.lr = spec.lr # Distance from CG to rear axle (m)
        self.track_width = spec.track_width # m
        self.wheelbase = spec.wheelbase
        self.length = spec.length
        self.width = spec.width

        self.max_steer_angle = np.radians(spec.max_steer_deg)
        self.max_steer_rate = np.radians(spec.steer_rate_deg)
        self.steer_tau = spec.steer_tau
        self.throttle_tau = spec.throttle_tau
        self.brake_tau = spec.brake_tau
        self.max_accel = spec.max_accel # m/s^2 (corresponds to engine force)
        self.max_brake = spec.max_brake # m/s^2
        self.max_speed = spec.max_speed # m/s, governed
        self.drag_coef = spec.drag_coef # Aerodynamic drag coefficient
        self.frontal_area = spec.frontal_area # m^2
        self.air_density = 1.225 # kg/m^3
        self.gravity = 9.81

        # Tire parameters (Linear tire model with friction circle)
        self.Cf = spec.cornering_stiffness # Front cornering stiffness (N/rad)
        self.Cr = spec.cornering_stiffness # Rear cornering stiffness (N/rad)
        # Effective, not nominal: a vehicle that would tip before it slid
        # carries reduced grip as a stand-in for roll-over. See fleet.py.
        self.mu = spec.effective_mu
        
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
        # Actuator states: what the vehicle is ACTUALLY doing, as opposed to
        # what it was last told to do.
        self.throttle = 0.0
        self.brake = 0.0
        
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
            
        steer_target = steering_input * self.max_steer_angle
        if not self.actuator_lag:
            self.steering_angle = steer_target
            self.throttle = throttle_input
            self.brake = brake_input

        # Integrate the physics on a finer grid than the control period.
        #
        # This is not polish, it is correctness. The lateral dynamics are
        # stiff: explicit Euler needs dt * (Cf + Cr) / (m * vx) < 2, i.e.
        # vx > 6.7 m/s at a 0.1 s step. Below that the integration is
        # unstable, and because the friction clip bounds the force it does
        # not blow up — it settles into a limit cycle that looks like
        # plausible noise. Measured before this existed, holding a constant
        # 0.3 steering input at 3 m/s: the bus locked into a two-cycle
        # (+0.10, +0.14, +0.10, ...) and the tuktuk oscillated between -0.19
        # and +1.16 rad/s, changing sign 20 times in 24 steps. A parking
        # manoeuvre lives entirely in that band.
        #
        # At dt/10 the bound falls to 0.67 m/s, below the speed at which the
        # kinematic fallback takes over, so the whole operating range is
        # stable. Forces are recomputed per sub-step because drag and
        # braking both depend on the speed being integrated.
        sub = self.dt / self.substeps
        for _ in range(self.substeps):
            if self.actuator_lag:
                self._actuate(sub, steer_target, throttle_input, brake_input)
            self._advance(sub, self.throttle, self.brake, slope_angle)

        # Governor and heading wrap apply once per control step, not per
        # sub-step: they are limits on the reported state, not forces.
        self.vx = float(np.clip(self.vx, -self.max_speed, self.max_speed))
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        return self.get_observation(external_sensors)

    def _actuate(self, dt, steer_target, throttle_target, brake_target):
        """Move the actuators toward what they were commanded.

        Nothing on a vehicle responds instantly, and a policy trained
        against actuators that do will learn to depend on a response no real
        vehicle has — the classic symptom being high-frequency steering
        chatter that transfers to hardware as a shaking wheel.

        Steering gets a RATE LIMIT as well as a lag, because the two
        constrain different things: the lag is how long the assistance takes
        to build, the rate is the hard ceiling on how fast the wheel can be
        turned at all. A first-order lag alone still allows an arbitrarily
        fast initial move.

        Throttle and brake are first-order only. The brake constant is where
        the fleet separates: 0.08 s on a motorcycle's disc against 0.45 s
        for a bus's air lines, which at 20 m/s is nine metres of travel
        before retardation even begins.
        """
        # Plain arithmetic rather than np.clip: this runs once per sub-step
        # per agent, so ten times per control step, and numpy's dispatch
        # overhead on a scalar dwarfs the work itself — it measured ~20% of
        # total env throughput.
        alpha = dt / (self.steer_tau + dt)
        delta = alpha * (steer_target - self.steering_angle)
        limit = self.max_steer_rate * dt
        if delta > limit:
            delta = limit
        elif delta < -limit:
            delta = -limit
        self.steering_angle += delta

        a_thr = dt / (self.throttle_tau + dt)
        a_brk = dt / (self.brake_tau + dt)
        self.throttle += a_thr * (throttle_target - self.throttle)
        self.brake += a_brk * (brake_target - self.brake)

    def _advance(self, dt, throttle_input, brake_input, slope_angle):
        """One physics sub-step of length `dt`."""
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
            vx_before = self.vx
            if self.gear == Gear.PARK:
                self.vx = 0.0
            else:
                self.vx += acceleration * dt
                
            self.vy = 0.0
            self.yaw_rate = (self.vx / self.wheelbase) * np.tan(self.steering_angle)
            
            # Come to a complete stop rather than creeping or juddering —
            # but as a physical rule, not a speed threshold.
            #
            # This used to be `abs(vx) < 0.1 and braking -> vx = 0`, which is
            # frame-rate dependent in a way that only showed up once the
            # integrator was sub-stepped: with dt = 0.01 a vehicle gains
            # 0.03 m/s per sub-step, so it can never clear a 0.1 m/s gate
            # that is re-applied every sub-step, and a vehicle holding both
            # throttle and brake stayed pinned at exactly zero forever.
            #
            # The rule brakes are actually subject to is that they can bring
            # you to rest but cannot push you backwards, so the test is
            # whether this sub-step would have reversed the direction of
            # travel. That is identical at every `dt`.
            # `vx_before != 0` matters: without it, a vehicle pulling away
            # from rest with any brake applied has vx_before * vx == 0 and
            # gets pinned at zero forever. Brakes resist motion; they do not
            # prevent a standing start.
            stopping = brake_input > 0 or self.gear == Gear.PARK
            if stopping and vx_before != 0.0 and vx_before * self.vx <= 0.0:
                self.vx = 0.0
                self.yaw_rate = 0.0
                
            self.x += self.vx * np.cos(self.heading) * dt
            self.y += self.vx * np.sin(self.heading) * dt
            self.heading += self.yaw_rate * dt
            
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
            Fz_front_wheel = Fz_f / 2.0
            Fz_rear_wheel = Fz_r / 2.0
            
            # 4. Compute lateral forces using linear tire model
            Fy_fl = (self.Cf / 2.0) * alpha_fl
            Fy_fr = (self.Cf / 2.0) * alpha_fr
            Fy_rl = (self.Cr / 2.0) * alpha_rl
            Fy_rr = (self.Cr / 2.0) * alpha_rr
            
            # Clip each tyre to the grip its OWN axle load can supply.
            # Fz_r was previously computed and then never used, so the rear
            # tyres were allowed the front axle's limit — on a
            # front-heavy car that is ~33% more rear grip than physical,
            # which biases every vehicle toward understeer for free.
            max_front = self.mu * Fz_front_wheel
            max_rear = self.mu * Fz_rear_wheel
            Fy_fl = np.clip(Fy_fl, -max_front, max_front)
            Fy_fr = np.clip(Fy_fr, -max_front, max_front)
            Fy_rl = np.clip(Fy_rl, -max_rear, max_rear)
            Fy_rr = np.clip(Fy_rr, -max_rear, max_rear)
            
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
            self.vx += ax * dt
            self.vy += ay * dt
            self.yaw_rate += yaw_accel * dt
            
            # 7. Update Global Position
            X_dot = self.vx * np.cos(self.heading) - self.vy * np.sin(self.heading)
            Y_dot = self.vx * np.sin(self.heading) + self.vy * np.cos(self.heading)
            
            self.x += X_dot * dt
            self.y += Y_dot * dt
            self.heading += self.yaw_rate * dt
            
        
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


class Sedan(Vehicle):
    """The original vehicle, unchanged: `Vehicle` bound to the sedan spec.

    Kept as its own name because `Vehicle.md`, the tests and the env all
    refer to it, and because "the baseline vehicle" is worth being able to
    say in one word.
    """

    def __init__(self, spawn_point=(0.0, 0.0), destination=(100.0, 100.0), dt=0.1):
        super().__init__(spawn_point, destination, dt, spec=SEDAN)


def make(type_name: str, **kwargs) -> Vehicle:
    """A vehicle by fleet name — `make("bus")`, `make("tuktuk")`."""
    from fleet import FLEET

    if type_name not in FLEET:
        raise ValueError(f"unknown vehicle type {type_name!r}. "
                         f"Known: {', '.join(FLEET)}")
    return Vehicle(spec=FLEET[type_name], **kwargs)
