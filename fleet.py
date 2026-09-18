"""The fleet — physical parameters for each vehicle type.

Five types, chosen so each one changes the *conflict* rather than the paint.
A vehicle earns a slot by moving one of the four axes that decide how it
interacts with traffic:

    footprint        what gaps it fits through, how much lane it occupies
    accel / brake    gap acceptance at a merge, stopping distance
    turning radius   whether it takes a junction in one movement or swings wide
    roll stability   a high centre of gravity caps cornering before grip does

    python fleet.py        # the table, generated from the specs

Rather than repeat that table here where it would go stale, the short
version: the fleet spans 0.75 m to 2.55 m wide, 180 kg to 12 t, a 2.0 m
turning radius to 8.6 m, and 54 km/h to 180 km/h.

Numbers come from `LANCER3D-dev/vehicle_data.xlsx` where that sheet has
them, corrected where it is internally inconsistent — its sedan brake force
of 1500 N implies 0.95 m/s^2, which is an order of magnitude below what any
car does. Braking is set from what the tyres can actually deliver instead.

## What is NOT modelled, and why

**Roll-over.** A bus corners at about 7.4 m/s^2 before it tips, which is
*below* the ~7.4 m/s^2 its tyres would slide at — so a real bus rolls rather
than slides. Modelling that properly needs a suspension and load-transfer
model, and charging it needs an eighth cost channel, which at six
simultaneous constraints is where plain dual ascent already oscillates.

Instead, tippy vehicles carry a lower `mu`, so they lose grip roughly where
they would have tipped. The failure looks like understeer instead of a roll.
# ponytail: grip-limited stand-in for roll-over; add a load-transfer model
# and a `rollover` cost channel if the paper needs to talk about tipping.

**Leaning.** A motorcycle balances by leaning; here it is a very narrow
four-wheeler. It gets the right footprint, mass and acceleration — which is
what decides whether it fits a gap and how it changes the traffic — and the
wrong reason for staying upright. It is exempt from the roll-stability
`mu` penalty, because leaning is precisely how a real one avoids tipping.

**Articulated vehicles.** A bendy bus or a semi-trailer needs a hitch angle
and a second body, which is different code rather than different numbers.
Left out until an experiment asks for one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GRAVITY = 9.81


@dataclass(frozen=True)
class VehicleSpec:
    """Everything the dynamics and the sensors need about one vehicle type.

    Frozen: a spec is shared by every vehicle of that type in a run, and a
    mutable one would let a single agent's physics drift away from its
    fleet-mates halfway through training.
    """

    name: str
    length: float            # m, nose to tail
    width: float             # m, across the body (mirrors included)
    mass: float              # kg, laden
    wheelbase: float         # m, front axle to rear axle
    cg_fraction: float       # centre of gravity, 0 = front axle, 1 = rear axle
    track_width: float       # m, between wheel centres
    cg_height: float         # m, above the road
    max_steer_deg: float     # at the road wheel, not the steering wheel
    max_accel: float         # m/s^2
    max_brake: float         # m/s^2
    max_speed: float         # m/s
    drag_coef: float
    frontal_area: float      # m^2
    mu: float                # tyre friction coefficient
    leans: bool = False      # exempt from the roll-stability grip penalty

    # -- derived ----------------------------------------------------------

    @property
    def lf(self) -> float:
        """CG to front axle."""
        return self.wheelbase * self.cg_fraction

    @property
    def lr(self) -> float:
        """CG to rear axle."""
        return self.wheelbase * (1.0 - self.cg_fraction)

    @property
    def yaw_inertia(self) -> float:
        """Yaw moment of inertia, as a uniform box. Reproduces the sedan's
        hand-set 3000 kg m^2 to within 2%, so it is not a worse number than
        the one it replaces — it just scales to the rest of the fleet."""
        return self.mass * (self.length ** 2 + self.width ** 2) / 12.0

    @property
    def cornering_stiffness(self) -> float:
        """Per-axle cornering stiffness, scaled with mass so every vehicle
        carries roughly the same tyre load per unit of stiffness. Anchored on
        the sedan's original 100 kN/rad at 1500 kg."""
        return 66.7 * self.mass

    @property
    def rollover_accel(self) -> float:
        """Lateral acceleration at which this vehicle would tip, ignoring
        suspension: `g * (track / 2) / cg_height`. Below `mu * g` means it
        rolls before it slides."""
        return GRAVITY * (self.track_width / 2.0) / self.cg_height

    @property
    def effective_mu(self) -> float:
        """Grip, reduced for a vehicle that would tip before it slid — the
        stand-in for roll-over described in this module's docstring."""
        if self.leans:
            return self.mu
        return min(self.mu, self.rollover_accel / GRAVITY)

    @property
    def turning_radius(self) -> float:
        """Kerb-to-kerb radius at full lock, from the bicycle approximation."""
        return self.wheelbase / math.tan(math.radians(self.max_steer_deg))


SEDAN = VehicleSpec(
    # Exactly the numbers vehicle.py used before there was a fleet, so the
    # baseline vehicle's behaviour is unchanged by this file existing.
    name="sedan", length=4.50, width=1.80, mass=1500.0,
    wheelbase=2.80, cg_fraction=0.43, track_width=1.80, cg_height=0.55,
    max_steer_deg=35.0, max_accel=3.0, max_brake=8.0, max_speed=50.0,
    drag_coef=0.30, frontal_area=2.20, mu=0.90,
)

TUKTUK = VehicleSpec(
    # Auto rickshaw. Tiny, slow, and tippy: the rollover limit works out at
    # 8.7 m/s^2 against a tyre limit of 7.4, so it is right at the edge — it
    # corners on three wheels and everybody in one knows it.
    name="tuktuk", length=2.63, width=1.30, mass=450.0,
    wheelbase=2.00, cg_fraction=0.35, track_width=1.15, cg_height=0.65,
    max_steer_deg=45.0, max_accel=1.5, max_brake=4.5, max_speed=15.0,
    drag_coef=0.44, frontal_area=1.90, mu=0.75,
)

MOTORCYCLE = VehicleSpec(
    # Narrow enough to use a gap no car can. Leans, so no grip penalty.
    name="motorcycle", length=2.00, width=0.75, mass=180.0,
    wheelbase=1.35, cg_fraction=0.48, track_width=0.30, cg_height=0.60,
    max_steer_deg=30.0, max_accel=4.0, max_brake=7.0, max_speed=33.0,
    drag_coef=0.60, frontal_area=0.70, mu=0.85, leans=True,
)

TRUCK = VehicleSpec(
    name="truck", length=7.50, width=2.40, mass=5800.0,
    wheelbase=4.20, cg_fraction=0.35, track_width=2.00, cg_height=1.20,
    max_steer_deg=26.0, max_accel=1.2, max_brake=5.5, max_speed=28.0,
    drag_coef=0.80, frontal_area=6.50, mu=0.75,
)

BUS = VehicleSpec(
    # The thing that changes everyone else's gap acceptance. Twelve metres
    # long, so it cannot clear a junction in the time a car can, and it rolls
    # before it slides.
    name="bus", length=12.20, width=2.55, mass=12000.0,
    wheelbase=6.00, cg_fraction=0.40, track_width=2.10, cg_height=1.40,
    max_steer_deg=35.0, max_accel=1.0, max_brake=5.0, max_speed=22.0,
    drag_coef=0.70, frontal_area=7.30, mu=0.75,
)

FLEET = {spec.name: spec for spec in (SEDAN, TUKTUK, MOTORCYCLE, TRUCK, BUS)}

# A plausible urban mix, used when a run asks for "the fleet" without saying
# how much of each. Weighted toward small vehicles because that is what a
# dense city street actually holds — and because a traffic stream that is
# one-fifth buses is not a traffic stream anyone has driven in.
DEFAULT_MIX = {"sedan": 0.40, "motorcycle": 0.25, "tuktuk": 0.20,
               "truck": 0.10, "bus": 0.05}


def describe() -> str:
    """One line per type — the table in this module's docstring, generated
    from the specs rather than copied, so it cannot go stale."""
    head = (f"{'type':12s} {'L x W':>13s} {'mass':>8s} {'turn r':>7s} "
            f"{'roll':>6s} {'grip':>6s} {'top':>6s}")
    rows = [head, "-" * len(head)]
    for spec in FLEET.values():
        rows.append(
            f"{spec.name:12s} {spec.length:6.2f} x {spec.width:4.2f} "
            f"{spec.mass:7.0f}kg {spec.turning_radius:6.1f}m "
            f"{spec.rollover_accel:5.1f} {spec.effective_mu:6.2f} "
            f"{spec.max_speed * 3.6:5.0f}kmh")
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
