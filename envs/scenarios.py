"""The seven training scenarios, built to real road geometry.

These are the environments the study trains on. The eight older layouts in
`road_network.py` remain buildable and are kept for regression, but these
are the set.

    highway_straight   3-lane dual carriageway, both directions
    cross_two_lane     crossroads of two single-lane-each-way roads
    roundabout         circulatory carriageway with flared entries
    highway_ramps      on-ramp and off-ramp with speed-change lanes
    highway_parking    motorway -> off-ramp -> service road -> car park
    turn_lanes         lane change into dedicated left and right turn lanes
    overbridge         grade separation: a deck over a road, with ramps

## Dimensions are generic, not a national standard

Metric, mid-range values that are recognisably right rather than compliant
with any one country's manual. Every one is a spec field, so a sweep can
move it without touching a builder. `LANE`, `SHOULDER`, `MEDIAN` and the
rest are at the top of this file, in one place.

## Traffic side is a parameter

Nothing here bakes in which side traffic drives on. Carriageways are built
as **one-way pairs** wherever a real road is divided, and which of the pair
carries which direction is decided by the lane graph from
`World(drive_side=...)`. That is also why the dual carriageway is two
one-way pieces rather than one six-lane two-way piece: a motorway has a
barrier down the middle, and a vehicle cannot cross it.

## The map ends where the road continues

All seven have OPEN BOUNDARIES: an arm simply stops at the edge, because
the real road runs on beyond it. They are marked `closed: False` so the
perimeter pass leaves them alone. A bypass loop around a motorway is not a
real place, and neither is a ring road drawn around a single crossroads —
these layouts are excerpts of a road network, not towns. Episodes run end
to end rather than bay to bay.
"""

from __future__ import annotations

import math

from envs.road_network import RoadNetwork, DEFAULT_LANE_WIDTH

# -- generic cross-section, all metres ------------------------------------
LANE = DEFAULT_LANE_WIDTH     # 3.5
SHOULDER = 2.5                # paved shoulder / hard strip
MEDIAN = 5.0                  # between opposing carriageways
BRIDGE_HEIGHT = 5.5           # deck soffit clearance over the road below
RAMP_GRADE = 0.05             # 5%, a comfortable maximum for a road ramp

# Speed-change lanes. A vehicle joining a motorway needs room to reach
# traffic speed before it must merge, and one leaving needs room to slow
# after it has left; too short and the only safe policy is to stop, which
# is not what a slip road is for.
# Speed limits, m/s. A motorway, a slip road, a street and a car park are
# four different roads and a single limit makes three of them wrong.
MOTORWAY_LIMIT = 27.8     # 100 km/h
RAMP_LIMIT = 16.7         # 60 km/h
STREET_LIMIT = 13.9       # 50 km/h
JUNCTION_LIMIT = 11.1     # 40 km/h on a turn lane approach
CARPARK_LIMIT = 5.6       # 20 km/h

ACCEL_LANE = 180.0
DECEL_LANE = 120.0
TAPER = 60.0


def _oneway_pair(net: RoadNetwork, a, b, lanes: int, median: float,
                 z=(0.0, 0.0), stops=(), speed_limit=None):
    """A divided road from `a` to `b`: two one-way carriageways with
    `median` metres of open ground between their kerbs.

    `median` is the gap between the carriageways, NOT the distance between
    their centrelines — which is what it has to be, because the centreline
    separation depends on how wide each carriageway is. Treating the two as
    the same thing put a 5 m gap between the centrelines of two 10.5 m
    carriageways, so the opposing streams overlapped by 5.5 m and drove
    through each other.

    `stops` are fractions along the forward carriageway at which to place
    an intermediate node. A ramp has to join the carriageway WHERE IT
    JOINS, and a piece runs only between its two end nodes — so without a
    node at the merge point a slip road can only be attached to the far end
    of the motorway, which draws it as a diagonal across the whole map
    instead of a merge. That was exactly the bug: 72% of vehicles on the
    ramp scenario ended up off-road.

    Returns `(forward_chain, reverse_nodes)` where `forward_chain` is every
    node along the forward carriageway in order, so `chain[0]` is its start
    and `chain[-1]` its end.
    """
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    tx, ty = dx / length, dy / length
    nx, ny = ty, -tx                      # right of travel

    # Centre to centre: the median plus one full carriageway width.
    half = (median + lanes * net.lane_width) / 2.0
    fractions = [0.0] + sorted(stops) + [1.0]
    chain = []
    for k, t in enumerate(fractions):
        px, py = ax + dx * t, ay + dy * t
        kind = "end" if k in (0, len(fractions) - 1) else "tee"
        chain.append(net.add_node(px + nx * half, py + ny * half, kind))
    for u, v in zip(chain, chain[1:]):
        net.add_straight(u, v, lanes, oneway=True, z=z,
                         speed_limit=speed_limit)

    r0 = net.add_node(bx - nx * half, by - ny * half, "end")
    r1 = net.add_node(ax - nx * half, ay - ny * half, "end")
    net.add_straight(r0, r1, lanes, oneway=True, z=(z[1], z[0]),
                     speed_limit=speed_limit)
    return chain, (r0, r1)


def _curve(net: RoadNetwork, a: int, b: int, a_dir, b_dir, lanes: int,
           oneway: bool = False, steps: int = 6, speed_limit=None):
    """A smooth bend from node `a` to node `b`, entering along `a_dir` and
    leaving along `b_dir`, as a short chain of straights.

    Slip roads curve. Joining a slip road to an acceleration lane with one
    straight put a 25-degree kink in the middle of a 100 km/h manoeuvre,
    which nothing can track and which showed up as 80% of vehicles on the
    ramp scenario leaving the road.

    A cubic Hermite through the two poses, sampled into segments, because
    every query in `RoadNetwork` is piecewise anyway — an exact clothoid
    would be the real thing and would buy nothing a vehicle can feel.
    """
    ax, ay = net.nodes[a]
    bx, by = net.nodes[b]
    span = math.hypot(bx - ax, by - ay)
    # Tangent magnitude sets how hard the curve leans into each end.
    m0 = (a_dir[0] * span * 0.6, a_dir[1] * span * 0.6)
    m1 = (b_dir[0] * span * 0.6, b_dir[1] * span * 0.6)

    previous = a
    for k in range(1, steps + 1):
        t = k / steps
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        px = h00 * ax + h10 * m0[0] + h01 * bx + h11 * m1[0]
        py = h00 * ay + h10 * m0[1] + h01 * by + h11 * m1[1]
        node = b if k == steps else net.add_node(px, py, "tee")
        net.add_straight(previous, node, lanes, oneway=oneway,
                         speed_limit=speed_limit)
        previous = node


def _source_goal(net: RoadNetwork, source_xy, toward_xy, goal_xy):
    """Declare one end-to-end journey."""
    dx = toward_xy[0] - source_xy[0]
    dy = toward_xy[1] - source_xy[1]
    d = math.hypot(dx, dy) or 1.0
    net.add_source(source_xy[0], source_xy[1],
                   math.degrees(math.atan2(-dx / d, dy / d)))
    net.add_goal(*goal_xy)


# ── 1. straight dual carriageway ─────────────────────────────────────────

def build_highway_straight(net: RoadNetwork, spec: dict) -> None:
    """Three lanes each way, divided, running the length of the map.

    The simplest thing a motorway is, and the one every other highway
    scenario is a variation on: hold a lane at speed, overtake when the
    lane ahead is slower, do not drift into the barrier.
    """
    length = float(spec["length"])
    lanes = int(spec["lanes"])
    median = float(spec.get("median", MEDIAN))

    chain, (r0, r1) = _oneway_pair(
        net, (-length / 2.0, 0.0), (length / 2.0, 0.0), lanes, median,
        speed_limit=MOTORWAY_LIMIT)
    f0, f1 = chain[0], chain[-1]

    # One journey each way, so both carriageways carry traffic.
    _source_goal(net, net.nodes[f0], net.nodes[f1], net.nodes[f1])
    _source_goal(net, net.nodes[r0], net.nodes[r1], net.nodes[r1])


# ── 2. crossroads of two single-lane-each-way roads ──────────────────────

def build_cross_two_lane(net: RoadNetwork, spec: dict) -> None:
    """Two two-lane roads meeting at a square, unsignalised crossroads.

    One lane each way on both arms, so every turn crosses opposing traffic
    and there is nowhere to wait: the conflict is unavoidable rather than
    negotiable, which is what makes it worth training on.
    """
    arm = float(spec["arm"])
    lanes = int(spec["lanes"])

    centre = net.add_node(0.0, 0.0, "cross")
    ends = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        node = net.add_node(dx * arm, dy * arm, "end")
        net.add_straight(centre, node, lanes)
        ends.append(node)

    _source_goal(net, net.nodes[ends[1]], net.nodes[centre], net.nodes[ends[0]])
    _source_goal(net, net.nodes[ends[3]], net.nodes[centre], net.nodes[ends[2]])
    _source_goal(net, net.nodes[ends[0]], net.nodes[centre], net.nodes[ends[3]])


# ── 3. roundabout ────────────────────────────────────────────────────────

def build_roundabout(net: RoadNetwork, spec: dict) -> None:
    """A circulatory carriageway with approach arms.

    The ring is ONE-WAY, which is the whole character of a roundabout:
    everything on it is going the same way, and entering means finding a gap
    in a stream rather than crossing it. Which way round follows from the
    lane graph's traffic side.

    The central island is non-drivable and declared as such, so cutting
    across the middle is off-road rather than a shortcut.
    """
    radius = float(spec["radius"])
    arms = int(spec["arms"])
    arm = float(spec["arm"])
    ring_lanes = int(spec.get("ring_lanes", 2))
    arm_lanes = int(spec["lanes"])

    ring = []
    for k in range(arms):
        angle = 2.0 * math.pi * k / arms
        ring.append(net.add_node(radius * math.cos(angle),
                                 radius * math.sin(angle), "round"))

    for k in range(arms):
        a0 = 2.0 * math.pi * k / arms
        net.add_arc(ring[k], ring[(k + 1) % arms], (0.0, 0.0), radius,
                    a0, 2.0 * math.pi / arms, ring_lanes, oneway=True)

        angle = a0
        outer = net.add_node((radius + arm) * math.cos(angle),
                             (radius + arm) * math.sin(angle), "end")
        net.add_straight(ring[k], outer, arm_lanes)

    # The island: everything inside the ring's inner kerb.
    net.islands.append((0.0, 0.0, radius - ring_lanes * net.lane_width / 2.0))

    first = (radius + arm, 0.0)
    across = ((radius + arm) * math.cos(math.pi * 2 / arms * (arms // 2)),
              (radius + arm) * math.sin(math.pi * 2 / arms * (arms // 2)))
    _source_goal(net, first, (radius, 0.0), across)
    _source_goal(net, across, (0.0, 0.0), first)


# ── 4. motorway entry and exit ───────────────────────────────────────────

def build_highway_ramps(net: RoadNetwork, spec: dict) -> None:
    """A dual carriageway with an on-ramp and an off-ramp.

    Both are modelled the way they are built: a slip road, then a
    SPEED-CHANGE LANE running alongside the carriageway, then a taper. The
    speed-change lane is the part that matters — merging is a negotiation
    conducted at speed over a couple of hundred metres, and a scenario that
    joins a slip road straight onto a carriageway is asking for a different
    and much harder manoeuvre than the real one.
    """
    length = float(spec["length"])
    lanes = int(spec["lanes"])
    median = float(spec.get("median", MEDIAN))
    ramp_lanes = int(spec.get("ramp_lanes", 1))

    # Nodes ON the carriageway where the slip roads actually join it: the
    # merge first, the diverge afterwards, so one journey can do both.
    chain, (r0, r1) = _oneway_pair(
        net, (-length / 2.0, 0.0), (length / 2.0, 0.0), lanes, median,
        stops=(0.35, 0.65), speed_limit=MOTORWAY_LIMIT)
    f0, n_merge, n_diverge, f1 = chain

    # Slip roads leave on the OUTSIDE of the carriageway, away from the
    # median. Getting this backwards put the acceleration lane in the
    # central reserve between the two opposing streams, which is both
    # off-road and the one place on a motorway nothing may be.
    fy = net.nodes[f0][1]
    side = math.copysign(1.0, fy)
    edge = fy + side * (lanes * net.lane_width / 2.0 + LANE / 2.0)
    merge_x = net.nodes[n_merge][0]
    diverge_x = net.nodes[n_diverge][0]

    # -- on-ramp: slip road, acceleration lane, merge taper -------------
    slip_in = net.add_node(merge_x - ACCEL_LANE - TAPER * 2.0,
                           edge + side * 30.0, "end")
    accel_a = net.add_node(merge_x - ACCEL_LANE, edge, "tee")
    # Curved slip road, straight acceleration lane, then the merge.
    _curve(net, slip_in, accel_a, (1.0, 0.0), (1.0, 0.0), ramp_lanes,
           oneway=True, speed_limit=RAMP_LIMIT)
    net.add_straight(accel_a, n_merge, ramp_lanes, oneway=True, speed_limit=RAMP_LIMIT)

    # -- off-ramp: diverge taper, deceleration lane, slip road ----------
    decel_b = net.add_node(diverge_x + DECEL_LANE, edge, "tee")
    slip_out = net.add_node(diverge_x + DECEL_LANE + TAPER * 2.0,
                            edge + side * 30.0, "end")
    net.add_straight(n_diverge, decel_b, ramp_lanes, oneway=True, speed_limit=RAMP_LIMIT)
    _curve(net, decel_b, slip_out, (1.0, 0.0), (1.0, 0.0), ramp_lanes,
           oneway=True, speed_limit=RAMP_LIMIT)

    # Join from the slip road and leave by the other: the episode is the
    # whole merge-cruise-diverge sequence.
    _source_goal(net, net.nodes[slip_in], net.nodes[accel_a],
                 net.nodes[slip_out])
    _source_goal(net, net.nodes[f0], net.nodes[n_merge], net.nodes[f1])
    _source_goal(net, net.nodes[r0], net.nodes[r1], net.nodes[r1])


# ── 5. motorway to car park ──────────────────────────────────────────────

def build_highway_parking(net: RoadNetwork, spec: dict) -> None:
    """Leave the motorway, cross a service road, park. And back out again.

    The scenario spans the whole speed range in one episode: 100 km/h on the
    carriageway, 50 on the slip road, walking pace between parked cars. A
    policy that can only do one of those cannot finish it.
    """
    length = float(spec["length"])
    lanes = int(spec["lanes"])
    median = float(spec.get("median", MEDIAN))
    bays = int(spec.get("bays", 6))
    aisle = float(spec.get("aisle", 60.0))

    chain, (r0, r1) = _oneway_pair(
        net, (-length / 2.0, 0.0), (length / 2.0, 0.0), lanes, median,
        stops=(0.4,), speed_limit=MOTORWAY_LIMIT)
    f0, n_diverge, f1 = chain

    # Outside the carriageway, away from the median — see build_highway_ramps.
    fy = net.nodes[f0][1]
    side = math.copysign(1.0, fy)
    edge = fy + side * (lanes * net.lane_width / 2.0 + LANE / 2.0)
    diverge_x = net.nodes[n_diverge][0]

    # Off-ramp down to a service road, leaving the carriageway where the
    # diverge node actually is rather than at its far end.
    decel_b = net.add_node(diverge_x + DECEL_LANE, edge, "tee")
    service_y = edge + side * 55.0
    service_a = net.add_node(diverge_x + DECEL_LANE + TAPER * 2.0,
                             service_y, "tee")
    net.add_straight(n_diverge, decel_b, 1, oneway=True, speed_limit=RAMP_LIMIT)
    _curve(net, decel_b, service_a, (1.0, 0.0), (0.0, side), 1,
           oneway=True, speed_limit=RAMP_LIMIT)

    # The car park: one aisle, chained through a node per rank of bays so
    # the whole lot is a single connected road rather than a set of stubs
    # hanging off one long piece.
    lot_y = service_y + side * 30.0
    x0 = diverge_x + DECEL_LANE + TAPER * 2.0
    lot_a = net.add_node(x0, lot_y, "tee")
    lot_b = net.add_node(x0 + aisle, lot_y, "end")
    net.add_straight(service_a, lot_a, 2, speed_limit=CARPARK_LIMIT)

    bay_nodes = []
    previous = lot_a
    for k in range(bays):
        bx = x0 + aisle * (k + 0.5) / bays
        entry = net.add_node(bx, lot_y, "tee")
        net.add_straight(previous, entry, 2, speed_limit=CARPARK_LIMIT)
        previous = entry
        for bay_side in (1.0, -1.0):
            end = net.add_node(bx, lot_y + bay_side * side * 9.0, "end")
            net.add_straight(entry, end, 1, speed_limit=CARPARK_LIMIT)
            bay_nodes.append(end)
    net.add_straight(previous, lot_b, 2, speed_limit=CARPARK_LIMIT)

    _source_goal(net, net.nodes[f0], net.nodes[n_diverge],
                 net.nodes[bay_nodes[0]])
    _source_goal(net, net.nodes[bay_nodes[-1]], net.nodes[lot_a],
                 net.nodes[f1])


# ── 6. lane change into turn lanes ───────────────────────────────────────

def build_turn_lanes(net: RoadNetwork, spec: dict) -> None:
    """An approach that widens into dedicated left, ahead and right lanes.

    The manoeuvre is the point: a vehicle arriving in the wrong lane has to
    change lanes, in traffic, before the lanes divide. That puts `lane_keep`
    and `wrong_way` in direct tension with reaching the goal, which is the
    conflict the constrained formulation exists to arbitrate.

    **Each turn lane feeds only its own exit.** They are separate one-way
    pieces that do not share a node inside the junction, so being in the
    right-turn lane and wanting to go left is not a tight turn — it is a
    route that does not exist. That is what makes the lane choice binding
    rather than advisory, and it is how a real signalised approach works.

    The lanes run PARALLEL for the length of the flare rather than fanning
    out from a point; a wedge would let a vehicle drift between movements
    all the way to the stop line.
    """
    arm = float(spec["arm"])
    lanes = int(spec["lanes"])
    flare = float(spec.get("flare", 70.0))
    taper = float(spec.get("taper", 25.0))
    stop_line = 8.0

    # The shared approach, before the lanes are designated.
    approach_end = net.add_node(-arm, 0.0, "end")
    divide = net.add_node(-flare, 0.0, "tee")
    net.add_straight(approach_end, divide, lanes)

    # The junction itself: a stub on each exit side, close in, so a turn is
    # a turn through the junction rather than a 150 m diagonal to the far
    # end of the map.
    box = 12.0
    j_north = net.add_node(0.0, box, "tee")
    j_south = net.add_node(0.0, -box, "tee")
    j_east = net.add_node(box, 0.0, "tee")

    movements = (("left", LANE, j_north), ("ahead", 0.0, j_east),
                 ("right", -LANE, j_south))
    for _name, offset, junction_exit in movements:
        # Taper out to the lane's own offset, run PARALLEL to the stop line,
        # then through the junction to that movement's exit alone.
        gate = net.add_node(-flare + taper, offset, "tee")
        stop = net.add_node(-stop_line, offset, "tee")
        net.add_straight(divide, gate, 1, oneway=True, speed_limit=JUNCTION_LIMIT)
        net.add_straight(gate, stop, 1, oneway=True, speed_limit=JUNCTION_LIMIT)
        net.add_straight(stop, junction_exit, 1, oneway=True, speed_limit=JUNCTION_LIMIT)

    # The exit roads, and the two-way crossing road that runs north-south
    # through the junction and supplies the conflicting traffic a turn has
    # to yield to.
    north = net.add_node(0.0, arm, "end")
    south = net.add_node(0.0, -arm, "end")
    east = net.add_node(arm, 0.0, "end")
    net.add_straight(j_north, north, lanes)
    net.add_straight(j_south, south, lanes)
    net.add_straight(j_east, east, lanes)
    net.add_straight(j_north, j_south, lanes)

    exits = {"left": north, "ahead": east, "right": south}

    # One journey per movement, so the fleet uses all three turn lanes and
    # some of it is always in the wrong one to begin with.
    for name, _offset, _exit in movements:
        _source_goal(net, net.nodes[approach_end], net.nodes[divide],
                     net.nodes[exits[name]])
    # And traffic on the crossing road, for the turns to conflict with.
    _source_goal(net, net.nodes[north], net.nodes[j_north], net.nodes[south])


# ── 7. overbridge ────────────────────────────────────────────────────────

def build_overbridge(net: RoadNetwork, spec: dict) -> None:
    """One road carried over another on a deck, with ramps up and down.

    The only layout with elevation, and the reason elevation exists: the
    deck and the road beneath share x and y, so without a height test every
    vehicle on the bridge would collide with one underneath and every lidar
    would return the deck as a wall across the road.

    Ramp gradients are held at a realistic 5%, which fixes the ramp length
    from the clearance: a 5.5 m deck needs 110 m of ramp at each end. A
    steeper ramp would be shorter and would not be a road.
    """
    length = float(spec["length"])
    lanes = int(spec["lanes"])
    clearance = float(spec.get("clearance", BRIDGE_HEIGHT))
    deck = float(spec.get("deck", 70.0))
    ramp = clearance / RAMP_GRADE

    # The road underneath, at grade, running across.
    under_a = net.add_node(0.0, -length / 2.0, "end")
    under_b = net.add_node(0.0, length / 2.0, "end")
    net.add_straight(under_a, under_b, lanes)

    # The road over the top: approach, ramp up, deck, ramp down, approach.
    span = deck / 2.0
    x = [-span - ramp - length * 0.2, -span - ramp, -span, span,
         span + ramp, span + ramp + length * 0.2]
    nodes = [net.add_node(v, 0.0, "end" if i in (0, 5) else "tee")
             for i, v in enumerate(x)]
    heights = [0.0, 0.0, clearance, clearance, 0.0, 0.0]
    for i in range(5):
        net.add_straight(nodes[i], nodes[i + 1], lanes,
                         z=(heights[i], heights[i + 1]))

    _source_goal(net, net.nodes[nodes[0]], net.nodes[nodes[1]],
                 net.nodes[nodes[5]])
    _source_goal(net, net.nodes[under_a], net.nodes[under_b],
                 net.nodes[under_b])


# ── registry ─────────────────────────────────────────────────────────────

BUILDERS = {
    "highway_straight": build_highway_straight,
    "cross_two_lane": build_cross_two_lane,
    "roundabout_v2": build_roundabout,
    "highway_ramps": build_highway_ramps,
    "highway_parking": build_highway_parking,
    "turn_lanes": build_turn_lanes,
    "overbridge": build_overbridge,
}

# `closed` says whether the perimeter pass may ring this layout. A motorway
# that loops back on itself is not a real place, so the highway scenarios
# end at the map edge and their journeys run end to end.
# `attach_bays: False` on every one of them. These layouts declare their own
# sources and goals — the end of a carriageway, a bay in the car park, the
# far side of a junction — and the bay pass REPLACES a network's endpoints
# with the bays it creates. Letting it run would delete the scenario and
# leave a motorway whose task is to park on the hard shoulder.
DEFAULTS = {
    "highway_straight": {"length": 1400.0, "lanes": 3, "median": MEDIAN,
                         "closed": False, "attach_bays": False},
    "cross_two_lane":   {"arm": 140.0, "lanes": 2, "closed": False, "attach_bays": False},
    "roundabout_v2":    {"radius": 22.0, "arms": 4, "arm": 130.0,
                         "lanes": 2, "ring_lanes": 2, "closed": False,
                         "attach_bays": False},
    "highway_ramps":    {"length": 1400.0, "lanes": 3, "median": MEDIAN,
                         "ramp_lanes": 1, "closed": False, "attach_bays": False},
    "highway_parking":  {"length": 1000.0, "lanes": 3, "median": MEDIAN,
                         "bays": 6, "aisle": 60.0, "closed": False, "attach_bays": False},
    "turn_lanes":       {"arm": 150.0, "lanes": 3, "flare": 70.0,
                         "closed": False, "attach_bays": False},
    "overbridge":       {"length": 400.0, "lanes": 2,
                         "clearance": BRIDGE_HEIGHT, "deck": 70.0,
                         "closed": False, "attach_bays": False},
}

KINDS = tuple(BUILDERS)
