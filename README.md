# Autonomous Robot Navigation System with EKF-SLAM


## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Hardware Components](#hardware-components)
4. [Software Pipeline](#software-pipeline)
5. [Installation & Setup](#installation--setup)


---

## Project Overview

This project implements an autonomous mobile robot system that combines computer vision, simultaneous localization and mapping (SLAM), path planning, and motion control to navigate unknown environments. The robot can explore its surroundings, detect and manipulate colored blocks, and deliver them to matching colored gates.

### Key Capabilities
- **Autonomous Exploration**: Systematically explores unknown environments to build a map
- **EKF-SLAM**: Real-time localization and mapping using Extended Kalman Filter
- **ArUco Marker Detection**: Uses ArUco markers for precise localization and object identification
- **Path Planning**: A* algorithm with safety-aware cost functions for optimal navigation
- **Block Manipulation**: Detects, picks up, and delivers colored blocks to matching gates
- **Obstacle Avoidance**: Grid-based mapping with safety margins for collision-free navigation

---

## System Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Control Loop                     │
│  (main.py - State Machine with Multiple Operating Modes)    │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴────────┬──────────────┬──────────────┬─────────┐
    │                 │              │              │         │
┌───▼────┐    ┌──────▼──────┐  ┌───▼────┐  ┌─────▼────┐  ┌─▼──────┐
│ Vision │    │ EKF-SLAM    │  │ Path   │  │ Motion   │  │ Gripper│
│ System │    │             │  │Planning│  │ Control  │  │Control │
└───┬────┘    └──────┬──────┘  └───┬────┘  └─────┬────┘  └────────┘
    │                │              │              │
    │         ┌──────▼──────┐  ┌───▼────────┐    │
    │         │ Exploration │  │ Grid Map   │    │
    │         │  Strategy   │  │            │    │
    │         └─────────────┘  └────────────┘    │
    │                                             │
┌───▼─────────────────────────────────────────────▼──────┐
│           Robot Controller (EV3 Interface)              │
└─────────────────────────────────────────────────────────┘
```

### Component Communication
- **Publisher/Subscriber Pattern**: Camera images and robot state are transmitted to a visualization viewer
- **ZMQ Messaging**: Asynchronous communication between robot and remote viewer
- **Multi-threaded Design**: Camera runs in separate thread for optimal frame rates

---

## Hardware Components

### Robot Platform
- **LEGO Mindstorms EV3**: Main computing platform
- **Two-wheel differential drive**: Enables precise turning and movement control
- **Motor gripper**: For block manipulation

### Sensors
- **USB Camera** (864×480 @ 30fps)
  - Model: Sonix Technology USB 2.0 Camera
  - Field of View: ~110°
  - Configurable exposure and gain for different lighting conditions

### Camera Calibration
The system uses pre-calibrated camera intrinsics:
- **Camera Matrix**: 
  ```
  [635.93,  0,      444.27]
  [0,       634.26, 256.84]
  [0,       0,      1     ]
  ```
- **Distortion Coefficients**: Corrects for lens distortion
- Calibration performed using standard checkerboard pattern

---

## Software Pipeline

### 1. Image Acquisition & Processing

#### Camera Thread
The camera operates in a dedicated thread to ensure real-time image capture:

```python
# Camera continuously captures frames at 30 FPS
Camera Thread → LifoQueue → Main Loop (latest frame)
```

**Key Features:**
- Thread-safe LIFO queue ensures latest frame is always used
- Configurable exposure time (1-5000) and gain (0-100)
- Motion-JPEG compression for efficient transmission
- Auto-flip for inverted camera mounting

#### ArUco Marker Detection
The vision system detects ArUco markers from the original dictionary:

```python
Input: Raw BGR Image (864×480)
  ↓
ArUco Detection (cv2.aruco.detectMarkers)
  ↓
Pose Estimation (cv2.solvePnP)
  ↓
Output: [marker_id, distance (r), angle (α), world position (x,y)]
```

**Marker Categories:**
- **ID 1-300**: Boundary markers (environment perimeter)
- **ID 301-400**: Obstacle markers  
- **ID 401-420**: Red gate markers
- **ID 421-450**: Red block markers (pairs)
- **ID 501-520**: Blue gate markers
- **ID 521-550**: Blue block markers (pairs)

### 2. Coordinate Transformations

The system performs a series of coordinate transformations to map markers from camera space to world space:

```
Camera Frame → Robot Frame → World Frame
```

#### Transformation Pipeline

**Step 1: Camera to Robot Transform**
```python
# Camera mounted at offset from robot center
T_camera_robot = [
    Rotation: (x=-124°, y=0°, z=-90°)
    Translation: (x=0.03m, y=0m, z=0.28m)
]
```

**Step 2: Robot to World Transform**  
```python
# Robot's current pose in world coordinates
T_robot_world = [
    Rotation: (x=-30°, y=0°, z=0°)
    Translation: (x, y, θ) from SLAM
]
```

**Step 3: Combined Transformation**
```python
T_marker_world = T_robot_world @ T_camera_robot @ T_marker_camera
```

The vision system computes:
- **Distance (r)**: Euclidean distance from robot to marker
- **Bearing angle (α)**: Angle relative to robot's heading
- **World position (x, y)**: Absolute 2D coordinates in world frame

### 3. EKF-SLAM (Simultaneous Localization and Mapping)

The core localization system uses an Extended Kalman Filter to simultaneously estimate:
- Robot pose: (x, y, θ)
- Landmark positions: (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)

#### State Vector
```
μ = [robot_x, robot_y, robot_θ, 
     landmark₁_x, landmark₁_y, 
     landmark₂_x, landmark₂_y, 
     ..., 
     landmarkₙ_x, landmarkₙ_y]ᵀ
```

#### Covariance Matrix
```
Σ = [Robot covariance (3×3)      | Robot-Landmark cross-covariance  ]
    [Landmark-Robot covariance   | Landmark covariances (2n×2n)     ]
```

#### Prediction Step (Motion Model)

When the robot moves, wheel encoders measure left and right wheel rotations (l, r):

**For Circular Motion** (|r - l| > threshold):
```python
α = (r - l) / WIDTH          # Change in heading
R = l / α                     # Turning radius

# Update position
x_new = x - R·sin(θ) + R·sin(θ + α)
y_new = y + R·cos(θ) - R·cos(θ + α)
θ_new = (θ + α) mod 2π

# Jacobian for covariance propagation
G = [1  0  -R·cos(θ) + R·cos(θ+α)]
    [0  1  -R·sin(θ) + R·sin(θ+α)]
    [0  0   1                     ]
```

**For Straight Motion** (r ≈ l):
```python
x_new = x + l·cos(θ)
y_new = y + l·sin(θ)
θ_new = θ

G = [1  0  -l·sin(θ)]
    [0  1   l·cos(θ)]
    [0  0   1       ]
```

**Covariance Update**:
```python
Σ_new = G·Σ·Gᵀ + V·Q·Vᵀ
```
Where Q is the process noise from motor uncertainties.

#### Correction Step (Measurement Model)

When a landmark is detected:

**Measurement**: (r_measured, α_measured)

**Expected measurement** from current state:
```python
dx = landmark_x - robot_x
dy = landmark_y - robot_y
r_expected = √(dx² + dy²)
α_expected = atan2(dy, dx) - robot_θ
```

**Measurement Jacobian**:
```python
H = [-dx/r   -dy/r    0      dx/r    dy/r  ]
    [dy/r²  -dx/r²   -1    -dy/r²   dx/r² ]
```

**Kalman Update**:
```python
# Innovation
z = [r_measured - r_expected, α_measured - α_expected]

# Kalman Gain
K = Σ·Hᵀ·(H·Σ·Hᵀ + R)⁻¹

# State update
μ = μ + K·z

# Covariance update  
Σ = (I - K·H)·Σ
```

#### Landmark Management
- **Addition**: New landmarks initialized with large uncertainty (σ² = 0.5m²)
- **Minimum sightings**: Landmarks added only after 1+ confirmatory detections
- **Association**: Landmarks matched by ArUco ID (data association solved)

### 4. Grid-Based Mapping

The system maintains an occupancy grid map for path planning:

**Grid Specification:**
- Resolution: 1 cm per cell (0.01 m)
- Size: 4m × 4m (400×400 cells)
- Center: (0, 0) in world coordinates
- Value range: 0.0 (free) to 1.0 (occupied)

#### Grid Population

**Coordinate Conversion**:
```python
def world_to_grid(x, y):
    grid_x = int((x + width/2) / resolution)
    grid_y = int((y + height/2) / resolution)
    return (grid_x, grid_y)
```

**Obstacle Representation**:
```python
# Boundary/obstacle markers
radius_cells = int(0.3m / 0.01m) = 30 cells
grid[marker_position] = 0.7  # High cost region

# Blocks (after detection)
radius_cells = int(0.25m / 0.01m) = 25 cells  
grid[block_position] = 0.7

# Safety margins expand occupied regions
```

The grid map is continuously updated as new landmarks are detected during exploration.

### 5. Path Planning (A* Algorithm)

The robot uses A* with a safety-aware cost function for path planning.

#### Algorithm Details

**Cost Function**:
```python
f(n) = g(n) + h(n) + safety_cost(n)

where:
- g(n): Actual cost from start to node n
- h(n): Heuristic (Manhattan distance to goal)
- safety_cost(n): Penalty based on proximity to obstacles
```

**Safety Cost Calculation**:
```python
safety_value = grid[node]  # 0.0 to 1.0
safety_penalty = 5.0 × safety_value
```

This encourages paths through open space while still allowing passage through narrow gaps when necessary.

**8-Directional Movement**:
- Cardinal directions (↑↓←→): cost = 1.0
- Diagonal directions (↗↘↖↙): cost = √2 ≈ 1.414

**Path Simplification**:
After finding the optimal path, waypoints are extracted at direction changes:
```python
# Remove collinear points
simplified_path = [start]
for i in range(1, len(path)-1):
    if direction_changes(path[i-1], path[i], path[i+1]):
        simplified_path.append(path[i])
simplified_path.append(goal)
```

### 6. Path Following & Motion Control

The robot uses a PID controller for smooth path following.

#### Pure Pursuit-Style Control

**Target Selection**:
```python
# Follow waypoints in sequence
current_waypoint = path[0]
distance_to_waypoint = ||robot_pos - waypoint||
```

**Angle Error Calculation**:
```python
# Desired direction to waypoint
desired_θ = atan2(waypoint_y - robot_y, waypoint_x - robot_x)

# Error (wrapped to [-π, π])
error = (desired_θ - robot_θ + π) mod 2π - π
```

**PID Control**:
```python
turn = Kp·error + Ki·∫error·dt + Kd·(derror/dt)

# Parameters
Kp = 0.5   # Proportional gain
Ki = 0.01  # Integral gain
Kd = 0.3   # Derivative gain
```

**Speed Control**:
```python
# Distance-based speed modulation
distance = ||robot_pos - waypoint||
speed = 5 + 10·min(distance, 1.0)

# Slow down for sharp turns
if |turn| > 60°:
    speed = 0  # Stop and turn
```

**Waypoint Removal**:
```python
if distance_to_waypoint < 5cm:
    path.pop(0)  # Move to next waypoint
```

#### Gate Alignment

When delivering blocks to gates, the robot aligns perpendicular to the gate:
```python
# Calculate gate orientation from marker pair
gate_direction = (right_marker - left_marker) / ||...||
perpendicular = [-gate_direction.y, gate_direction.x]

# Compute drop location offset from gate center
drop_location = gate_center + offset·perpendicular

# Align robot perpendicular to gate before releasing
```

### 8. Block Manipulation

#### Pickup Sequence

```python
1. Detect block (via marker pair)
2. Plan path to approach position (12cm from block)
3. Open gripper
4. Follow path with reduced speed near target
5. Verify arrival (distance < 12cm)
6. Close gripper (wait 1s for mechanical operation)
7. Switch to delivery mode
```

#### Delivery Sequence

```python
1. Detect matching colored gate
2. Compute drop location perpendicular to gate
3. Plan path to drop location
4. Follow path
5. Align robot perpendicular to gate orientation
6. Drive forward (15 speed, 4s)
7. Open gripper
8. Drive backward (-15 speed, 4s)
9. Mark block as delivered
10. Switch to exploration mode (search for next block)
```

#### Gripper Control

The gripper uses a single motor (Port B):
```python
# Open: rotate -60° at speed 5
# Close: rotate +60° at speed 5
```

State tracking prevents redundant operations.

---

## Installation & Setup

### Prerequisites

#### Hardware
- LEGO Mindstorms EV3 with ev3dev OS installed
- USB webcam (calibrated)
- Computer for viewer (optional)

#### Software Dependencies

**On EV3 (Python 3.9+):**
```bash
pip install opencv-contrib-python==4.5.5.64
pip install numpy scipy
pip install ev3-dc
pip install imagezmq jsonpickle
pip install rich PyYAML
```

**On PC (for viewer):**
```bash
pip install opencv-contrib-python
pip install numpy imagezmq jsonpickle
pip install PyQt5 imutils Pillow
```

### Operating Modes

The robot supports multiple modes, switchable via keyboard:

| Key | Mode | Description |
|-----|------|-------------|
| `m` | **Manual** | Keyboard control (WASD keys) |
| `e` | **Exploration** | Autonomous environment mapping |
| `t` | **Thinking** | Path planning mode |
| `p` | **PathFollower** | Follow computed path |
| `l` | **Load** | Load saved map |
| `o` | **Open Gripper** | Manual gripper control |
| `c` | **Close Gripper** | Manual gripper control |
| `q` | **Quit** | Stop program |


## License

This project is for educational purposes.
---

## Acknowledgments

Developed as part of a robotics course project offered from department of Computational Neuroscience @ Uni Gö
---

