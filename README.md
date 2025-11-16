# Autonomous Robot Navigation System with EKF-SLAM

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Hardware Components](#hardware-components)
4. [Software Pipeline](#software-pipeline)
5. [Installation & Setup](#installation--setup)
6. [Usage](#usage)
7. [Technical Details](#technical-details)
8. [Future Improvements](#future-improvements)

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

### 7. Exploration Strategy

The robot uses a frontier-based exploration approach:

#### Exploration State Machine

```
Initial Rotation (30s)
    ↓
Identify Furthest Unvisited Marker
    ↓
Plan Path (A*)
    ↓
Follow Path
    ↓
[New markers detected?] → Replan if obstacles detected
    ↓
[Block detected?] → Switch to Block Delivery Mode
    ↓
[Sufficient exploration?] → Manual Mode
```

#### Marker Selection Heuristic

```python
def get_furthest_marker(robot_pos):
    max_distance = 0
    target = None
    
    for marker in [obstacles, boundaries]:
        # Ignore isolated markers (spacing > 2m)
        if nearest_neighbor_distance(marker) > 2.0:
            continue
        
        # Ignore very close markers (< 30cm)
        distance = ||marker - robot_pos||
        if distance < 0.3:
            continue
            
        if distance > max_distance:
            max_distance = distance
            target = marker
    
    return target
```

#### Block Detection

Blocks are identified by ArUco marker pairs:
```python
# Red blocks: markers 421-450 (odd-even pairs)
# Blue blocks: markers 521-550 (odd-even pairs)

if marker_n AND marker_(n+1) both visible:
    block_position = marker_(n+1) + 0.1m·direction
    register_block(block_position, color)
else:
    track_as_potential_block(marker_n)
```

#### Path Recalculation

The robot recalculates paths when:
1. New obstacles detected near current path (< 1m)
2. New blocks detected near path (< 0.5m)
3. Robot stuck (no progress for > 30s)

Cooldown period: 0.5s between recalculations to avoid thrashing.

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

### Configuration

#### 1. Camera Calibration

If using a different camera, recalibrate:
```bash
# Use OpenCV calibration tool with checkerboard pattern
# Update camera_intrinsics.yml with new parameters
```

#### 2. Robot Configuration

Edit `robot-code/config.yaml`:

```yaml
robot:
  wheel_radius: 0.027216  # Measure your wheels (meters)
  width: 0.11             # Wheel-to-wheel distance (meters)
  delta_t: 0.1            # Control loop period (seconds)

camera:
  y_offset: 0             # Camera offset from robot center
  z_offset: 0             # Camera height
  y_angle: 0              # Camera tilt angle (radians)
  exposure_time: 60       # 1-5000
  gain: 100               # 0-100

ekf_slam:
  motor_std: 4            # Motor noise (degrees)
  dist_std: 0.1           # Distance measurement noise (meters)
  angle_std: 4            # Angle measurement noise (degrees)

main:
  grid_resolution: 0.01   # Grid cell size (meters)
  grid_width: 4.0         # Map width (meters)
  grid_height: 4.0        # Map height (meters)

exploration:
  speed_lost: 10          # Speed when lost
  speed_cruise: 15        # Normal exploration speed
  amount_of_blocks: 4     # Number of blocks to deliver
```

#### 3. Network Setup (for viewer)

**On EV3**, find IP address:
```bash
hostname -I
```

**On PC**, edit `subscriber.py`:
```python
self.image_hub = imagezmq.ImageHub(
    open_port='tcp://YOUR_EV3_IP:5555', 
    REQ_REP=False
)
```

---

## Usage

### Running the Robot

#### 1. Start the Viewer (on PC)
```bash
cd /path/to/robot
python viewer.py
```

#### 2. Deploy Code to EV3
```bash
scp -r robot-code/ robot@YOUR_EV3_IP:~/
```

#### 3. Run on EV3
```bash
ssh robot@YOUR_EV3_IP
cd robot-code/
python main.py
```

### Operating Modes

The robot supports multiple modes, switchable via keyboard:

| Key | Mode | Description |
|-----|------|-------------|
| `m` | **Manual** | Keyboard control (WASD keys) |
| `e` | **Exploration** | Autonomous environment mapping |
| `t` | **Thinking** | Path planning mode |
| `p` | **PathFollower** | Follow computed path |
| `r` | **Race** | High-speed mode (reduced SLAM updates) |
| `l` | **Load** | Load saved map |
| `o` | **Open Gripper** | Manual gripper control |
| `c` | **Close Gripper** | Manual gripper control |
| `q` | **Quit** | Stop program |

#### Manual Control (Mode: `m`)
- `w`: Forward
- `s`: Backward  
- `a`: Turn left
- `d`: Turn right
- `c`: Stop

#### Autonomous Operation

**Typical sequence:**
1. Press `e` to start exploration
2. Robot rotates for 30s to scan environment
3. Robot explores, building map and detecting blocks/gates
4. When block + matching gate found → automatic delivery
5. Returns to exploration after delivery
6. Continues until all blocks delivered or exploration complete

### Monitoring & Debugging

#### Viewer Display

The viewer shows:
- Live camera feed with detected markers
- Robot position and heading (green triangle)
- Landmark positions with uncertainty ellipses
- Detected vs. estimated landmark positions
- Grid map overlay
- Current path (if active)

#### Console Output

The robot prints status information:
```
[green]Found red gate and blocks - Switching to Thinking mode
[green]Thinking path to red Block with ID-422
[green]Path Found
[green]Following path to 422
[green]Destination Arrived
```

#### Performance Metrics

Monitor timing in console:
```
cam fps: 28
[red]Warning! dt = 0.15  # Control loop exceeded 0.1s
[red]EKF_SLAM correction time: 0.12  # SLAM update took too long
```

### Saving Maps

Maps are automatically saved during exploration:
```
robot-code/maps/map.png
```

The map visualization includes:
- Boundary markers (green)
- Obstacle markers (orange)
- Red/blue gates (red/blue stars)
- Detected blocks (red/blue circles)
- Block markers with IDs
- Robot position (green triangle)
- Current path (green dashed line)

---

## Technical Details

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Camera Frame Rate | 30 FPS |
| Control Loop Frequency | 10 Hz (0.1s period) |
| ArUco Detection Latency | ~20-30ms |
| EKF Prediction Time | <10ms (typical) |
| EKF Correction Time | ~10-20ms per landmark |
| A* Planning Time | ~100-500ms (depends on map size) |
| Localization Accuracy | ±5cm (under good conditions) |
| Heading Accuracy | ±3° (under good conditions) |

### Coordinate Systems

**World Frame** (Right-handed):
- Origin: Center of environment
- +X: Right
- +Y: Forward  
- +θ: Counter-clockwise from +X axis

**Robot Frame**:
- Origin: Between wheels
- +X: Forward
- +Y: Left
- +θ: Counter-clockwise

**Camera Frame**:
- Origin: Camera optical center
- +Z: Looking direction
- +X: Right in image
- +Y: Down in image

### Error Sources & Mitigation

**Wheel Slippage**:
- Impact: Odometry drift
- Mitigation: EKF fuses odometry with visual landmarks

**Marker Detection Failures**:
- Impact: Missing measurements
- Mitigation: Minimum sighting threshold, tracking of potential blocks

**Lighting Variations**:
- Impact: Detection reliability
- Mitigation: Configurable exposure/gain, histogram equalization

**Map Ambiguity**:
- Impact: Path planning through unknown areas
- Mitigation: Safety margins, conservative cost function

### Computational Complexity

**EKF-SLAM**:
- State dimension: 3 + 2n (n landmarks)
- Prediction: O(n²) - full covariance matrix
- Correction: O(n) per landmark
- Memory: O(n²) for covariance matrix

**A\* Path Planning**:
- Worst case: O(b^d) where b=branching factor (8), d=depth
- Typical: O(n log n) with good heuristic
- Memory: O(n) for open set

**Exploration**:
- Marker selection: O(m) where m = number of markers
- Path recalculation: Triggered by events, not continuous

### Failure Modes & Recovery

**Robot Lost** (no landmarks visible):
- Action: Rotate in place to scan for markers
- Timeout: 30s, then switch to manual mode

**Path Planning Failure** (no path found):
- Action: Rotate 10° and retry
- Attempts: 36 (full rotation)
- Fallback: Manual mode

**Block Lost During Approach**:
- Action: Mark as potential block, continue exploration
- Retry: If both markers redetected

**Gripper Failure**:
- Detection: Block markers still visible after pickup
- Action: Continue (no automatic retry)

---

## Future Improvements

### Algorithmic Enhancements

1. **FastSLAM 2.0**: Particle filter approach for improved scalability
2. **Loop Closure Detection**: Recognize previously visited areas to correct drift
3. **Multi-hypothesis Tracking**: Handle ambiguous data associations
4. **Dynamic Replanning**: Real-time path adjustment during execution
5. **Semantic Mapping**: Classify regions (boundaries, gates, block zones)

### Perception Upgrades

1. **Deep Learning**: CNN-based block detection without markers
2. **Sensor Fusion**: IMU integration for improved odometry
3. **3D Mapping**: Depth camera for obstacle height estimation
4. **Lighting Robustness**: Automatic exposure control based on histogram
5. **Multi-Camera**: 360° vision coverage

### Control Improvements

1. **Model Predictive Control (MPC)**: Optimal trajectory tracking
2. **Adaptive PID**: Auto-tune parameters based on performance
3. **Differential Flatness**: Smooth trajectory generation
4. **Torque Control**: Direct motor current control for precision

### System Architecture

1. **ROS2 Integration**: Standard robotics middleware
2. **Behavior Trees**: Hierarchical task decomposition
3. **Distributed Computing**: Offload SLAM to more powerful PC
4. **State Persistence**: Save/resume missions across power cycles
5. **Multi-Robot Collaboration**: Cooperative exploration and delivery

### Software Engineering

1. **Unit Tests**: Coverage for all algorithms (pytest framework)
2. **Simulation**: Gazebo/Webots integration for virtual testing
3. **CI/CD Pipeline**: Automated testing and deployment
4. **Documentation**: Sphinx-generated API docs
5. **Profiling**: Identify performance bottlenecks (cProfile)

---

## References

### Academic Papers
- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Durrant-Whyte, H., & Bailey, T. (2006). "Simultaneous Localization and Mapping: Part I." *IEEE Robotics & Automation Magazine*, 13(2), 99-110.
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.

### Libraries & Tools
- OpenCV: https://opencv.org/
- ArUco Markers: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- ev3dev: https://www.ev3dev.org/
- ZMQ: https://zeromq.org/

### Hardware
- LEGO Mindstorms EV3: https://www.lego.com/en-us/themes/mindstorms
- Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

---

## License

This project is for educational purposes. Feel free to use and modify with attribution.

---

## Acknowledgments

Developed as part of a robotics course project. Special thanks to the ev3dev community and OpenCV contributors.

---

## Contact

For questions or collaboration:
- GitHub: [Your GitHub Profile]
- Email: [Your Email]

---

*Last Updated: November 2024*

