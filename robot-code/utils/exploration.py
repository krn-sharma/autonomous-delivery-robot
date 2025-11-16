import numpy as np
from rich import print
from .pid_controller import PIDController
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import time
import matplotlib.pyplot as plt
import os
from .grid_map import GridMap
import heapq
from .follow_path import PathFollower
from .a_star import a_star, simplify_path

def line_segments_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

class Exploration:
    def __init__(self, config):
        self.safety_radius = 0.3
        self.map_padding = 1.0
        
        self.min_blocks_before_return = 6
        self.block_size = 0.3
        self.block_safety_margin = 0.2
        self.safety_margin = 0.20
        
        self.border_markers = {}
        self.obstacle_markers = {}
        self.max_marker_gap = config.exploration.max_marker_gap
        self.min_border_markers = config.exploration.min_border_markers
        self.border_complete = False
        self.hull_points = None
        self.gaps = []
        self.current_gap = None
        self.last_time = time.time()
        self.last_robot_pos = None
        self.last_robot_pos_time = time.time()
        
        self.current_target = None
        
        self.min_markers_for_movement = 3
        self.last_valid_markers = 0
        self.required_valid_readings = 3
        self.last_movement_time = time.time()
        self.emergency_timeout = 6.0
        self.in_emergency_turn = False
        self.emergency_turn_start = 0
        self.movement_cooldown = 0.5
        
        self.initial_search_time = 0
        
        self.start_gate_seen = False
        self.start_gate_color = None
        
        # Add new triangle exploration parameters
        self.triangle_points = None
        self.current_target_idx = 0
        self.triangle_size = 3.0  # Size of the triangle in meters
        self.target_reached_threshold = 0.3  # Distance threshold to consider target reached
        
        # Add mode tracking
        self.mode = 'thinking'  # 'thinking' or 'following'
        self.current_path = None
        self.last_marker_count = 0
        self.path_recalc_cooldown = 0.5  # Minimum time between path recalculations
        self.last_path_calc = 0
        
        # Add pathfollower
        self.path_follower = None
        self.config = config  # Store config for pathfollower initialization
        
        self.triangle_points = [
            np.array([1.0, 1.0]),    # Top right
            np.array([-1.0, 1.0]),   # Top left
            np.array([0.0, 0.0])     # Bottom center (start)
        ]

        self.last_block_count = 0  # Add this line
        
        # Initialize grid map with values from main
        self.grid_map = GridMap(width=config.main.grid_width, 
                              height=config.main.grid_height, 
                              resolution=config.main.grid_resolution)
        
        self.checked_markers = set()  # Set für bereits besuchte Marker
        
        # Use block and gate tracking from main instance
        self.main = config.main_instance
        
        # Initialize exploration-specific variables
        self.border_markers = {}
        self.obstacle_markers = {}
        self.max_marker_gap = config.exploration.max_marker_gap
        self.min_border_markers = config.exploration.min_border_markers
        self.blocks = config.exploration.blocks
        self.border_complete = False
        self.hull_points = None
        self.gaps = []
        self.current_gap = None
        
        self.red_blocks = {}
        self.blue_blocks = {}
        
        self.potential_red_blocks = {}   # For single red block markers
        self.potential_blue_blocks = {}  # For single blue block markers
        
        self.red_gate = {}   # Marker 401-420
        self.blue_gate = {}  # Marker 501-520
        self.gate_positions = {
            'red': None,     # red gates
            'blue': None     # blue gates
        }

        self.min_gap_width = 0.10  # 10cm minimum gap width to detect

    def detect_blocks(self, data):
        """Detect blocks based on marker pairs and track potential blocks from single markers."""
        if data.landmark_estimated_ids is None or data.landmark_estimated_positions is None:
            print("[red]No estimated landmark data available")
            return

        marker_positions = {}
        for i, marker_id in enumerate(data.landmark_estimated_ids):
            if i < len(data.landmark_estimated_positions):
                marker_pos = data.landmark_estimated_positions[i]
                marker_positions[marker_id] = np.array(marker_pos)

        # Track potential red blocks (421-450)
        for marker_id in marker_positions:
            if 421 <= marker_id <= 450:
                base_id = marker_id if marker_id % 2 == 1 else marker_id - 1
                
                if base_id in marker_positions and base_id + 1 in marker_positions:
                    pos1 = marker_positions[base_id]
                    pos2 = marker_positions[base_id + 1]
                    
                    direction = pos2 - pos1
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        block_pos = pos2 + direction * 0.1
                        
                        if base_id not in self.red_blocks:
                            print(f"[red]Detected red block at {block_pos} (markers {base_id}, {base_id+1})")
                        self.potential_red_blocks.pop(base_id, None)
                        self.potential_red_blocks.pop(base_id + 1, None)
                        self.red_blocks[base_id] = {
                            'marker1': {"position": pos1, "id": base_id},
                            'marker2': {"position": pos2, "id": base_id+1},
                            'position': block_pos,
                            'last_seen': self.last_time
                        }
                        
                        if self.current_target is not None and self.current_target.get('type') == 'potential_block' and (self.current_target['id'] == base_id or self.current_target['id'] == base_id + 1):
                            self.current_target = None
                            self.mode = 'thinking'
                            self.path_follower = None
                
                else:
                    marker_pos = marker_positions[marker_id]
                    self.potential_red_blocks[marker_id] = {
                        'position': marker_pos,
                        'id': marker_id,
                        'last_seen': self.last_time
                    }
                    print(f"[yellow]Potential red block marker detected: {marker_id}")

        # Track potential blue blocks (521-550)
        for marker_id in marker_positions:
            if 521 <= marker_id <= 550:
                base_id = marker_id if marker_id % 2 == 1 else marker_id - 1
                
                if base_id in marker_positions and base_id + 1 in marker_positions:
                    pos1 = marker_positions[base_id]
                    pos2 = marker_positions[base_id + 1]
                    
                    direction = pos2 - pos1
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        block_pos = pos2 + direction * 0.1
                        
                        if base_id not in self.blue_blocks:
                            print(f"[blue]Detected blue block at {block_pos} (markers {base_id}, {base_id+1})")
                        self.potential_blue_blocks.pop(base_id, None)
                        self.potential_blue_blocks.pop(base_id + 1, None)
                        self.blue_blocks[base_id] = {
                            'marker1': {"position": pos1, "id": base_id},
                            'marker2': {"position": pos2, "id": base_id+1},
                            'position': block_pos,
                            'last_seen': self.last_time
                        }

                        if self.current_target is not None and self.current_target.get('type') == 'potential_block' and (self.current_target['id'] == base_id or self.current_target['id'] == base_id + 1):
                            self.current_target = None
                            self.mode = 'thinking'
                            self.path_follower = None
                
                else:
                    marker_pos = marker_positions[marker_id]
                    self.potential_blue_blocks[marker_id] = {
                        'position': marker_pos,
                        'id': marker_id,
                        'last_seen': self.last_time
                    }
                    print(f"[yellow]Potential blue block marker detected: {marker_id}")

        current_time = self.last_time
        timeout = 2.0
        
        for blocks in [self.potential_red_blocks, self.potential_blue_blocks]:
            to_remove = []
            for marker_id, block_data in blocks.items():
                if current_time - block_data['last_seen'] > timeout:
                    to_remove.append(marker_id)
            for marker_id in to_remove:
                blocks.pop(marker_id)

    def get_block_position(self, block_id):
        
        if block_id in self.red_blocks:
            block = self.red_blocks[block_id]
        elif block_id in self.blue_blocks:
            block = self.blue_blocks[block_id]
        elif block_id - 1 in self.red_blocks:
            block = self.red_blocks[block_id - 1]
        elif block_id - 1 in self.blue_blocks:
            block = self.blue_blocks[block_id - 1]
        else:
            return None
        
        if block is None:
            block = self.blue_blocks[str(block_id)]

        return block['position']

    def detect_gate(self, data):
        """Detect and update gate positions based on SLAM estimated positions"""
        if data.landmark_estimated_ids is None or data.landmark_estimated_positions is None:
            return None
        
        self.red_gate.clear()
        self.blue_gate.clear()
        
        red_markers = []
        blue_markers = []
        
        for i, marker_id in enumerate(data.landmark_estimated_ids):
            if i >= len(data.landmark_estimated_positions):
                continue
            
            marker_pos = np.array(data.landmark_estimated_positions[i])
            
            if 401 <= marker_id <= 420:
                red_markers.append({
                    'id': marker_id,
                    'position': marker_pos,
                    'last_seen': self.last_time
                })
                
            elif 501 <= marker_id <= 520:
                blue_markers.append({
                    'id': marker_id,
                    'position': marker_pos,
                    'last_seen': self.last_time
                })

        if len(red_markers) > 0:
            self.gate_positions['red'] = red_markers
        if len(blue_markers) > 0:
            self.gate_positions['blue'] = blue_markers
        
    def find_border_gaps(self):
        """Find gaps in the border line that are wider than min_gap_width."""
        if len(self.border_markers) < 3:
            return None
            
        positions = np.array([marker['position'] for marker in self.border_markers.values()])
        
        try:
            hull = ConvexHull(positions)
            hull_points = positions[hull.vertices]
            self.hull_points = hull_points
            
            gaps = []
            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % len(hull_points)]
                
                gap_width = np.linalg.norm(p2 - p1)
                
                if gap_width > self.min_gap_width:
                    gap_center = (p1 + p2) / 2
                    gaps.append({
                        'width': gap_width,
                        'center': gap_center,
                        'point1': p1,
                        'point2': p2,
                        'id': i
                    })
            
            return gaps if gaps else None
            
        except Exception as e:
            print(f"[red]Error calculating convex hull: {e}")
            return None

    def calculate_map_coverage(self, data):
        """Calculate map coverage based on landmark positions."""
        if data.landmark_estimated_ids is None or data.landmark_estimated_positions is None:
            return
        
        self.border_markers.clear()
        self.obstacle_markers = {}
        
        for i, marker_id in enumerate(data.landmark_estimated_ids):
            if i >= len(data.landmark_estimated_positions):
                continue
            
            marker_pos = np.array(data.landmark_estimated_positions[i])
            
            # Border markers (IDs 1-300)
            if 1 <= marker_id <= 300:
                self.border_markers[marker_id] = {
                    'position': marker_pos,
                    'last_seen': self.last_time
                }
            
            # Obstacle markers (IDs 301-400)
            elif 301 <= marker_id <= 400:
                self.obstacle_markers[marker_id] = {
                    'position': marker_pos,
                    'last_seen': self.last_time
                }
        
        gaps = self.find_border_gaps()
        if gaps:
            self.gaps = gaps
            if self.current_gap is None and gaps:
                self.current_gap = gaps[0]
        else:
            self.current_gap = None

    def get_furthest_marker(self, robot_pos):
        """Find the furthest border or obstacle marker from current position."""
        max_dist = 0
        furthest_marker = None
        
        all_markers = []
        for marker_id, marker_data in self.border_markers.items():
            all_markers.append(marker_data['position'])
        for marker_id, marker_data in self.obstacle_markers.items():
            all_markers.append(marker_data['position'])
            
        if len(all_markers) < 2:  # Need at least 2 markers to find nearest neighbor
            return None
            
        all_markers = np.array(all_markers)
        
        furthest_marker = None
        max_dist = 0.0

        for marker_id, marker_data in self.obstacle_markers.items():
            marker_pos = marker_data['position']
            
            distances = np.linalg.norm(all_markers - marker_pos, axis=1)
            if len(distances) > 1:  # Check if we have enough distances
                nearest_neighbor_dist = np.partition(distances, 1)[1]
                
                if nearest_neighbor_dist > 2.0:
                    continue
                    
                dist = np.linalg.norm(marker_pos - robot_pos)
                
                if dist < 0.3:  # Falls die Entfernung weniger als 30 cm beträgt
                    furthest_marker = None
                elif dist > max_dist:
                    max_dist = dist
                    furthest_marker = {
                        'position': marker_pos,
                        'id': marker_id,
                        'type': 'obstacle'
                    }
                    
        if furthest_marker:
            furthest_pos = furthest_marker['position']
            for marker_id, marker_data in self.obstacle_markers.items():
                marker_pos = marker_data['position']
                if np.linalg.norm(marker_pos - furthest_pos) < 0.3: # 30cm
                    self.checked_markers.add(marker_id)
        
        if furthest_marker is None:
            for marker_id, marker_data in self.border_markers.items():
                marker_pos = marker_data['position']
                
                distances = np.linalg.norm(all_markers - marker_pos, axis=1)
                if len(distances) > 1:
                    nearest_neighbor_dist = np.partition(distances, 1)[1]
                    
                    if nearest_neighbor_dist > 2.0:
                        continue
                        
                    dist = np.linalg.norm(marker_pos - robot_pos)
                    if dist > max_dist:
                        max_dist = dist
                        furthest_marker = {
                            'position': marker_pos,
                            'id': marker_id,
                            'type': 'border'
                        }
        
        return furthest_marker

    def check_blocks_near_path(self, current_path, blocks):
        """Check if any blocks are near the current path"""
        if not current_path:
            return False
            
        world_path = []
        for point in current_path:
            x, y = self.grid_map.grid_to_world(point[0], point[1])
            world_path.append((x, y))
            
        for block_data in blocks.values():
            block_pos = np.array(block_data['position'])
            
            for i in range(len(world_path) - 1):
                p1 = np.array(world_path[i])
                p2 = np.array(world_path[i + 1])
                
                segment = p2 - p1
                if np.all(segment == 0):
                    continue
                    
                t = max(0, min(1, np.dot(block_pos - p1, segment) / np.dot(segment, segment)))
                projection = p1 + t * segment
                distance = np.linalg.norm(block_pos - projection)
                
                if distance < 0.5:
                    return True
                    
        return False

    def tramp(self, data):
        """Main exploration function using furthest marker targeting."""
        current_time = time.time()
        self.last_time = current_time
        
        if data.robot_position is None:
            return 0, 0, False
        
        robot_pos = np.array(data.robot_position)
        
        if self.last_robot_pos is None or np.linalg.norm(robot_pos - self.last_robot_pos) > 0.1:
            self.last_robot_pos = robot_pos
            self.last_time = current_time
            
        if current_time - self.last_robot_pos_time > 30:
            self.mode = 'thinking'
            self.path_follower = None
            self.last_robot_pos_time = current_time
        
        self.detect_blocks(data)
        self.detect_gate(data)
        self.calculate_map_coverage(data)
        
        if self.current_target is None:
            if len(self.main.delivered_blocks) >= 2 and len(self.main.delivered_blocks) == (len(self.red_blocks) + len(self.blue_blocks)) and self.current_gap is None or len(self.main.delivered_blocks) >= self.blocks:
                return 0, 0, True
            elif len(self.main.delivered_blocks) >= 2 and len(self.main.delivered_blocks) == (len(self.red_blocks) + len(self.blue_blocks)) and self.current_gap is not None:
                self.current_target = {
                    'position': self.current_gap['center'],
                    'type': 'gap',
                    'id': self.current_gap['id']
                }
                print(f"[green]New target: {self.current_target['type']} gap {self.current_target['id']}")

            else:
                furthest_marker = self.get_furthest_marker(robot_pos)
                if furthest_marker is not None:
                    self.current_target = furthest_marker
                    print(f"[green]New target: {furthest_marker['type']} marker {furthest_marker['id']}")
                else:
                    return 0, 10, False
        
        # Check if we've reached the current target
        dist_to_target = np.linalg.norm(robot_pos - self.current_target['position'])
        if dist_to_target < self.target_reached_threshold:
            print(f"[green]Reached target point {self.current_target['id']}, moving to next point")
            self.current_target = None
            self.mode = 'thinking' 
            self.path_follower = None
        
        # Check if we need to switch to thinking mode
        marker_count = (len(self.border_markers) + len(self.obstacle_markers))
        time_since_last_calc = time.time() - self.last_path_calc
        
        new_markers = []
        if marker_count != self.last_marker_count:
            for marker_dict in [self.border_markers, self.obstacle_markers]:
                for marker_id, marker_data in marker_dict.items():
                    if marker_id >= self.last_marker_count:
                        new_markers.append(marker_data)
        
        red_block_count = len(self.red_blocks)
        blue_block_count = len(self.blue_blocks)
        total_blocks = red_block_count + blue_block_count
        
        if self.potential_red_blocks or self.potential_blue_blocks:
            print(f"[yellow]Potential blocks: {len(self.potential_red_blocks)} red, {len(self.potential_blue_blocks)} blue")
            
            if self.current_target is None or self.current_target.get('type') != 'potential_block':
                furthest_dist = 0
                furthest_block = None
                
                for block_id, block_data in self.potential_red_blocks.items():
                    dist = np.linalg.norm(robot_pos - block_data['position'])
                    if dist > furthest_dist:
                        furthest_dist = dist
                        furthest_block = {
                            'position': block_data['position'],
                            'id': block_id,
                            'type': 'potential_block',
                            'color': 'red'
                        }
                
                for block_id, block_data in self.potential_blue_blocks.items():
                    dist = np.linalg.norm(robot_pos - block_data['position'])
                    if dist > furthest_dist:
                        furthest_dist = dist
                        furthest_block = {
                            'position': block_data['position'],
                            'id': block_id,
                            'type': 'potential_block',
                            'color': 'blue'
                        }
                
                if furthest_block:
                    self.current_target = furthest_block
                    print(f"[yellow]Investigating furthest potential {furthest_block['color']} block {furthest_block['id']} at distance {furthest_dist:.2f}m")
                    self.mode = 'thinking'
        
        recalculate = False
        if time_since_last_calc > self.path_recalc_cooldown:
            # Check if new markers are near path
            if new_markers and self.check_markers_near_path(self.current_path, new_markers):
                print(f"[yellow]New markers near path detected, recalculating...")
                recalculate = True
            # Check if new blocks are near path
            elif total_blocks > self.last_block_count:
                print(f"[yellow]New blocks near path detected, recalculating...")
                recalculate = True
        
        if recalculate:
            self.mode = 'thinking'
            self.path_follower = None
        
        self.last_marker_count = marker_count
        self.last_block_count = total_blocks
        
        if self.mode == 'thinking':
            self.grid_map.update_from_exploration(self, self.main.delivered_blocks)
            
            if self.current_target is None:
                return 0, 0, False
            
            routes = a_star(self.grid_map, self.current_target['position'], data)
            new_path = simplify_path(routes)
            self.grid_map.save_map("maps/map.png", self, data.robot_position, new_path)
            
            if not new_path or len(new_path) < 2:
                print("[red]No valid path found in thinking mode, rotating to search")
                return 0, 10, False
            
            world_path = [self.grid_map.grid_to_world(pos[0], pos[1]) for pos in new_path]
            print(world_path)
            
            self.path_follower = PathFollower(world_path, self.config)
            
            self.mode = 'following'
            self.last_path_calc = current_time

            print("[green]Switched to following mode with new path")
        
        if self.mode == 'following' and self.path_follower is not None:
            speed, turn, arrived = self.path_follower.follow_path(data, 0.15)
            
            print(f"[green]Following path to {self.current_target['id']}")
            
            # Reduce speed when approaching potential blocks
            if self.current_target.get('type') == 'potential_block':
                dist_to_target = np.linalg.norm(robot_pos - self.current_target['position'])
                if dist_to_target < 1.0:
                    #speed = speed * 0.5 
                    print(f"[yellow]Approaching potential block carefully at {speed:.2f} speed")
            
            if arrived:
                print("[green]Destination Arrived")
                self.mode = 'thinking'
                return 0, 0, False
            
            self.last_time = current_time
            return speed, turn, False
        
        return 0, 0, False

    def check_markers_near_path(self, current_path, marker_positions, threshold=1.0):
        """Check if any markers are near the current path"""
        if not current_path:
            return False
            
        world_path = []
        for point in current_path:
            x, y = self.grid_map.grid_to_world(point[0], point[1])
            world_path.append((x, y))
            
        for pos in marker_positions:
            marker_pos = np.array(pos['position'])
            
            for i in range(len(world_path) - 1):
                p1 = np.array(world_path[i])
                p2 = np.array(world_path[i + 1])
                
                segment = p2 - p1
                if np.all(segment == 0):
                    continue
                    
                t = max(0, min(1, np.dot(marker_pos - p1, segment) / np.dot(segment, segment)))
                projection = p1 + t * segment
                distance = np.linalg.norm(marker_pos - projection)
                
                if distance < threshold:
                    return True
                    
        return False