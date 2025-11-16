import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
import math

class Thinking:
    def __init__(self):
        self.t = 1
        # Initialize the grid map with a 2x2 m area and 1 cm resolution
        self.grid_resolution = 0.01  # 1 cm resolution
        self.grid_size = int(4 / self.grid_resolution)  # 2 m / 0.01 m
        self.gridMap = np.zeros((self.grid_size, self.grid_size), dtype=np.int64)  # 0 indicates empty cells

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        grid_x = int((x + 3.0) / self.grid_resolution)
        grid_y = int((y + 2.0) / self.grid_resolution)
        return grid_x, grid_y

    def add_markers_to_grid(self, positions, marker_type):
        """Add markers to the grid map."""
        for pos in positions:
            grid_x, grid_y = self.world_to_grid(pos[0], pos[1])
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.gridMap[grid_y, grid_x] = marker_type

    def grid_to_world(self, grid_x, grid_y):
        return (grid_x * self.grid_resolution - 3.0, grid_y * self.grid_resolution - 2.0)

    def think(self, data):
        # Extract SLAM data
        landmark_ids = data.landmark_estimated_ids
        landmark_positions = data.landmark_estimated_positions
        robot_position = data.robot_position
        # Clear the grid map
        self.gridMap.fill(0)

        # Map marker IDs to their roles
        for idx, marker_id in enumerate(landmark_ids):
            position = landmark_positions[idx]
            if 1 <= marker_id <= 300:  # Outer boundary markers
                self.add_markers_to_grid([position], marker_type=2)  # 2 for outer boundary
                self.add_danger_zone(position, marker_type=2)
            elif 301 <= marker_id <= 400:  # Obstacle markers
                self.add_markers_to_grid([position], marker_type=3)  # 3 for obstacles
                self.add_danger_zone(position, marker_type=3)
            elif 401 <= marker_id <= 420:  # Red gate markers
                self.add_markers_to_grid([position], marker_type=4)  # 4 for red gate
            elif 501 <= marker_id <= 520:  # Blue gate markers
                self.add_markers_to_grid([position], marker_type=5)  # 5 for blue gate
            elif 421 <= marker_id <= 450:  # Red blocks
                self.add_markers_to_grid([position], marker_type=6)  # 6 for red block
            elif 521 <= marker_id <= 550:  # Blue blocks
                self.add_markers_to_grid([position], marker_type=7)  # 7 for blue block

        # Add the robot position to the grid map
        grid_x, grid_y = self.world_to_grid(robot_position[0], robot_position[1])
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            self.gridMap[grid_y, grid_x] = 1  # 1 for the robot

    def add_danger_zone(self, position, marker_type):
        """
        Fills a 15x15 square centered at the given position with the specified marker type.
        Assumes world_to_grid converts the world position to grid coordinates.
        """
        center_x, center_y = self.world_to_grid(position[0], position[1])
        half_zone = 7  # 15 cells total: center + 7 cells on each side

        for dx in range(-half_zone, half_zone + 1):
            for dy in range(-half_zone, half_zone + 1):
                grid_x = center_x + dx
                grid_y = center_y + dy
                # Check if the cell is within the grid boundaries
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    self.gridMap[grid_y, grid_x] = marker_type


    def a_star(self, goal, data):
        """A* search algorithm for pathfinding with robot clearance."""

        self.think(data)
        grid_x, grid_y = self.world_to_grid(data.robot_position[0], data.robot_position[1])
        start = (grid_x, grid_y)
        goal = self.world_to_grid(goal[0], goal[1])

        # Choose an appropriate 8-direction heuristic. Octile distance is often best.
        def heuristic(a, b):
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            D  = 1.0
            D2 = math.sqrt(2)
            # Octile distance:
            return D * (dx + dy) + (D2 - 2*D) * min(dx, dy)

        clearance = 3

        def is_cell_safe(cell):
            """Check if the cell and its surrounding area (robot footprint) are safe."""
            x, y = cell
            for dx in range(-clearance, clearance + 1):
                for dy in range(-clearance, clearance + 1):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                        return False  # Out of bounds
                    if self.gridMap[ny, nx] in [2, 3]:
                        return False  # Obstacle or boundary
            return True

        # 8-direction neighbors
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0),
                    (1, 1), (-1, 1), (1, -1), (-1, -1)]

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()  # from start to goal
                self.display_map(path=path)
                return path

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                    if not is_cell_safe(neighbor):
                        continue

                    # Cost = 1.0 for straight moves, sqrt(2) for diagonals
                    move_cost = math.sqrt(2) if (dx != 0 and dy != 0) else 1.0
                    tentative_g_score = g_score[current] + move_cost

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found


    def find_nearest_block(self, color, data, blocks):
        """Find the nearest block of the given color.

        Args:
            color (str or None): 'red' to find a red block, 'blue' to find a blue block, or None to find any block.
            data (SimpleNamespace): Contains robot state and landmark information.
            blocks (list): List of block IDs.

        Returns:
            tuple: (nearest_block_id, nearest_color) or (None, None) if no block is found.
        """

        min_dist = float('inf')
        nearest_block_id = None
        nearest_color = None

        # Get robot's current position in world coordinates
        robot_x, robot_y = data.robot_position

        # Create a mapping from estimated landmark IDs to positions
        landmark_id_to_position = {lid: pos for lid, pos in zip(data.landmark_estimated_ids, data.landmark_estimated_positions)}

        # Iterate over only the relevant blocks (using their estimated positions)
        #print(landmark_id_to_position,blocks)
        for block_id in blocks:
            if block_id in landmark_id_to_position:
                #print("here")
                block_x, block_y = landmark_id_to_position[block_id]  # Retrieve world coordinates
                dist = ((block_x - robot_x) ** 2 + (block_y - robot_y) ** 2) ** 0.5  # Euclidean distance

                if dist < min_dist:
                    min_dist = dist
                    nearest_block_id = block_id  # Store the ID instead of position

        # Determine color based on the first time search
        if color is None and nearest_block_id:
            nearest_color = 'red' if nearest_block_id in self.red_blocks else 'blue'
        else:
            nearest_color = color  # If color is already known, return the same

        print(f"Nearest {nearest_color} block ID: {nearest_block_id}")
        return nearest_block_id


    def display_map(self, filename=f"grid/grid_map.png", path=None):
        """
        Display and save the grid map with an intuitive, cleaner visualization.
        
        Parameters:
            filename (str): Name of the file to save the map image.
            path (list of tuple, optional): A list of (row, col) coordinates representing the robot's path.
        """


        # Define RGB color mapping for different elements
        cmap = {
            0: (255, 255, 255),  # White - Empty space
            1: (50, 205, 50),    # Lime - Robot position
            2: (0, 0, 0),        # Black - Outer boundary
            3: (169, 169, 169),  # Gray - Obstacles
            4: (139, 0, 0),      # Dark Red - Red gate
            5: (0, 0, 255),      # Blue - Blue gate
            6: (255, 99, 71),    # Tomato - Red block
            7: (0, 255, 255)     # Cyan - Blue block
        }

        # Create an RGB image from the grid map
        colored_grid = np.zeros((*self.gridMap.shape, 3), dtype=np.uint8)
        for value, color in cmap.items():
            mask = self.gridMap == value
            colored_grid[mask] = color

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(colored_grid, origin="lower", interpolation="nearest")

        # Plot the robot position (if exists)
        robot_positions = np.argwhere(self.gridMap == 1)
        if robot_positions.size > 0:
            # To avoid duplicate legend entries, plot only once (even if multiple positions)
            ax.scatter(robot_positions[:, 1], robot_positions[:, 0],
                    color="lime", edgecolors="black", s=200, marker="X", label="Robot")

        # Plot the path if provided
        if path is not None and len(path) > 0:
            # Expecting path as list of (row, col) coordinates
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], color="orange",
                    linestyle='-', linewidth=2, marker='o', markersize=8, label="Path")

        # Create legend elements for all markers
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                    markersize=10, label='Boundary'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                    markersize=10, label='Obstacle'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                    markersize=10, label='Red Gate'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                    markersize=10, label='Blue Gate'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='tomato',
                    markersize=10, label='Red Block'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan',
                    markersize=10, label='Blue Block'),
            plt.Line2D([0], [0], marker='X', color='lime', markeredgecolor='black',
                    markersize=12, linestyle='None', label='Robot')
        ]
        if path is not None and len(path) > 0:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='orange', linestyle='-',
                        markersize=8, label='Path')
            )

        ax.legend(handles=legend_elements, loc="upper right", frameon=True)

        # Aesthetics: title, labels, and removal of ticks for a cleaner look
        ax.set_title("Grid Map Visualization", fontsize=16, fontweight='bold')
        ax.set_xlabel("X (grid cells)", fontsize=12)
        ax.set_ylabel("Y (grid cells)", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grid map saved as {filename}")


    def compute_drop_location(self, gate_markers, robot_position, offset=0.1):
        """
        Computes the drop location offset meters perpendicularly away from the gate center,
        selecting the side that is farther from the robot.

        Args:
            gate_markers (list): List of dictionaries representing gate markers. Each dictionary must have:
                - 'id': marker identifier.
                - 'position': The marker's (x, y) position (as a list or NumPy array).
            robot_position (array-like): The robot's (x, y) position.
            offset (float): Distance offset from the gate center in meters (default is 0.2m).

        Returns:
            np.ndarray: The (x, y) drop location, or None if there are not enough markers.
        """
        # Ensure we have at least two markers to define a gate
        if len(gate_markers) < 2:
            print("[red]Not enough markers to determine gate orientation.")
            return None

        # Sort markers by their x-coordinate (assuming left-to-right ordering)
        sorted_markers = sorted(gate_markers, key=lambda m: m['position'][0])
        
        # Define the left-most and right-most markers
        left_marker = sorted_markers[0]
        right_marker = sorted_markers[-1]
        
        # Convert positions to NumPy arrays
        left_pos = np.array(left_marker['position'])
        right_pos = np.array(right_marker['position'])
        
        # Compute the gate center and the direction from left to right
        gate_center = (left_pos + right_pos) / 2.0
        gate_direction = right_pos - left_pos
        norm = np.linalg.norm(gate_direction)
        if norm == 0:
            print("[Red]Left and right markers are identical; cannot compute gate direction.")
            return None
        gate_direction = gate_direction / norm
        
        # Compute a perpendicular unit vector to the gate.
        # One valid perpendicular is given by (-dy, dx)
        perp = np.array([-gate_direction[1], gate_direction[0]])
        perp_norm = np.linalg.norm(perp)
        if perp_norm == 0:
            print("[red]Cannot compute a perpendicular vector.")
            return None
        perp = perp / perp_norm
        
        # Compute the two candidate drop locations (offset from the gate center)
        drop_candidate1 = gate_center + offset * perp
        drop_candidate2 = gate_center - offset * perp
        
        # Convert the robot's position to a NumPy array
        robot_pos = np.array(robot_position)
        angle = np.arctan2(gate_direction[1], gate_direction[0])
        
        # Choose the candidate that is farther from the robot.
        if np.linalg.norm(drop_candidate1 - robot_pos) < np.linalg.norm(drop_candidate2 - robot_pos):
            return drop_candidate1, angle
        else:
            return drop_candidate2, angle
        



    
    def simplify_path(self, path):
        """Extracts key waypoints where the direction changes."""
        if len(path) < 3:
            return path  # No simplification needed for very short paths

        simplified_path = [path[0]]  # Start with the first point

        for i in range(1, len(path) - 1):
            prev = np.array(path[i - 1])
            curr = np.array(path[i])
            next = np.array(path[i + 1])

            # Calculate movement vectors
            vec1 = curr - prev
            vec2 = next - curr

            # If direction changes, store this point
            if not np.allclose(vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2), atol=0.1):
                simplified_path.append(curr)

        simplified_path.append(path[-1])  # Always include the last point
        return simplified_path


    def get_block_position(self, current_block, data):
        # Retrieve marker positions from SLAM data
        landmark_map = {lid: pos for lid, pos in zip(data.landmark_estimated_ids, data.landmark_estimated_positions)}

        P_large = landmark_map.get(current_block, None)         # Larger marker
        P_small = landmark_map.get(current_block - 1, None)     # Smaller marker

        # Convert positions to NumPy arrays
        P_large = np.array(P_large, dtype=float)
        P_small = np.array(P_small, dtype=float)

        # Compute unit vector along the marker pair direction
        d_vec = P_large - P_small
        norm = np.linalg.norm(d_vec)

        d_unit = d_vec / norm  # Normalize direction

        # Compute block position: 5 cm (0.05 m) in front of the larger marker along d_unit
        P_block = P_large + 0.1 * d_unit


        #print(P_block, P_large, P_small)
        return tuple(P_block)

