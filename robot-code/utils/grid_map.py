import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class GridMap:
    def __init__(self, width, height, resolution=0.01):
        self.resolution = resolution
        self.width = width
        self.height = height
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        self.grid = np.zeros((self.grid_height, self.grid_width))
        self.shape = self.grid.shape
        
    def world_to_grid(self, x, y):
        grid_x = int((x + self.width / 2) / self.resolution)
        grid_y = int((y + self.height / 2) / self.resolution)
        return np.clip(grid_x, 0, self.grid_width - 1), np.clip(grid_y, 0, self.grid_height - 1)
    
    def grid_to_world(self, grid_x, grid_y):
        x = grid_x * self.resolution - self.width / 2
        y = grid_y * self.resolution - self.height / 2
        return x, y
    
    def add_marker(self, x, y, radius=0.3, safety_margin=0.2):
        grid_x, grid_y = self.world_to_grid(x, y)
        grid_radius = int((radius + safety_margin) / self.resolution)
        
        for i in range(-grid_radius, grid_radius + 1):
            for j in range(-grid_radius, grid_radius + 1):
                new_x, new_y = grid_x + i, grid_y + j
                if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                    if np.sqrt(i**2 + j**2) <= grid_radius:
                        self.grid[new_y, new_x] = max(self.grid[new_y, new_x], 0.7)
                        
    def add_block(self, x, y, radius=0.25, safety_margin=0.05):
        """adds a block with margin to the map"""
        self.add_marker(x, y, radius=radius, safety_margin=safety_margin)
        
    def add_potential_block(self, x, y, radius=0.35, safety_margin=0.15):
        """adds a potential block with margin to the map"""
        self.add_marker(x, y, radius=radius, safety_margin=safety_margin)
    
    
    def is_valid_position(self, x, y, threshold=0.5):
        grid_x, grid_y = self.world_to_grid(x, y)
        return self.grid[grid_y, grid_x] < threshold
    
    def get_grid_and_origin(self):
        """returns the grid and its origin in world coordinates"""
        origin_x = -self.width/2
        origin_y = -self.height/2
        return self.grid, (origin_x, origin_y)
    
    def update_from_exploration(self, exploration, delivered_blocks):
        """Updates the grid map based on exploration data"""
        self.grid.fill(0)
        
        # Add border markers
        for marker_pos in exploration.border_markers.values():
            self.add_marker(marker_pos["position"][0], marker_pos["position"][1], radius=0.15, safety_margin=0.0)
        
        # Add obstacle markers
        for marker_pos in exploration.obstacle_markers.values():
            self.add_marker(marker_pos["position"][0], marker_pos["position"][1], radius=0.15, safety_margin=0.0)
        
        # Add blocks
        for blocks in [exploration.red_blocks, exploration.blue_blocks]:
            for block_data in blocks.values():
                if "marker1" in block_data and "marker2" in block_data and (block_data['marker1']['id'] in delivered_blocks or block_data['marker2']['id'] in delivered_blocks):
                    continue
                pos = block_data['position']
                self.add_block(pos[0], pos[1])
                
        # Add potential blocks
        for blocks in [exploration.potential_red_blocks, exploration.potential_blue_blocks]:
            for block_data in blocks.values():
                if "marker1" in block_data and "marker2" in block_data and (block_data['marker1']['id'] in delivered_blocks or block_data['marker2']['id'] in delivered_blocks):
                    continue
                pos = block_data['position']
                self.add_potential_block(pos[0], pos[1])
        
        return self.grid, (-self.width/2, -self.height/2), self.resolution

    def save_map(self, filename, exploration=None, robot_pos=None, current_path=None):
        """Saves a visualization of the map with all elements"""
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        extent = [-self.width/2, self.width/2, -self.height/2, self.height/2]
        ax.imshow(self.grid, origin='lower', extent=extent, cmap='gray_r')
        
        if exploration:
            # Plot border markers
            border_x = [pos["position"][0] for pos in exploration.border_markers.values()]
            border_y = [pos["position"][1] for pos in exploration.border_markers.values()]
            ax.scatter(border_x, border_y, c='green', marker='o', label='Border Markers')
            
            # Plot obstacle markers
            obstacle_x = [pos["position"][0] for pos in exploration.obstacle_markers.values()]
            obstacle_y = [pos["position"][1] for pos in exploration.obstacle_markers.values()]
            ax.scatter(obstacle_x, obstacle_y, c='orange', marker='o', label='Obstacle Markers')
            
            # Plot gates
            if exploration.gate_positions['red'] is not None:
                poses = exploration.gate_positions['red']
                for pos in poses:
                    ax.scatter(pos['position'][0], pos['position'][1], c='red', marker='*', s=200, label='Red Gate')
            
            if exploration.gate_positions['blue'] is not None:
                poses = exploration.gate_positions['blue']
                for pos in poses:
                    ax.scatter(pos['position'][0], pos['position'][1], c='blue', marker='*', s=200, label='Blue Gate')
                    
            # Plot potential blocks
            for blocks in [exploration.potential_red_blocks, exploration.potential_blue_blocks]:
                for block_data in blocks.values():
                    pos = block_data['position']
                    ax.scatter(pos[0], pos[1], c='gray', marker='o', label='Potential Block')
            
            # Plot red blocks and their markers
            for block_id, block_data in exploration.red_blocks.items():
                pos = block_data['position']
                marker1 = block_data['marker1']
                marker2 = block_data['marker2']
                
                # Plot block
                ax.scatter(pos[0], pos[1], c='red', marker='o', label='Red Block')
                
                # Plot markers with IDs
                ax.scatter(marker1['position'][0], marker1['position'][1], c='pink', marker='s', label='Block Marker')
                ax.text(marker1['position'][0], marker1['position'][1], str(marker1['id']), fontsize=8)
                ax.scatter(marker2['position'][0], marker2['position'][1], c='pink', marker='s')
                ax.text(marker2['position'][0], marker2['position'][1], str(marker2['id']), fontsize=8)
            
            # Plot blue blocks and their markers
            for block_id, block_data in exploration.blue_blocks.items():
                pos = block_data['position']
                marker1 = block_data['marker1']
                marker2 = block_data['marker2']
                
                # Plot block
                ax.scatter(pos[0], pos[1], c='blue', marker='o', label='Blue Block')
                
                # Plot markers with IDs
                ax.scatter(marker1['position'][0], marker1['position'][1], c='lightblue', marker='s', label='Block Marker')
                ax.text(marker1['position'][0], marker1['position'][1], str(marker1['id']), fontsize=8)
                ax.scatter(marker2['position'][0], marker2['position'][1], c='lightblue', marker='s')
                ax.text(marker2['position'][0], marker2['position'][1], str(marker2['id']), fontsize=8)
        
            if exploration.hull_points is not None:
                ax.scatter(exploration.hull_points[:, 0], exploration.hull_points[:, 1], c='black', marker='o', label='Convex Hull Points')
        
        if robot_pos is not None:
            ax.scatter(robot_pos[0], robot_pos[1], c='green', marker='^', s=200, label='Robot')
        
        if current_path is not None and len(current_path) > 1:
            path_x = []
            path_y = []
            for p in current_path:
                world_x, world_y = self.grid_to_world(p[0], p[1])
                path_x.append(world_x)
                path_y.append(world_y)
            ax.plot(path_x, path_y, 'g--', label='Current Path')
        
        ax.grid(True)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Grid Map Visualization')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[green]Map saved as {filename}")