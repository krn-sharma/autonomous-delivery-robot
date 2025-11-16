import heapq
import numpy as np

def a_star(grid_map, goal, data):
    """A* search algorithm for pathfinding with safety cost consideration."""
    
    grid_x, grid_y = grid_map.world_to_grid(data.robot_position[0], data.robot_position[1])
    start = (grid_x, grid_y)
    goal = grid_map.world_to_grid(goal[0], goal[1])

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def movement_cost(current, neighbor):
        # Base cost is 1 for movement
        base_cost = 1
        
        # Get the safety value (already in steps of 0.1)
        safety_value = grid_map.grid[neighbor[1], neighbor[0]]
        
        # Simple linear penalty based on discrete safety values
        # This will give penalties in steps of 0.5 for each 0.1 increase in safety value
        safety_penalty = 5 * safety_value
        
        return base_cost + safety_penalty

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if (0 <= neighbor[0] < grid_map.grid_width and 
                0 <= neighbor[1] < grid_map.grid_height and 
                grid_map.grid[neighbor[1], neighbor[0]] < 0.8):  # Allow passage through safer areas
                
                tentative_g_score = g_score[current] + movement_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  # No path found



def simplify_path(path):
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