from dataclasses import dataclass, field
from typing import List

@dataclass
class Message:
    id: int = 0
    timestamp: float = 0
    start: bool = False
    
    landmark_ids: List = field(default_factory=lambda: [])
    landmark_rs: List = field(default_factory=lambda: [])
    landmark_alphas: List = field(default_factory=lambda: [])
    landmark_positions: List = field(default_factory=lambda: [])
    
    landmark_estimated_ids: List = field(default_factory=lambda: [])
    landmark_estimated_positions: List = field(default_factory=lambda: [])
    landmark_estimated_stdevs: List = field(default_factory=lambda: [])

    robot_position: List = field(default_factory=lambda: [])
    robot_theta: float = 0.0
    robot_stdev: List = field(default_factory=lambda: [])
    
    grid_map: List = field(default_factory=lambda: [])
    grid_origin: List = field(default_factory=lambda: [])
    grid_resolution: float = 0.0
    current_path: List = field(default_factory=lambda: [])

    text: str = ""
