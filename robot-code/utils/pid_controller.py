import math

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-math.pi*0.75, math.pi*0.75)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.output_limits = output_limits  # (min_output, max_output)

    def update(self, error, dt):
        if dt <= 0:
            raise ValueError("dt must be greater than zero")
        
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        # Clamp the output using the provided limits
        min_output, max_output = self.output_limits
        output = max(min(output, max_output), min_output)

        return output
