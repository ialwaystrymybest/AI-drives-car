# ============================================================
# car.py — The Car class
# Each car has position, angle, speed, 8 distance sensors,
# and a fitness score based on how far it drives along the road.
# ============================================================

import math
import pygame
from config_variables import *

# Sensor ray angles relative to the car heading (degrees)
# 0 = straight ahead,  negative = left,  positive = right
SENSOR_ANGLES = [0, -45, 45, -90, 90, 180, -135, 135]


class Car:
    """A simple top-down car with physics, sensors, and fitness tracking."""

    def __init__(self, x, y, angle):
        # Position and movement
        self.x = x
        self.y = y
        self.angle = angle         # 0 = facing up (north), clockwise positive
        self.speed = 0.0

        # Status
        self.alive = True

        # Progress tracking (for fitness and stuck detection)
        self.road_index = 0        # Index of the nearest center-line point right now
        self.max_road_index = 0    # Furthest center-line point ever reached
        self.last_progress_frame = 0  # Frame number when progress was last made
        self.fitness = 0.0

        # Sensor data (filled in by cast_sensors)
        self.sensor_distances = [1.0] * SENSOR_COUNT   # Normalized 0-1
        self.sensor_endpoints = [(x, y)] * SENSOR_COUNT  # World coords of ray hits

        # Neural network outputs (saved for the visualizer)
        self.current_outputs = [0.0, 0.0, 0.0, 0.0]

    # ==============================================================
    # Cast all 8 sensor rays and find how far each one reaches
    # before leaving the road.  Returns a list of 8 floats (0-1).
    # ==============================================================
    def cast_sensors(self, road):
        self.sensor_distances = []
        self.sensor_endpoints = []

        for angle_offset in SENSOR_ANGLES:
            ray_angle = math.radians(self.angle + angle_offset)
            dx = math.sin(ray_angle)
            dy = -math.cos(ray_angle)

            # Walk along the ray in small steps until we leave the road
            hit_dist = SENSOR_LENGTH
            max_steps = SENSOR_LENGTH // SENSOR_STEP
            for step in range(1, max_steps + 1):
                dist = step * SENSOR_STEP
                check_x = self.x + dx * dist
                check_y = self.y + dy * dist
                if not road.is_on_road(check_x, check_y):
                    hit_dist = dist
                    break

            # Normalize: 0 = very close to edge, 1 = nothing detected (safe)
            self.sensor_distances.append(hit_dist / SENSOR_LENGTH)
            self.sensor_endpoints.append((
                self.x + dx * hit_dist,
                self.y + dy * hit_dist
            ))

        return self.sensor_distances

    # ==============================================================
    # Apply neural-network outputs and update car physics.
    # outputs = [accelerate, brake, turn_left, turn_right]
    # Each output is a float; values above 0.5 count as "pressed".
    # ==============================================================
    def update(self, outputs):
        if not self.alive:
            return

        self.current_outputs = list(outputs)

        # Decide which actions the neural network chose
        do_accelerate = outputs[0] > 0.5
        do_brake = outputs[1] > 0.5
        do_turn_left = outputs[2] > 0.5
        do_turn_right = outputs[3] > 0.5

        # Accelerate or brake
        if do_accelerate:
            self.speed += ACCELERATION
        if do_brake:
            self.speed -= BRAKE_POWER

        # Friction always slows the car down a little
        self.speed -= FRICTION

        # Keep speed between 0 and MAX_SPEED
        if self.speed < 0:
            self.speed = 0.0
        if self.speed > MAX_SPEED:
            self.speed = MAX_SPEED

        # Turning depends on speed — can't steer a parked car
        turn_factor = min(self.speed / 4.0, 1.0)
        if do_turn_left:
            self.angle -= TURN_SPEED * turn_factor
        if do_turn_right:
            self.angle += TURN_SPEED * turn_factor

        # Move forward in the direction the car is facing
        rad = math.radians(self.angle)
        self.x += math.sin(rad) * self.speed
        self.y -= math.cos(rad) * self.speed

    # ==============================================================
    # Check if this car should be eliminated (off-road, stuck, etc.)
    # and update its fitness score.
    # ==============================================================
    def check_alive(self, road, frame):
        if not self.alive:
            return

        # 1) Off-road check
        if not road.is_on_road(self.x, self.y):
            self.alive = False
            return

        # 2) Update road progress
        new_idx = road.get_nearest_index(self.x, self.y, self.road_index)
        self.road_index = new_idx
        if new_idx > self.max_road_index:
            self.max_road_index = new_idx
            self.last_progress_frame = frame

        # 3) Stuck timeout — no forward progress for too long
        if frame - self.last_progress_frame > STUCK_TIMEOUT_FRAMES:
            self.alive = False
            return

        # 4) Going backwards — car moved far behind its best position
        if self.road_index < self.max_road_index - BACKWARD_THRESHOLD:
            self.alive = False
            return

        # 5) Update fitness = distance along road + tiny speed bonus
        self.fitness = road.cumulative_dist[self.max_road_index]
        self.fitness += self.speed * SPEED_BONUS_WEIGHT

    # ==============================================================
    # Draw the car on screen (rotated rectangle + optional sensors)
    # ==============================================================
    def draw(self, screen, camera_x, camera_y, is_best=False, show_sensors=False):
        if not self.alive and not is_best:
            return  # Don't draw dead non-best cars

        # Convert world position to screen position
        scr_x = self.x - camera_x
        scr_y = self.y - camera_y + TOP_PANEL_HEIGHT

        # Skip if off-screen
        if scr_x < -100 or scr_x > SCREEN_WIDTH + 100:
            return
        if scr_y < TOP_PANEL_HEIGHT - 100 or scr_y > SCREEN_HEIGHT + 100:
            return

        # --- Draw sensor rays for the best car ---
        if show_sensors and self.alive:
            for ep in self.sensor_endpoints:
                ex = ep[0] - camera_x
                ey = ep[1] - camera_y + TOP_PANEL_HEIGHT
                pygame.draw.line(screen, COLOR_SENSOR_LINE,
                                 (int(scr_x), int(scr_y)), (int(ex), int(ey)), 1)
                pygame.draw.circle(screen, COLOR_SENSOR_DOT, (int(ex), int(ey)), 3)

        # --- Create a small surface for the car body, then rotate it ---
        car_surf = pygame.Surface((CAR_WIDTH, CAR_LENGTH), pygame.SRCALPHA)

        if is_best:
            body_color = COLOR_CAR_BEST
        else:
            body_color = (*COLOR_CAR_BODY, 110)  # Semi-transparent for non-best

        # Body rectangle with rounded corners
        pygame.draw.rect(car_surf, body_color,
                         (0, 0, CAR_WIDTH, CAR_LENGTH), border_radius=5)

        # Windshield near the front (top of the surface)
        windshield_color = (120, 190, 255, 200)
        pygame.draw.rect(car_surf, windshield_color,
                         (3, 3, CAR_WIDTH - 6, 7), border_radius=2)

        # Tail lights at the back
        tail_color = (255, 60, 60, 200)
        pygame.draw.rect(car_surf, tail_color, (1, CAR_LENGTH - 5, 4, 3))
        pygame.draw.rect(car_surf, tail_color, (CAR_WIDTH - 5, CAR_LENGTH - 5, 4, 3))

        # Rotate and blit
        rotated = pygame.transform.rotate(car_surf, -self.angle)
        rect = rotated.get_rect(center=(int(scr_x), int(scr_y)))
        screen.blit(rotated, rect)
