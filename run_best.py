# ============================================================
# run_best.py — Watch the best saved genome drive the road
# Loads best_genome.pkl (created by pressing S during training
# or automatically when training finishes) and replays it.
#
# Usage:   python run_best.py
# ============================================================

import os
import pickle
import pygame
import neat
from config_variables import *
from road import Road
from car import Car
from visualizer import draw_panel, draw_road_overlay, _get_font


def main():
    # --- Load the saved best genome ---
    pkl_path = os.path.join(os.path.dirname(__file__), "best_genome.pkl")
    if not os.path.exists(pkl_path):
        print("ERROR: best_genome.pkl not found!")
        print("Train first (python main.py) and press S to save, or let training finish.")
        return

    with open(pkl_path, "rb") as f:
        genome = pickle.load(f)

    print(f"Loaded genome with {len(genome.nodes)} nodes, "
          f"{len(genome.connections)} connections")

    # --- Load NEAT config (needed to build the network) ---
    config_path = os.path.join(os.path.dirname(__file__), "config_file.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # --- Build the neural network from the genome ---
    network = neat.nn.FeedForwardNetwork.create(genome, config)

    # --- Set up pygame ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NEAT Self-Driving Car — Best Genome Replay")
    clock = pygame.time.Clock()

    # --- Create the road (same seed as training) and a single car ---
    road = Road(seed=ROAD_SEED)
    car = Car(road.start_x, road.start_y, road.start_angle)

    frame = 0
    running = True

    while running:
        # --- Handle events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Restart the run
                    car = Car(road.start_x, road.start_y, road.start_angle)
                    frame = 0

        # --- Update the car if it is still alive ---
        if car.alive:
            sensor_data = car.cast_sensors(road)
            inputs = sensor_data + [car.speed / MAX_SPEED]
            outputs = network.activate(inputs)
            car.update(outputs)
            car.check_alive(road, frame)
            frame += 1

        # --- Camera follows the car ---
        cam_x = car.x - SCREEN_WIDTH / 2
        cam_y = car.y - (SCREEN_HEIGHT - TOP_PANEL_HEIGHT) * 0.65

        # --- Draw ---
        road_area = pygame.Rect(0, TOP_PANEL_HEIGHT, SCREEN_WIDTH,
                                SCREEN_HEIGHT - TOP_PANEL_HEIGHT)
        screen.fill(COLOR_GRASS, road_area)
        road.draw(screen, cam_x, cam_y)
        car.draw(screen, cam_x, cam_y, is_best=True, show_sensors=True)

        # Stats for the panel
        stats = {
            "generation": "Replay",
            "alive": 1 if car.alive else 0,
            "total": 1,
            "best_fitness": car.fitness,
            "best_ever": car.fitness,
            "speed": car.speed,
            "steering": "Straight",
            "action": "Coasting",
            "species": 0,
            "nodes": len(genome.nodes),
            "connections": sum(1 for c in genome.connections.values() if c.enabled),
            "sim_speed": 1,
            "fps": clock.get_fps(),
        }

        # Figure out steering/action labels
        outs = car.current_outputs
        if outs[2] > 0.5 and outs[3] <= 0.5:
            stats["steering"] = f"Left ({outs[2]:.2f})"
        elif outs[3] > 0.5 and outs[2] <= 0.5:
            stats["steering"] = f"Right ({outs[3]:.2f})"
        if outs[0] > 0.5:
            stats["action"] = "Accelerating"
        if outs[1] > 0.5:
            stats["action"] = "Braking"

        draw_panel(screen, genome, network, config, stats)
        draw_road_overlay(screen, stats)

        if not car.alive:
            font = _get_font(36)
            msg = font.render("Car crashed!  Press R to restart  |  ESC to quit",
                              True, (255, 100, 100))
            screen.blit(msg, (SCREEN_WIDTH // 2 - msg.get_width() // 2,
                              SCREEN_HEIGHT // 2))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
