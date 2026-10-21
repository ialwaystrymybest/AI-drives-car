# ============================================================
# main.py — Entry point: sets up NEAT, runs the pygame loop,
# evolves a population of self-driving cars generation by
# generation.  Watch the training live!
#
# Controls:
#   SPACE      Pause / resume
#   UP / DOWN  Change simulation speed (1x → 2x → 5x → 10x)
#   S          Save the best genome to best_genome.pkl
#   ESC        Quit
# ============================================================

import os
import sys
import pickle
import pygame
import neat
from config_variables import *
from road import Road
from car import Car
from visualizer import draw_panel, draw_road_overlay

# ---- Global state shared across generations ----
screen = None
clock = None
generation = 0
best_fitness_ever = 0.0
sim_speed = 1
SPEED_OPTIONS = [1, 2, 5, 10]
speed_index = 0
quit_requested = False
_cached_road = None  # Built once and reused so generations start instantly without lag


# ==============================================================
# This function is called by neat-python once per generation.
# It creates one car + one neural network per genome, then runs
# the pygame simulation until all cars die or time runs out.
# ==============================================================
def eval_genomes(genomes, config):
    global generation, best_fitness_ever, sim_speed, speed_index, quit_requested, _cached_road

    if quit_requested:
        raise SystemExit()

    generation += 1

    # --- Build or reuse the road (reused when ROAD_SEED is fixed) ---
    if _cached_road is None or ROAD_SEED is None:
        _cached_road = Road(seed=ROAD_SEED)
    road = _cached_road

    # --- Create a car and a neural network for every genome ---
    cars = []
    networks = []
    genome_list = []

    for genome_id, genome in genomes:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        car = Car(road.start_x, road.start_y, road.start_angle)
        cars.append(car)
        networks.append(network)
        genome_list.append(genome)
        genome.fitness = 0.0

    total_cars = len(cars)
    frame = 0
    paused = False

    # ----------------------------------------------------------
    # Main simulation loop for this generation
    # ----------------------------------------------------------
    while frame < MAX_FRAMES_PER_GEN:
        # --- Handle keyboard and window events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
                raise SystemExit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_requested = True
                    raise SystemExit()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed_index = min(speed_index + 1, len(SPEED_OPTIONS) - 1)
                    sim_speed = SPEED_OPTIONS[speed_index]
                elif event.key == pygame.K_DOWN:
                    speed_index = max(speed_index - 1, 0)
                    sim_speed = SPEED_OPTIONS[speed_index]
                elif event.key == pygame.K_s:
                    _save_best(genome_list, cars)

        if paused:
            _draw_frame(screen, road, cars, networks, genome_list,
                        config, frame, total_cars)
            clock.tick(FPS)
            continue

        # --- Run physics steps (multiple if sim_speed > 1) ---
        for _ in range(sim_speed):
            alive_count = 0
            for i in range(total_cars):
                if not cars[i].alive:
                    continue
                alive_count += 1

                # 1) Cast sensors to see the road
                sensor_data = cars[i].cast_sensors(road)

                # 2) Build the 9 inputs for the neural network
                inputs = sensor_data + [cars[i].speed / MAX_SPEED]

                # 3) Ask the neural network what to do
                outputs = networks[i].activate(inputs)

                # 4) Move the car
                cars[i].update(outputs)

                # 5) Check if still alive and update fitness
                cars[i].check_alive(road, frame)

                # 6) Store fitness on the genome (NEAT reads this later)
                genome_list[i].fitness = cars[i].fitness

            frame += 1
            if frame >= MAX_FRAMES_PER_GEN or alive_count == 0:
                break

        # --- Draw everything once per display frame ---
        _draw_frame(screen, road, cars, networks, genome_list,
                    config, frame, total_cars)
        clock.tick(FPS)

        # End generation if all cars are dead
        if all(not c.alive for c in cars):
            break

    # --- Final fitness assignment ---
    for i in range(total_cars):
        genome_list[i].fitness = cars[i].fitness
        if cars[i].fitness > best_fitness_ever:
            best_fitness_ever = cars[i].fitness


# ==============================================================
# Draw one frame: road, cars, top panel, overlay
# ==============================================================
def _draw_frame(screen, road, cars, networks, genome_list, config, frame, total_cars):
    global best_fitness_ever

    # Find the best alive car (highest fitness among alive cars)
    best_i = -1
    best_fit = -1
    alive_count = 0
    for i, car in enumerate(cars):
        if car.alive:
            alive_count += 1
            if car.fitness > best_fit:
                best_fit = car.fitness
                best_i = i

    # If no car is alive, use the one with overall best fitness
    if best_i < 0:
        for i, car in enumerate(cars):
            if car.fitness > best_fit:
                best_fit = car.fitness
                best_i = i

    # Camera follows the best car
    if best_i >= 0:
        cam_x = cars[best_i].x - SCREEN_WIDTH / 2
        cam_y = cars[best_i].y - (SCREEN_HEIGHT - TOP_PANEL_HEIGHT) * 0.65
    else:
        cam_x = 0
        cam_y = 0

    # --- Draw grass background for the road view area ---
    road_area = pygame.Rect(0, TOP_PANEL_HEIGHT, SCREEN_WIDTH,
                            SCREEN_HEIGHT - TOP_PANEL_HEIGHT)
    screen.fill(COLOR_GRASS, road_area)

    # --- Draw the road ---
    road.draw(screen, cam_x, cam_y)

    # --- Draw all non-best alive cars (semi-transparent) ---
    for i, car in enumerate(cars):
        if car.alive and i != best_i:
            car.draw(screen, cam_x, cam_y, is_best=False)

    # --- Draw the best car on top (highlighted, with sensors) ---
    if best_i >= 0:
        cars[best_i].draw(screen, cam_x, cam_y, is_best=True, show_sensors=True)

    # --- Build stats dictionary for the visualizer ---
    best_genome = genome_list[best_i] if best_i >= 0 else None
    best_net = networks[best_i] if best_i >= 0 else None
    best_car = cars[best_i] if best_i >= 0 else None

    # Figure out steering and action labels from outputs
    steering = "Straight"
    action = "Coasting"
    if best_car and best_car.alive:
        outs = best_car.current_outputs
        if outs[2] > 0.5 and outs[3] <= 0.5:
            steering = f"Left ({outs[2]:.2f})"
        elif outs[3] > 0.5 and outs[2] <= 0.5:
            steering = f"Right ({outs[3]:.2f})"
        elif outs[2] > 0.5 and outs[3] > 0.5:
            steering = "Both"
        if outs[0] > 0.5:
            action = "Accelerating"
        if outs[1] > 0.5:
            action = "Braking"

    # Count nodes and connections in the best genome
    n_nodes = len(best_genome.nodes) if best_genome else 0
    n_conns = sum(1 for c in best_genome.connections.values() if c.enabled) if best_genome else 0

    stats = {
        "generation": generation,
        "alive": alive_count,
        "total": total_cars,
        "best_fitness": best_fit if best_fit > 0 else 0,
        "best_ever": best_fitness_ever,
        "speed": best_car.speed if best_car else 0,
        "steering": steering,
        "action": action,
        "species": 0,  # Filled below if possible
        "nodes": n_nodes,
        "connections": n_conns,
        "sim_speed": sim_speed,
        "fps": clock.get_fps(),
    }

    # --- Draw the top panel (NN + stats) ---
    draw_panel(screen, best_genome, best_net, config, stats)

    # --- Draw the small overlay on the road view ---
    draw_road_overlay(screen, stats)

    pygame.display.flip()


# ==============================================================
# Save the genome with the highest fitness to a pickle file
# ==============================================================
def _save_best(genome_list, cars):
    best_i = 0
    for i in range(len(cars)):
        if cars[i].fitness > cars[best_i].fitness:
            best_i = i
    path = os.path.join(os.path.dirname(__file__), "best_genome.pkl")
    with open(path, "wb") as f:
        pickle.dump(genome_list[best_i], f)
    print(f"[Saved] best genome to {path}  (fitness = {cars[best_i].fitness:.1f})")


# ==============================================================
# Entry point
# ==============================================================
def main():
    global screen, clock

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NEAT Self-Driving Car — Training")
    clock = pygame.time.Clock()

    # Load the NEAT configuration file
    config_path = os.path.join(os.path.dirname(__file__), "config_file.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create the NEAT population
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats_reporter = neat.StatisticsReporter()
    population.add_reporter(stats_reporter)

    # Run evolution for up to 300 generations (or until fitness_threshold)
    try:
        winner = population.run(eval_genomes, 300)
        # Save the winner automatically
        path = os.path.join(os.path.dirname(__file__), "best_genome.pkl")
        with open(path, "wb") as f:
            pickle.dump(winner, f)
        print(f"\nBest genome saved to {path}")
    except SystemExit:
        print("\nTraining stopped by user.")

    pygame.quit()


if __name__ == "__main__":
    main()
