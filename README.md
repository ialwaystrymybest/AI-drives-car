# NEAT Self-Driving Car

An AI self-driving car simulation built from scratch in Python with **Pygame** and the **NEAT** (NeuroEvolution of Augmenting Topologies) genetic algorithm.

A population of cars learns to navigate a procedurally generated winding road with smooth curves, curbs, and scenery. You can watch the evolution and the best car's neural network live in real time.

---

## Features

- **Procedural Track Generation**: Generates smooth winding tracks using random waypoints and `scipy.interpolate.CubicSpline`, with rasterized $O(1)$ grid collision checking.
- **Top-Down Car Physics**: Acceleration, braking, speed-dependent steering, drag friction, and inertia.
- **8 Directional Distance Sensors**: Front, Front-Left, Front-Right, Left, Right, Back, Back-Left, Back-Right raycasts detecting the road boundary.
- **Live Neural Network HUD**: Visualizes the active brain of the generation's best car, with connections weighted by strength and nodes glowing according to their real-time activation.
- **Interactive Controls**: Pause, adjust simulation speed up to 10x, save genomes on demand, or replay the best trained car.
- **Beginner-Friendly Architecture**: Clean modular files, plain Python loops and functions, simple classes, no decorators or complex abstractions, and descriptive comments throughout.

---

## Project Structure

```text
.
├── config_variables.py   # All tweakable parameters (speeds, colors, dimensions, etc.)
├── config_file.txt       # NEAT algorithm configuration (population, mutations, inputs/outputs)
├── car.py                # Car physics, sensor raycasting, and fitness tracking
├── road.py               # Procedural road generation (CubicSpline) and fast collision grid
├── visualizer.py         # Real-time neural network graph and stats dashboard renderer
├── main.py               # Main training loop with live visualization and keyboard controls
├── run_best.py           # Replay viewer for saved best genomes (best_genome.pkl)
├── requirements.txt      # Project dependencies
└── best_genome.pkl       # Saved trained champion genome
```

---

## Getting Started

### 1. Prerequisites & Environment Setup

Python 3.10+ is recommended. Activate the existing virtual environment:

```bash
source venv/bin/activate
```

Or install the dependencies directly if setting up a fresh environment:

```bash
pip install -r requirements.txt
```

*(Dependencies: `pygame`, `neat-python`, `numpy`, `scipy`)*

---

## Running the Project

### Train and Watch Live
Start the training process:

```bash
python main.py
```

### Controls During Training
| Key | Action |
|---|---|
| <kbd>SPACE</kbd> | Pause / Resume simulation |
| <kbd>UP</kbd> | Increase simulation speed (1x &rarr; 2x &rarr; 5x &rarr; 10x) |
| <kbd>DOWN</kbd> | Decrease simulation speed (10x &rarr; 5x &rarr; 2x &rarr; 1x) |
| <kbd>S</kbd> | Save the current best performing genome to `best_genome.pkl` |
| <kbd>ESC</kbd> | Stop training and exit |

---

### Replay the Best Driver
To watch the saved champion car drive on its own:

```bash
python run_best.py
```

Replay controls:
- <kbd>R</kbd>: Restart the run from the starting line.
- <kbd>ESC</kbd>: Quit replay.

---

## Experimenting & Customization

All physics, sensor, rendering, and simulation settings are isolated in [config_variables.py](config_variables.py).
Try tweaking:
- `ROAD_WIGGLE`: Increase for sharper, harder turns.
- `MAX_SPEED` & `TURN_SPEED`: Change car handling dynamics.
- `SENSOR_COUNT` & `SENSOR_LENGTH`: Alter how far and wide the cars can see.
- `POPULATION_SIZE`: Change the number of cars per generation (also update `pop_size` in `config_file.txt`).
