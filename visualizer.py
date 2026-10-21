# ============================================================
# visualizer.py — Live neural network drawing + stats panel
# Shows the current best car's brain (nodes and connections)
# and live training statistics in the top panel of the window.
# ============================================================

import pygame
from config_variables import *

# Try standard pygame.font first, or fall back to pygame._freetype
# (needed on some Python 3.14 environments where pygame.font has a circular import)
try:
    import pygame.font
    pygame.font.init()
    _test = pygame.font.Font(None, 16)
    _USE_FREETYPE = False
except Exception:
    from pygame import _freetype
    _freetype.init()
    _USE_FREETYPE = True

# Cache fonts so we only create them once
_fonts = {}


class _FreetypeWrapper:
    """Wrapper around pygame._freetype.Font to provide the same render() API as pygame.font.Font."""

    def __init__(self, size):
        self._font = _freetype.Font(None, size)

    def render(self, text, antialias, color, bgcolor=None):
        surf, _ = self._font.render(str(text), color, bgcolor)
        return surf

    def size(self, text):
        rect = self._font.get_rect(str(text))
        return (rect.width, rect.height)


def _create_font(size):
    """Return a pygame font or freetype wrapper font of the given size."""
    if _USE_FREETYPE:
        return _FreetypeWrapper(size)
    return pygame.font.Font(None, size)


def _get_font(size):
    """Return a cached pygame font of the given size."""
    if size not in _fonts:
        _fonts[size] = _create_font(size)
    return _fonts[size]


# Labels shown next to each input and output neuron
INPUT_LABELS = ["S1 Front", "S2 F-Left", "S3 F-Right", "S4 Left",
                "S5 Right", "S6 Back", "S7 B-Left", "S8 B-Right", "Speed"]
OUTPUT_LABELS = ["Accel", "Brake", "Left", "Right"]


# Cache computed node positions for genomes so we do not recompute layout every frame
_node_positions_cache = {}


# ==============================================================
# Figure out where to draw each neuron in the network panel.
# Returns a dict:  node_key -> (pixel_x, pixel_y)
# ==============================================================
def _compute_node_positions(genome, config, panel_rect):
    """Lay out input, hidden, and output nodes in columns."""
    cache_key = (getattr(genome, "key", None), panel_rect.width, panel_rect.height)
    if cache_key[0] is not None and cache_key in _node_positions_cache:
        return _node_positions_cache[cache_key]

    input_keys = list(config.genome_config.input_keys)    # e.g. [-1, -2, … -9]
    output_keys = list(config.genome_config.output_keys)  # e.g. [0, 1, 2, 3]

    # --- Find hidden nodes using neat's feed-forward layer ordering ---
    enabled_conns = [cg.key for cg in genome.connections.values() if cg.enabled]

    hidden_layers = []
    try:
        from neat.graphs import feed_forward_layers
        res = feed_forward_layers(input_keys, output_keys, enabled_conns)
        # In neat-python 2.0, res is a (layers, required) tuple; in 0.92 it is just layers
        layers = res[0] if isinstance(res, tuple) else res
        for layer in layers:
            hidden = [n for n in layer if n not in output_keys and n not in input_keys]
            if hidden:
                hidden_layers.append(sorted(hidden))
    except Exception:
        pass  # If something goes wrong, fall through to fallback

    # Fallback: check if any hidden nodes in the genome were missed
    all_known_hidden = set()
    for layer in hidden_layers:
        all_known_hidden.update(layer)
    missed_hidden = [n for n in genome.nodes.keys()
                     if n not in output_keys and n not in input_keys and n not in all_known_hidden]
    if missed_hidden:
        hidden_layers.append(sorted(missed_hidden))

    # Build the complete list of columns: [inputs, hidden…, outputs]
    all_columns = [input_keys] + hidden_layers + [output_keys]
    num_cols = len(all_columns)

    # Margins inside the neural network card (leaves room for input/output text)
    left_margin = 100
    right_margin = 105
    available_w = panel_rect.width - left_margin - right_margin
    max_spacing = 260 if num_cols <= 2 else (175 if num_cols == 3 else 135)

    if (num_cols - 1) * max_spacing <= available_w:
        total_span = (num_cols - 1) * max_spacing
        start_x = panel_rect.x + left_margin + (available_w - total_span) // 2
        col_spacing = max_spacing
    else:
        start_x = panel_rect.x + left_margin
        col_spacing = int(available_w / (num_cols - 1))

    # Vertical margins inside the neural network card
    margin_top = 46
    margin_bottom = 18
    usable_h = panel_rect.height - margin_top - margin_bottom

    positions = {}
    for col_i, column in enumerate(all_columns):
        px = start_x + col_i * col_spacing
        num_nodes = len(column)
        for row_i, node_key in enumerate(column):
            if num_nodes == 1:
                py = panel_rect.y + margin_top + usable_h // 2
            else:
                py = panel_rect.y + margin_top + int(row_i / (num_nodes - 1) * usable_h)
            positions[node_key] = (px, py)

    if cache_key[0] is not None:
        if len(_node_positions_cache) > 200:
            _node_positions_cache.clear()
        _node_positions_cache[cache_key] = (positions, input_keys, output_keys)

    return positions, input_keys, output_keys


# ==============================================================
# Pick a color that shows how "active" a neuron is.
# Warm orange-yellow for positive, cool blue for negative.
# ==============================================================
def _activation_color(value):
    v = max(-1.0, min(1.0, value))
    if v >= 0:
        r = int(60 + 195 * v)
        g = int(60 + 160 * v)
        b = 60
    else:
        r = 60
        g = 60
        b = int(60 + 195 * abs(v))
    return (r, g, b)


# ==============================================================
# Main draw function: renders the top panel with NN + stats
# ==============================================================
def draw_panel(screen, genome, network, neat_config, stats):
    """
    Draw the top panel containing:
      - LEFT:  live neural network of the best car
      - RIGHT: 2-column dashboard stats (generation, alive, fitness, speed, etc.)
    """
    # --- Top panel background ---
    panel = pygame.Rect(0, 0, SCREEN_WIDTH, TOP_PANEL_HEIGHT)
    pygame.draw.rect(screen, COLOR_PANEL_BG, panel)
    pygame.draw.line(screen, COLOR_PANEL_BORDER,
                     (0, TOP_PANEL_HEIGHT - 1), (SCREEN_WIDTH, TOP_PANEL_HEIGHT - 1), 2)

    font_title = _get_font(19)
    font_sm = _get_font(16)
    font_val = _get_font(16)

    # ----------------------------------------------------------
    # Card layout dimensions
    # ----------------------------------------------------------
    nn_w = 680
    nn_rect = pygame.Rect(12, 8, nn_w, TOP_PANEL_HEIGHT - 16)
    stats_x = nn_rect.right + 12
    stats_w = SCREEN_WIDTH - stats_x - 12
    stats_rect = pygame.Rect(stats_x, 8, stats_w, TOP_PANEL_HEIGHT - 16)

    # Draw card backgrounds and borders
    card_bg = (24, 26, 36)
    pygame.draw.rect(screen, card_bg, nn_rect, border_radius=8)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, nn_rect, 1, border_radius=8)
    pygame.draw.rect(screen, card_bg, stats_rect, border_radius=8)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, stats_rect, 1, border_radius=8)

    # Header divider lines inside cards
    divider_color = (36, 42, 58)
    pygame.draw.line(screen, divider_color, (nn_rect.x + 12, nn_rect.y + 32), (nn_rect.right - 12, nn_rect.y + 32), 1)
    pygame.draw.line(screen, divider_color, (stats_rect.x + 12, stats_rect.y + 32), (stats_rect.right - 12, stats_rect.y + 32), 1)

    # Card titles
    t1 = font_title.render("Neural Network (Best Car)", True, COLOR_TEXT)
    screen.blit(t1, (nn_rect.x + 16, nn_rect.y + 7))
    t2 = font_title.render("Training Stats", True, COLOR_TEXT)
    screen.blit(t2, (stats_rect.x + 16, stats_rect.y + 7))

    # If there is no genome to show, just print a message inside the NN card
    if genome is None or network is None:
        msg = font_title.render("Waiting for cars…", True, COLOR_TEXT_DIM)
        screen.blit(msg, (nn_rect.centerx - msg.get_width() // 2, nn_rect.centery - msg.get_height() // 2))
    else:
        # ----------------------------------------------------------
        # LEFT CARD: Neural network visualization
        # ----------------------------------------------------------
        positions, input_keys, output_keys = _compute_node_positions(genome, neat_config, nn_rect)

        # Get activation values from the network (filled after network.activate())
        values = getattr(network, "values", {})

        # --- Draw connections first (behind the nodes) ---
        for conn_key, conn_gene in genome.connections.items():
            if not conn_gene.enabled:
                continue
            in_key, out_key = conn_key
            if in_key not in positions or out_key not in positions:
                continue

            weight = conn_gene.weight
            color = COLOR_POSITIVE_WEIGHT if weight > 0 else COLOR_NEGATIVE_WEIGHT
            thickness = max(1, min(3, int(abs(weight) * 1.5)))

            pygame.draw.line(screen, color,
                             positions[in_key], positions[out_key], thickness)

        # --- Draw nodes (circles) ---
        node_radius = 8
        for node_key, (px, py) in positions.items():
            val = values.get(node_key, 0.0)
            fill_color = _activation_color(val)
            pygame.draw.circle(screen, fill_color, (px, py), node_radius)
            pygame.draw.circle(screen, COLOR_NODE_RING, (px, py), node_radius, 1)

        # --- Draw input labels (left of input nodes) ---
        for i, key in enumerate(input_keys):
            if key in positions:
                px, py = positions[key]
                label = font_sm.render(INPUT_LABELS[i], True, COLOR_TEXT_DIM)
                screen.blit(label, (px - label.get_width() - 12, py - label.get_height() // 2))

        # --- Draw output labels + live values (right of output nodes) ---
        for i, key in enumerate(output_keys):
            if key in positions:
                px, py = positions[key]
                val = values.get(key, 0.0)
                label_text = f"{OUTPUT_LABELS[i]} {val:.2f}"
                color = (255, 235, 100) if val > 0.5 else COLOR_TEXT_DIM
                label = font_sm.render(label_text, True, color)
                screen.blit(label, (px + 14, py - label.get_height() // 2))

    # ----------------------------------------------------------
    # RIGHT CARD: 2-Column Dashboard Stats
    # ----------------------------------------------------------
    col1_x = stats_rect.x + 20
    col2_x = stats_rect.x + int(stats_rect.width * 0.50)
    row_y = stats_rect.y + 44
    row_h = 32

    alive_num = stats.get("alive", 0)
    alive_color = (80, 225, 120) if alive_num > 0 else (240, 80, 80)
    steering_val = stats.get("steering", "Straight")
    steering_color = (240, 220, 100) if steering_val != "Straight" else COLOR_TEXT
    action_val = stats.get("action", "Coasting")
    if "Accel" in action_val:
        action_color = (80, 220, 120)
    elif "Brake" in action_val:
        action_color = (240, 80, 80)
    else:
        action_color = COLOR_TEXT

    col1_data = [
        ("Generation:", f"{stats.get('generation', 0)}", (240, 245, 255)),
        ("Cars Alive:", f"{alive_num} / {stats.get('total', 0)}", alive_color),
        ("Best Fitness:", f"{stats.get('best_fitness', 0):,.0f}", (255, 255, 255)),
        ("All-Time Best:", f"{stats.get('best_ever', 0):,.0f}", (255, 215, 60)),
        ("Best Speed:", f"{stats.get('speed', 0):.1f} px/f", COLOR_TEXT),
    ]

    col2_data = [
        ("Sim Speed:", f"{stats.get('sim_speed', 1)}x  ({stats.get('fps', 0):.0f} FPS)", COLOR_TEXT),
        ("Topology:", f"{stats.get('nodes', 0)} nodes, {stats.get('connections', 0)} conns", COLOR_TEXT),
        ("Species:", f"{stats.get('species', 1)}", COLOR_TEXT),
        ("Steering:", f"{steering_val}", steering_color),
        ("Action:", f"{action_val}", action_color),
    ]

    for r in range(5):
        cy = row_y + r * row_h
        # Column 1
        l1, v1, c1 = col1_data[r]
        lbl1 = font_sm.render(l1, True, COLOR_TEXT_DIM)
        val1 = font_val.render(v1, True, c1)
        screen.blit(lbl1, (col1_x, cy))
        screen.blit(val1, (col1_x + 125, cy))

        # Column 2
        l2, v2, c2 = col2_data[r]
        lbl2 = font_sm.render(l2, True, COLOR_TEXT_DIM)
        val2 = font_val.render(v2, True, c2)
        screen.blit(lbl2, (col2_x, cy))
        screen.blit(val2, (col2_x + 105, cy))


# ==============================================================
# Small overlay text on the road view (bottom-left of the sim area)
# ==============================================================
def draw_road_overlay(screen, stats):
    font = _get_font(20)
    gen = stats.get("generation", 0)
    alive = stats.get("alive", 0)
    total = stats.get("total", 0)
    best = stats.get("best_fitness", 0)
    spd = stats.get("speed", 0)

    text = f"Gen {gen}  |  Alive {alive}/{total}  |  Best {best:.0f}  |  Speed {spd:.1f}"
    surf = font.render(text, True, (255, 255, 255))

    # Semi-transparent background behind the text
    pad = 6
    bg = pygame.Surface((surf.get_width() + pad * 2, surf.get_height() + pad * 2), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 140))
    screen.blit(bg, (8, TOP_PANEL_HEIGHT + 8))
    screen.blit(surf, (8 + pad, TOP_PANEL_HEIGHT + 8 + pad))
