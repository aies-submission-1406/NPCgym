def colorToVector(color):
    return map(lambda x: int(x, 16) / 256.0, [color[1:3], color[3:5], color[5:7]])


def formatColor(r, g, b):
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


DEFAULT_GRID_SIZE = 30.0
INFO_PANE_HEIGHT = 40
BACKGROUND_COLOR = formatColor(0, 0, 0)
WALL_COLOR = formatColor(142.0 / 255.0, 126.0 / 255.0, 247 / 255.0)
INFO_PANE_COLOR = formatColor(0.4, 0.4, 0)
SCORE_COLOR = formatColor(0.9, 0.9, 0.9)
PACMAN_OUTLINE_WIDTH = 2
PACMAN_CAPTURE_OUTLINE_WIDTH = 4
SCORE_X = 20
GHOST_BLINK_TIME = 6

GHOST_COLORS = []
GHOST_COLORS.append(formatColor(0.9, 0, 0))  # Red
GHOST_COLORS.append(formatColor(0, 0.3, 0.9))  # Blue
GHOST_COLORS.append(formatColor(0.98, 0.41, 0.07))  # Orange
# GHOST_COLORS.append(formatColor(0,.3,.9)) # Blue
# GHOST_COLORS.append(formatColor(.98,.41,.07)) # Orange
GHOST_COLORS.append(formatColor(0.1, 0.75, 0.7))  # Green
GHOST_COLORS.append(formatColor(1.0, 0.6, 0.0))  # Yellow
GHOST_COLORS.append(formatColor(0.4, 0.13, 0.91))  # Purple

TEAM_COLORS = GHOST_COLORS[:2]
FOOD_SHAPE = [(-0.2, 0.0), (0.6, 0.0), (0.6, 0.3), (-0.2, 0.3)]
GHOST_SHAPE = [
    (0, 0.3),
    (0.25, 0.75),
    (0.5, 0.3),
    (0.75, 0.75),
    (0.75, -0.5),
    (0.5, -0.75),
    (-0.5, -0.75),
    (-0.75, -0.5),
    (-0.75, 0.75),
    (-0.5, 0.3),
    (-0.25, 0.75),
]
GHOST_SIZE = 0.65
SCARED_COLOR = formatColor(1, 1, 1)

SCARED_COLORS = []
SCARED_COLORS.append(formatColor(0.9, 0.7, 0.7))  # Red
SCARED_COLORS.append(formatColor(1.0, 1.0, 1.0))  # Blue is just white
SCARED_COLORS.append(formatColor(0.95, 0.71, 0.48))  # Orange
# SCARED_COLORS.append(formatColor(0.6, 0.75, 1.0)) # Blue
# SCARED_COLORS.append(formatColor(1.0, 0.75, 0.5)) # Orange
SCARED_COLORS.append(formatColor(0.6, 1.0, 0.75))  # Green
SCARED_COLORS.append(formatColor(1.0, 1.0, 0.5))  # Yellow
SCARED_COLORS.append(formatColor(0.7, 0.6, 1.0))  # Purple

GHOST_VEC_COLORS = map(colorToVector, GHOST_COLORS)
SCARED_VEC_COLORS = map(colorToVector, GHOST_COLORS)

PACMAN_COLOR = formatColor(255.0 / 255.0, 255.0 / 255.0, 61.0 / 255)
PACMAN_SCALE = 0.5
# pacman_speed = 0.25

# Food
FOOD_COLOR = formatColor(1, 1, 1)
# FOOD_SIZE = 0.1
FOOD_SIZE = 0.15

# Laser
LASER_COLOR = formatColor(1, 0, 0)
LASER_SIZE = 0.02

# Capsule graphics
CAPSULE_COLOR = formatColor(1, 1, 1)
# CAPSULE_SIZE = 0.25
CAPSULE_SIZE = 0.3

# Drawing walls
WALL_RADIUS = 0.15
