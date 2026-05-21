import math

import numpy as np
import pygame

from .game import Directions
from .graphics_constants import *

SCORE_ON_TOP = False
SCALE_BAR = True


def add(x, y):
    return (x[0] + y[0], x[1] + y[1])


def line(surface, here, there, color=(0, 0, 0), width=2):
    x0, y0 = here
    x1, y1 = there
    pygame.draw.line(surface, color, (x0, y0), (x1, y1), width)
    return pygame.Rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))


def circle(surface, pos, r, outlineColor, fillColor, endpoints=None, style="pieslice", width=2):
    x, y = pos
    if endpoints is None:
        e0, e1 = 0, 359
    else:
        e0, e1 = endpoints
        while e0 > e1:
            e1 += 360

    arc_rect = pygame.Rect(x - r, y - r, 2 * r, 2 * r)
    bounding_rect = pygame.Rect(x - r - 1, y - r - 1, 2 * r + 2, 2 * r + 2)
    # Convert angles to radians
    e0_rad = math.radians(e0)
    e1_rad = math.radians(e1)

    # Handle full circle
    if e1 - e0 >= 360 or style == "arc":
        if fillColor:
            pygame.draw.circle(surface, fillColor, (x, y), r)
        if outlineColor and width > 0:
            pygame.draw.circle(surface, outlineColor, (x, y), r, width)
        return bounding_rect

    # Draw filled pie slice
    if fillColor:
        points = [(x, y)]
        for angle in range(e0, e1 + 1):
            rad = math.radians(angle)
            px = x + r * math.cos(rad)
            py = y + r * math.sin(rad)
            points.append((px, py))
        pygame.draw.polygon(surface, fillColor, points)

    # Draw arc outline
    if outlineColor and width > 0:
        pygame.draw.arc(surface, outlineColor, arc_rect, e0_rad, e1_rad, width)
    return bounding_rect


def polygon(surface, coords, outlineColor, fillColor=None, filled=1, smoothed=1, behind=0, width=1):
    if fillColor is None:
        fillColor = outlineColor
    if filled == 0:
        fillColor = None  # no fill

    # Draw filled polygon first (underneath)
    if fillColor is not None:
        pygame.draw.polygon(surface, fillColor, coords)

    # Draw outline on top
    if outlineColor is not None and width > 0:
        pygame.draw.polygon(surface, outlineColor, coords, width)
    # Compute bounding box
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    bbox = pygame.Rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    return bbox


class PacmanGraphicsPyGame:
    def __init__(self, zoom=1.0, frameTime=0.0, capture=False, show_window=False):
        self.have_window = 0
        self.zoom = zoom
        self.gridSize = DEFAULT_GRID_SIZE * zoom
        self.capture = capture
        self.frameTime = frameTime
        self.image = None
        self.show_window = show_window
        self.window = None

    def checkNullDisplay(self):
        return self.image is None

    def draw_all(self, state, isBlue=False):
        self.isBlue = isBlue

        # self.drawDistributions(state)
        # self.distributionImages = None  # Initialized lazily
        self.drawStaticObjects(state)
        self.drawAgentObjects(state)
        score = state.score
        self.drawScore(score)
        # Information
        # self.previousState = state

    def initialize(self, state, isBlue=False):
        self.finish()  # finish if necessary (there is a check for if there is a window
        self.startGraphics(state)
        self.draw_all(state, isBlue)

    def startGraphics(self, state):
        # self.layout = state.layout
        layout = state.layout
        self.width = layout.width
        self.height = layout.height
        self.screen_width, self.screen_height = self.make_window(self.width, self.height)
        # removing infoPane -no
        # self.infoPane = InfoPane(layout, self.gridSize, utils=self.graphicsUtils)
        # self.currentState = layout

    def drawStaticObjects(self, state):
        layout = state.layout
        self.drawWalls(layout.walls)
        self.drawFood(state.food)
        self.drawCapsules(state.capsules)

    def drawAgentObjects(self, state):
        # self.agentImages = []  # (agentState, image)
        for index, agent in enumerate(state.agentStates):
            if agent.isPacman:
                rect = self.drawPacman(agent, index)
                # self.agentImages.append((agent, rect))
            else:
                rect = self.drawGhost(agent, index)
                # self.agentImages.append((agent, rect))

    def refresh_py_game(self):
        pygame.event.pump()
        pygame.display.update()

    def update(self, newState):
        self.window.fill(BACKGROUND_COLOR)
        self.draw_all(newState)  # drawing everything is just the easiest and makes hardly a difference
        # agentIndex = newState._agentMoved
        # agentState = newState.agentStates[agentIndex]
        # #
        # # if self.agentImages[agentIndex][0].isPacman != agentState.isPacman: self.swapImages(agentIndex, agentState)
        # prevState, prevImage = self.agentImages[agentIndex]
        # if agentState.isPacman:
        #     prevImage = self.animatePacman(agentState, agentIndex,prevState, prevImage)
        # else:
        #     pass
        #     # self.moveGhost(agentState, agentIndex, prevState, prevImage)
        # self.agentImages[agentIndex] = (agentState, prevImage)

    def animatePacman(self, pacman, agentIndex, prevPacman, image):
        px, py = self.getPosition(pacman)
        pos = px, py
        self.window.fill(BACKGROUND_COLOR, image)
        return self.drawPacman(pacman, agentIndex)

    def updateView(self):
        # when all updates are done, update image repr
        # self.image = self.graphicsUtils.image()
        self.image = np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def calculate_screen_dimensions(self, width, height):
        grid_width = (width - 1) * self.gridSize
        grid_height = (height - 1) * self.gridSize
        screen_width = 2 * self.gridSize + grid_width
        screen_height = 2 * self.gridSize + grid_height
        return screen_width, screen_height

    def make_window(self, width, height):
        screen_width, screen_height = self.calculate_screen_dimensions(width, height)
        # self.graphicsUtils.begin_graphics(screen_width,
        #                                   screen_height,
        #                                   BACKGROUND_COLOR,
        #                                   "Pacman Gym", self.show_window)
        pygame.init()
        pygame.display.set_caption("PAC-Man")
        if self.show_window:
            self.window = pygame.display.set_mode((screen_width, screen_height))
        else:
            self.window = pygame.Surface((screen_width, screen_height))
        return screen_width, screen_height

    def drawPacman(self, pacman, index):
        position = self.getPosition(pacman)
        screen_point = self.to_screen(position)
        endpoints_x, endpoints_y = self.getEndpoints(self.getDirection(pacman))
        endpoints = int(endpoints_x), int(endpoints_y)

        width = PACMAN_OUTLINE_WIDTH
        outlineColor = PACMAN_COLOR
        fillColor = PACMAN_COLOR

        if self.capture:
            outlineColor = TEAM_COLORS[index % 2]
            fillColor = GHOST_COLORS[index]
            width = PACMAN_CAPTURE_OUTLINE_WIDTH

        return circle(
            self.window,
            screen_point,
            PACMAN_SCALE * self.gridSize,
            fillColor=fillColor,
            outlineColor=outlineColor,
            endpoints=endpoints,
            width=width,
        )

    def getEndpoints(self, direction, position=(0, 0)):
        x, y = position
        pos = x - int(x) + y - int(y)
        width = 30 + 80 * math.sin(math.pi * pos)

        delta = width / 2
        if direction == Directions.WEST:
            endpoints = (180 + delta, 180 - delta)
        elif direction == Directions.NORTH:
            endpoints = (90 + delta, 90 - delta)
        elif direction == Directions.SOUTH:
            endpoints = (270 + delta, 270 - delta)
        else:
            endpoints = (0 + delta, 0 - delta)
        return endpoints

    def getGhostColor(self, ghost, ghostIndex):
        if ghost.scaredTimer > 0:
            if GHOST_BLINK_TIME > 0:
                if ghost.scaredTimer <= GHOST_BLINK_TIME:
                    if ghost.scaredTimer % 2 == 1:
                        return SCARED_COLORS[ghostIndex]
                    else:
                        return GHOST_COLORS[ghostIndex]

                else:
                    return SCARED_COLORS[ghostIndex]
            else:
                return SCARED_COLORS[ghostIndex]
            return SCARED_COLORS[ghostIndex]
        else:
            return GHOST_COLORS[ghostIndex]

    def drawGhost(self, ghost, agentIndex):
        pos = self.getPosition(ghost)
        dir = self.getDirection(ghost)
        (screen_x, screen_y) = self.to_screen(pos)
        coords = []
        for x, y in GHOST_SHAPE:
            coords.append((x * self.gridSize * GHOST_SIZE + screen_x, y * self.gridSize * GHOST_SIZE + screen_y))

        colour = self.getGhostColor(ghost, agentIndex)
        rect = polygon(self.window, coords, colour, fillColor=colour)

        WHITE = formatColor(1.0, 1.0, 1.0)
        BLACK = formatColor(0.0, 0.0, 0.0)

        dx = 0
        dy = 0
        if dir == "North":
            dy = -0.2
        if dir == "South":
            dy = 0.2
        if dir == "East":
            dx = 0.2
        if dir == "West":
            dx = -0.2
        circle(
            self.window,
            (
                screen_x + self.gridSize * GHOST_SIZE * (-0.3 + dx / 1.5),
                screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy / 1.5),
            ),
            self.gridSize * GHOST_SIZE * 0.2,
            WHITE,
            WHITE,
        )
        circle(
            self.window,
            (
                screen_x + self.gridSize * GHOST_SIZE * (0.3 + dx / 1.5),
                screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy / 1.5),
            ),
            self.gridSize * GHOST_SIZE * 0.2,
            WHITE,
            WHITE,
        )
        circle(
            self.window,
            (screen_x + self.gridSize * GHOST_SIZE * (-0.3 + dx), screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy)),
            self.gridSize * GHOST_SIZE * 0.08,
            BLACK,
            BLACK,
        )
        circle(
            self.window,
            (screen_x + self.gridSize * GHOST_SIZE * (0.3 + dx), screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy)),
            self.gridSize * GHOST_SIZE * 0.08,
            BLACK,
            BLACK,
        )
        return rect

    def getPosition(self, agentState):
        if agentState.configuration == None:
            return (-1000, -1000)
        return agentState.getPosition()

    def getDirection(self, agentState):
        if agentState.configuration == None:
            return Directions.STOP
        return agentState.configuration.getDirection()

    def finish(self):
        if self.window is not None and self.show_window:
            pygame.display.quit()
            pygame.quit()
        self.image = None
        self.window = None

    def to_screen(self, point):
        (x, y) = point
        # y = self.height - y
        x = (x + 1) * self.gridSize
        y = (self.height - y) * self.gridSize
        return (x, y)

    # Fixes some TK issue with off-center circles
    def to_screen2(self, point):
        (x, y) = point
        # y = self.height - y
        x = (x + 1) * self.gridSize
        y = (self.height - y) * self.gridSize
        return (x, y)

    def drawWalls(self, wallMatrix):
        wallColor = WALL_COLOR
        for xNum, x in enumerate(wallMatrix):
            if self.capture and (xNum * 2) < wallMatrix.width:
                wallColor = TEAM_COLORS[0]
            if self.capture and (xNum * 2) >= wallMatrix.width:
                wallColor = TEAM_COLORS[1]

            for yNum, cell in enumerate(x):
                if cell:  # There's a wall here
                    pos = (xNum, yNum)
                    screen = self.to_screen(pos)
                    screen2 = self.to_screen2(pos)

                    # draw each quadrant of the square based on adjacent walls
                    wIsWall = self.isWall(xNum - 1, yNum, wallMatrix)
                    eIsWall = self.isWall(xNum + 1, yNum, wallMatrix)
                    nIsWall = self.isWall(xNum, yNum + 1, wallMatrix)
                    sIsWall = self.isWall(xNum, yNum - 1, wallMatrix)
                    nwIsWall = self.isWall(xNum - 1, yNum + 1, wallMatrix)
                    swIsWall = self.isWall(xNum - 1, yNum - 1, wallMatrix)
                    neIsWall = self.isWall(xNum + 1, yNum + 1, wallMatrix)
                    seIsWall = self.isWall(xNum + 1, yNum - 1, wallMatrix)

                    # NE quadrant
                    if (not nIsWall) and (not eIsWall):
                        # inner circle
                        # self.graphicsUtils.circle(screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor, (0, 91),
                        #                           'arc', width=6)
                        circle(
                            surface=self.window,
                            pos=screen2,  # <- a tuple (e.g., (100, 100))
                            r=WALL_RADIUS * self.gridSize,
                            outlineColor=wallColor,
                            fillColor=wallColor,  # <- no fill
                            endpoints=(0, 91),
                            style="arc",
                            width=6,
                        )

                    if (nIsWall) and (not eIsWall):
                        # vertical line
                        # self.graphicsUtils.line(add(screen, (self.gridSize * WALL_RADIUS, 0)),
                        #                         add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (-0.5) - 1)),
                        #                         wallColor, width=6)
                        line(
                            self.window,
                            add(screen, (self.gridSize * WALL_RADIUS, 0)),
                            add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (-0.5) - 1)),
                            color=wallColor,
                            width=6,
                        )
                    if (not nIsWall) and (eIsWall):
                        # horizontal line
                        # self.graphicsUtils.line(add(screen, (0, self.gridSize * (-1) * WALL_RADIUS)), add(screen,
                        #                                                                                   (self.gridSize * 0.5 + 1,
                        #                                                                                    self.gridSize * (
                        #                                                                                        -1) * WALL_RADIUS)),
                        #                         wallColor, width=6)
                        line(
                            self.window,
                            add(screen, (0, self.gridSize * (-1) * WALL_RADIUS)),
                            add(screen, (self.gridSize * 0.5 + 1, self.gridSize * (-1) * WALL_RADIUS)),
                            color=wallColor,
                            width=6,
                        )
                    if (nIsWall) and (eIsWall) and (not neIsWall):
                        # outer circle
                        # self.graphicsUtils.circle(
                        #     add(screen2, (self.gridSize * 2 * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS)),
                        #     WALL_RADIUS * self.gridSize - 1, wallColor, wallColor, (180, 271), 'arc', width=6)
                        circle(
                            surface=self.window,
                            pos=add(
                                screen2, (self.gridSize * 2 * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS)
                            ),  # <- a tuple (e.g., (100, 100))
                            r=WALL_RADIUS * self.gridSize - 1,
                            outlineColor=wallColor,
                            fillColor=wallColor,  # <- no fill
                            endpoints=(180, 271),
                            style="arc",
                            width=6,
                        )

                        # self.graphicsUtils.line(
                        #     add(screen, (self.gridSize * 2 * WALL_RADIUS - 1, self.gridSize * (-1) * WALL_RADIUS)),
                        #     add(screen, (self.gridSize * 0.5 + 1, self.gridSize * (-1) * WALL_RADIUS)), wallColor,
                        #     width=6)
                        line(
                            self.window,
                            add(screen, (self.gridSize * 2 * WALL_RADIUS - 1, self.gridSize * (-1) * WALL_RADIUS)),
                            add(screen, (self.gridSize * 0.5 + 1, self.gridSize * (-1) * WALL_RADIUS)),
                            color=wallColor,
                            width=6,
                        )
                        # self.graphicsUtils.line(
                        #     add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS + 1)),
                        #     add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (-0.5))), wallColor, width=6)
                        line(
                            self.window,
                            add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS + 1)),
                            add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (-0.5))),
                            color=wallColor,
                            width=6,
                        )
                    # NW quadrant
                    if (not nIsWall) and (not wIsWall):
                        # inner circle
                        # self.graphicsUtils.circle(screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor, (90, 181),
                        #                           'arc', width=6)
                        circle(
                            surface=self.window,
                            pos=screen2,
                            r=WALL_RADIUS * self.gridSize,
                            outlineColor=wallColor,
                            fillColor=wallColor,  # <- no fill
                            endpoints=(90, 181),
                            style="arc",
                            width=6,
                        )
                    if (nIsWall) and (not wIsWall):
                        # vertical line
                        # self.graphicsUtils.line(add(screen, (self.gridSize * (-1) * WALL_RADIUS, 0)), add(screen,
                        #                                                                                   (self.gridSize * (
                        #                                                                                       -1) * WALL_RADIUS,
                        #                                                                                    self.gridSize * (
                        #                                                                                        -0.5) - 1)),
                        #                         wallColor, width=6)

                        line(
                            self.window,
                            add(screen, (self.gridSize * (-1) * WALL_RADIUS, 0)),
                            add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (-0.5) - 1)),
                            color=wallColor,
                            width=6,
                        )
                    if (not nIsWall) and (wIsWall):
                        # horizontal line
                        # self.graphicsUtils.line(add(screen, (0, self.gridSize * (-1) * WALL_RADIUS)), add(screen,
                        #                                                                                   (self.gridSize * (
                        #                                                                                       -0.5) - 1,
                        #                                                                                    self.gridSize * (
                        #                                                                                        -1) * WALL_RADIUS)),
                        #                         wallColor, width=6)
                        line(
                            self.window,
                            add(screen, (0, self.gridSize * (-1) * WALL_RADIUS)),
                            add(screen, (self.gridSize * (-0.5) - 1, self.gridSize * (-1) * WALL_RADIUS)),
                            color=wallColor,
                            width=6,
                        )
                    if (nIsWall) and (wIsWall) and (not nwIsWall):
                        # outer circle
                        # self.graphicsUtils.circle(
                        #     add(screen2, (self.gridSize * (-2) * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS)),
                        #     WALL_RADIUS * self.gridSize - 1, wallColor, wallColor, (270, 361), 'arc', width=6)
                        circle(
                            surface=self.window,
                            pos=add(screen2, (self.gridSize * (-2) * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS)),
                            r=WALL_RADIUS * self.gridSize - 1,
                            outlineColor=wallColor,
                            fillColor=wallColor,  # <- no fill
                            endpoints=(270, 361),
                            style="arc",
                            width=6,
                        )

                        # self.graphicsUtils.line(
                        #     add(screen, (self.gridSize * (-2) * WALL_RADIUS + 1, self.gridSize * (-1) * WALL_RADIUS)),
                        #     add(screen, (self.gridSize * (-0.5), self.gridSize * (-1) * WALL_RADIUS)), wallColor,
                        #     width=6)
                        line(
                            self.window,
                            add(screen, (self.gridSize * (-2) * WALL_RADIUS + 1, self.gridSize * (-1) * WALL_RADIUS)),
                            add(screen, (self.gridSize * (-0.5), self.gridSize * (-1) * WALL_RADIUS)),
                            color=wallColor,
                            width=6,
                        )
                        # self.graphicsUtils.line(
                        #     add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS + 1)),
                        #     add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (-0.5))), wallColor,
                        #     width=6)
                        line(
                            self.window,
                            add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS + 1)),
                            add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (-0.5))),
                            color=wallColor,
                            width=6,
                        )

                    # SE quadrant
                    if (not sIsWall) and (not eIsWall):
                        # inner circle
                        # self.graphicsUtils.circle(screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor,
                        #                           (270, 361), 'arc', width=6)
                        circle(
                            surface=self.window,
                            pos=screen2,
                            r=WALL_RADIUS * self.gridSize,
                            outlineColor=wallColor,
                            fillColor=wallColor,  # <- no fill
                            endpoints=(270, 361),
                            style="arc",
                            width=6,
                        )
                    if (sIsWall) and (not eIsWall):
                        # vertical line
                        # self.graphicsUtils.line(add(screen, (self.gridSize * WALL_RADIUS, 0)),
                        #                         add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (0.5) + 1)),
                        #                         wallColor, width=6)

                        line(
                            self.window,
                            add(screen, (self.gridSize * WALL_RADIUS, 0)),
                            add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (0.5) + 1)),
                            color=wallColor,
                            width=6,
                        )
                    if (not sIsWall) and (eIsWall):
                        # horizontal line
                        # self.graphicsUtils.line(add(screen, (0, self.gridSize * (1) * WALL_RADIUS)), add(screen,
                        #                                                                                  (self.gridSize * 0.5 + 1,
                        #                                                                                   self.gridSize * (
                        #                                                                                       1) * WALL_RADIUS)),
                        #                         wallColor, width=6)
                        line(
                            self.window,
                            add(screen, (0, self.gridSize * (1) * WALL_RADIUS)),
                            add(screen, (self.gridSize * 0.5 + 1, self.gridSize * (1) * WALL_RADIUS)),
                            wallColor,
                            width=6,
                        )
                    if (sIsWall) and (eIsWall) and (not seIsWall):
                        # outer circle
                        # self.graphicsUtils.circle(
                        #     add(screen2, (self.gridSize * 2 * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS)),
                        #     WALL_RADIUS * self.gridSize - 1, wallColor, wallColor, (90, 181), 'arc', width=6)
                        circle(
                            surface=self.window,
                            pos=add(screen2, (self.gridSize * 2 * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS)),
                            r=WALL_RADIUS * self.gridSize - 1,
                            outlineColor=wallColor,
                            fillColor=wallColor,  # <- no fill
                            endpoints=(90, 181),
                            style="arc",
                            width=6,
                        )
                        # self.graphicsUtils.line(
                        #     add(screen, (self.gridSize * 2 * WALL_RADIUS - 1, self.gridSize * (1) * WALL_RADIUS)),
                        #     add(screen, (self.gridSize * 0.5, self.gridSize * (1) * WALL_RADIUS)), wallColor, width=6)
                        line(
                            self.window,
                            add(screen, (self.gridSize * 2 * WALL_RADIUS - 1, self.gridSize * (1) * WALL_RADIUS)),
                            add(screen, (self.gridSize * 0.5, self.gridSize * (1) * WALL_RADIUS)),
                            wallColor,
                            width=6,
                        )
                        # self.graphicsUtils.line(
                        #     add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS - 1)),
                        #     add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (0.5))), wallColor, width=6)
                        line(
                            self.window,
                            add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS - 1)),
                            add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (0.5))),
                            wallColor,
                            width=6,
                        )

                    # SW quadrant
                    if (not sIsWall) and (not wIsWall):
                        # inner circle
                        # self.graphicsUtils.circle(screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor,
                        #                           (180, 271), 'arc', width=6)
                        circle(
                            surface=self.window,
                            pos=screen2,
                            r=WALL_RADIUS * self.gridSize,
                            outlineColor=wallColor,
                            fillColor=wallColor,  # <- no fill
                            endpoints=(180, 271),
                            style="arc",
                            width=6,
                        )
                    if (sIsWall) and (not wIsWall):
                        # vertical line
                        # self.graphicsUtils.line(add(screen, (self.gridSize * (-1) * WALL_RADIUS, 0)), add(screen,
                        #                                                                                   (self.gridSize * (
                        #                                                                                       -1) * WALL_RADIUS,
                        #                                                                                    self.gridSize * (
                        #                                                                                        0.5) + 1)),
                        #                         wallColor, width=6)
                        line(
                            self.window,
                            add(screen, (self.gridSize * (-1) * WALL_RADIUS, 0)),
                            add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (0.5) + 1)),
                            wallColor,
                            width=6,
                        )
                    if (not sIsWall) and (wIsWall):
                        # horizontal line
                        # self.graphicsUtils.line(add(screen, (0, self.gridSize * (1) * WALL_RADIUS)), add(screen,
                        #                                                                                  (self.gridSize * (
                        #                                                                                      -0.5) - 1,
                        #                                                                                   self.gridSize * (
                        #                                                                                       1) * WALL_RADIUS)),
                        #                         wallColor, width=6)
                        line(
                            self.window,
                            add(screen, (0, self.gridSize * (1) * WALL_RADIUS)),
                            add(screen, (self.gridSize * (-0.5) - 1, self.gridSize * (1) * WALL_RADIUS)),
                            wallColor,
                            width=6,
                        )
                    if (sIsWall) and (wIsWall) and (not swIsWall):
                        # outer circle
                        # self.graphicsUtils.circle(
                        #     add(screen2, (self.gridSize * (-2) * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS)),
                        #     WALL_RADIUS * self.gridSize - 1, wallColor, wallColor, (0, 91), 'arc', width=6)
                        circle(
                            surface=self.window,
                            pos=add(screen2, (self.gridSize * (-2) * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS)),
                            r=WALL_RADIUS * self.gridSize - 1,
                            outlineColor=wallColor,
                            fillColor=wallColor,  # <- no fill
                            endpoints=(0, 91),
                            style="arc",
                            width=6,
                        )
                        # self.graphicsUtils.line(
                        #     add(screen, (self.gridSize * (-2) * WALL_RADIUS + 1, self.gridSize * (1) * WALL_RADIUS)),
                        #     add(screen, (self.gridSize * (-0.5), self.gridSize * (1) * WALL_RADIUS)), wallColor,
                        #     width=6)
                        line(
                            self.window,
                            add(screen, (self.gridSize * (-2) * WALL_RADIUS + 1, self.gridSize * (1) * WALL_RADIUS)),
                            add(screen, (self.gridSize * (-0.5), self.gridSize * (1) * WALL_RADIUS)),
                            wallColor,
                            width=6,
                        )
                        # self.graphicsUtils.line(
                        #     add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS - 1)),
                        #     add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (0.5))), wallColor,
                        #     width=6)
                        line(
                            self.window,
                            add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS - 1)),
                            add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (0.5))),
                            wallColor,
                            width=6,
                        )

    def isWall(self, x, y, walls):
        if x < 0 or y < 0:
            return False
        if x >= walls.width or y >= walls.height:
            return False
        return walls[x][y]

    def drawFood(self, foodMatrix):
        color = FOOD_COLOR

        for xNum, x in enumerate(foodMatrix):
            if self.capture and (xNum * 2) <= foodMatrix.width:
                color = TEAM_COLORS[0]
            if self.capture and (xNum * 2) > foodMatrix.width:
                color = TEAM_COLORS[1]
            for yNum, cell in enumerate(x):
                if cell:  # There's food here
                    screen_x, screen_y = self.to_screen((xNum, yNum))
                    coords = []
                    for x, y in FOOD_SHAPE:
                        coords.append(
                            (x * self.gridSize * GHOST_SIZE + screen_x, y * self.gridSize * GHOST_SIZE + screen_y)
                        )

                    rect = polygon(self.window, coords, color, fillColor=color)

                    # circle(self.window,(screen_x,screen_y),FOOD_SIZE * self.gridSize,
                    #                                 outlineColor=color, fillColor=color,
                    #                                 width=1)
                else:
                    pass

    def drawCapsules(self, capsules):
        for capsule in capsules:
            (screen_x, screen_y) = self.to_screen(capsule)
            circle(
                self.window,
                (screen_x, screen_y),
                CAPSULE_SIZE * self.gridSize,
                outlineColor=CAPSULE_COLOR,
                fillColor=CAPSULE_COLOR,
                width=1,
            )

    def drawScore(self, score):
        font = pygame.font.SysFont("Consolas", 36, bold=True)

        max_score = 1750

        bar_width_max = int(0.9 * self.screen_width)
        bar_height = 40
        bar_bg_color = (60, 60, 60)
        bar_fg_color = (0, 180, 0)
        text_color = SCORE_COLOR
        padding = 10

        if SCORE_ON_TOP:
            bar_x = SCORE_X
        else:
            score_text_max_right = font.render("-8888", True, text_color).get_rect().right
            bar_x = score_text_max_right + padding
        bar_y = self.height * self.gridSize

        if SCALE_BAR:
            scale_score = math.sqrt(max(score, 0))
            scale_max = math.sqrt(max_score)
            progress = min(max(scale_score / scale_max, 0), 1.0)
        else:
            progress = min(max(score / max_score, 0), 1.0)

        pygame.draw.rect(self.window, bar_bg_color, (bar_x, bar_y, bar_width_max, bar_height), border_radius=6)
        pygame.draw.rect(
            self.window, bar_fg_color, (bar_x, bar_y, int(bar_width_max * progress), bar_height), border_radius=6
        )
        if SCORE_ON_TOP:
            score_text = font.render(f"Score: {score}", True, text_color)
        else:
            score_text = font.render(f"{score}", True, text_color)
        text_rect = score_text.get_rect()
        text_rect.centery = bar_y + bar_height // 2
        text_rect.x = SCORE_X + padding

        self.window.blit(score_text, text_rect)
