"""Pygame Soccer -- shared configuration constants.

This module is the **single source of truth** for all read-only
configuration values (window geometry, colours, field layout, and
entity defaults).  Both :mod:`main` and :mod:`entities` import from here,
which eliminates the previous circular dependency between those two
modules (``entities`` used to import constants from ``main`` while
``main`` imported entity classes from ``entities``).

The module is dependency-free (no pygame import required) and therefore
safe to import in any context, including fresh headless interpreters.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

WINDOW_WIDTH: int = 800
WINDOW_HEIGHT: int = 600
WINDOW_TITLE: str = "Pygame Soccer"
FPS: int = 60

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

PITCH_GREEN: tuple[int, int, int] = (34, 139, 34)
LINE_WHITE: tuple[int, int, int] = (255, 255, 255)

# ---------------------------------------------------------------------------
# Field geometry (derived from window size)
# ---------------------------------------------------------------------------

FIELD_MARGIN: int = 40          # distance from window edge to pitch boundary
CENTER_CIRCLE_RADIUS: int = 60
GOAL_BOX_WIDTH: int = 120       # depth of each goal box from the goal line
GOAL_BOX_HEIGHT: int = 200      # width of each goal box along the goal line
LINE_WIDTH: int = 3

# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------

BALL_RADIUS: int = 10
BALL_COLOR: tuple[int, int, int] = (255, 255, 255)
BALL_DAMPING: float = 0.999     # per-frame velocity retention (1.0 = no loss)

# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

PLAYER_RADIUS: int = 15
PLAYER_SPEED: float = 300.0     # pixels per second
PLAYER_COLOR_P1: tuple[int, int, int] = (0, 0, 255)   # blue
PLAYER_COLOR_P2: tuple[int, int, int] = (255, 0, 0)   # red
