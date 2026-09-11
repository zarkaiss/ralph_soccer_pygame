"""Pygame Soccer -- entity definitions (Player and Ball).

This module provides the two core gameplay entities:

* :class:`Ball` -- a circle that moves with a velocity vector, bounces
  off the four pitch edges, and applies a small per-frame damping factor.
* :class:`Player` -- a circle driven by a normalised keyboard direction
  vector at a fixed speed, clamped (not bounced) to the pitch bounds.

Both classes subclass ``pygame.sprite.Sprite`` so they can be added to a
``pygame.sprite.Group`` later, and both expose the uniform
``update(dt)`` / ``draw(surface)`` interface expected by
``main.Game``.

The module is **headless-safe**: importing it does NOT require a display
surface.  ``pygame.init()`` is *not* called here; the host application
(``main.py``) is responsible for initialising pygame before any
``Surface``-dependent work happens.

All geometry / colour / entity constants are imported from :mod:`config`
so that the pitch bounds stay in a single source of truth and no circular
dependency exists between :mod:`main` and :mod:`entities`.
"""

from __future__ import annotations

import pygame
from pygame.math import Vector2

# ---------------------------------------------------------------------------
# Shared constants (single source of truth in config)
# ---------------------------------------------------------------------------
from config import (
    BALL_COLOR,
    BALL_DAMPING,
    BALL_RADIUS,
    FIELD_MARGIN,
    GOAL_BOX_HEIGHT,
    PLAYER_COLOR_P1,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

# ---------------------------------------------------------------------------
# Pitch bounds helper
# ---------------------------------------------------------------------------

def _pitch_bounds() -> tuple[float, float, float, float]:
    """Return the playable pitch bounds as ``(left, top, right, bottom)``.

    The bounds are inset by ``FIELD_MARGIN`` from the window edges,
    matching the field drawn by ``main.draw_field``.
    """
    left: float = float(FIELD_MARGIN)
    top: float = float(FIELD_MARGIN)
    right: float = float(WINDOW_WIDTH - FIELD_MARGIN)
    bottom: float = float(WINDOW_HEIGHT - FIELD_MARGIN)
    return left, top, right, bottom


def _pitch_center() -> Vector2:
    """Return the centre of the playable pitch as a ``Vector2``."""
    return Vector2(WINDOW_WIDTH / 2.0, WINDOW_HEIGHT / 2.0)


# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------

class Ball(pygame.sprite.Sprite):
    """A bouncing ball that moves with a velocity vector.

    Attributes
    ----------
    position:
        Current centre of the ball in window coordinates.
    velocity:
        Current velocity in pixels/second.
    radius:
        Radius of the ball in pixels.
    color:
        Fill colour for rendering.
    damping:
        Per-frame velocity retention factor (``1.0`` = no loss,
        ``0.999`` = 0.1% loss per frame).  Applied multiplicatively in
        :meth:`update` to simulate rolling friction / air drag.
    """

    def __init__(
        self,
        radius: int = BALL_RADIUS,
        color: tuple[int, int, int] = BALL_COLOR,
        damping: float = BALL_DAMPING,
    ) -> None:
        super().__init__()
        self.radius: int = radius
        self.color: tuple[int, int, int] = color
        self.damping: float = damping
        self.position: Vector2 = _pitch_center().copy()
        self.velocity: Vector2 = Vector2(0.0, 0.0)

    # -- Per-frame hooks ----------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance the ball by *dt* seconds.

        Steps:
        1. Integrate position: ``position += velocity * dt``.
        2. Apply per-frame damping: ``velocity *= damping``.
        3. Clamp to the pitch bounds, inverting the relevant velocity
           component on collision (elastic bounce).  The left/right wall
           bounces are skipped while the ball is inside the goal mouth
           (``WINDOW_HEIGHT/2 +/- GOAL_BOX_HEIGHT/2``) so the ball can
           cross the goal line; top/bottom bounces always apply.

        Parameters
        ----------
        dt:
            Elapsed time in seconds since the previous frame.
        """
        # 1. Integrate
        self.position += self.velocity * dt

        # 2. Damping (simple per-frame friction model)
        self.velocity *= self.damping

        # 3. Boundary clamping + bounce
        left, top, right, bottom = _pitch_bounds()

        # Left/right walls -- SKIPPED when the ball is inside the goal mouth
        # (the vertical span WINDOW_HEIGHT/2 +/- GOAL_BOX_HEIGHT/2) so the
        # ball can cross the goal line and be detected as a goal.
        goal_mouth_top = WINDOW_HEIGHT / 2 - GOAL_BOX_HEIGHT / 2
        goal_mouth_bottom = WINDOW_HEIGHT / 2 + GOAL_BOX_HEIGHT / 2
        in_goal_mouth = goal_mouth_top <= self.position.y <= goal_mouth_bottom

        if not in_goal_mouth:
            # Left wall
            if self.position.x - self.radius < left:
                self.position.x = left + self.radius
                self.velocity.x = abs(self.velocity.x)
            # Right wall
            elif self.position.x + self.radius > right:
                self.position.x = right - self.radius
                self.velocity.x = -abs(self.velocity.x)

        # Top wall
        if self.position.y - self.radius < top:
            self.position.y = top + self.radius
            self.velocity.y = abs(self.velocity.y)
        # Bottom wall
        elif self.position.y + self.radius > bottom:
            self.position.y = bottom - self.radius
            self.velocity.y = -abs(self.velocity.y)

    def draw(self, surface: pygame.Surface) -> None:
        """Render the ball as a filled circle on *surface*."""
        pygame.draw.circle(
            surface,
            self.color,
            (int(self.position.x), int(self.position.y)),
            self.radius,
        )

    # -- State management ---------------------------------------------------

    def reset(self) -> None:
        """Recenter the ball on the pitch and zero its velocity."""
        self.position = _pitch_center().copy()
        self.velocity = Vector2(0.0, 0.0)

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid
        return (
            f"Ball(position={self.position}, velocity={self.velocity}, "
            f"radius={self.radius})"
        )


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player(pygame.sprite.Sprite):
    """A keyboard-driven player circle.

    The player moves at a fixed ``speed`` (pixels/second) in the
    direction set by :meth:`set_direction`.  Position is clamped to the
    pitch bounds (no bouncing) -- the out-of-bounds velocity component
    is zeroed on contact.

    Attributes
    ----------
    position:
        Current centre of the player in window coordinates.
    velocity:
        Current velocity in pixels/second (derived from direction * speed).
    radius:
        Radius of the player in pixels.
    color:
        Fill colour for rendering.
    speed:
        Movement speed in pixels/second.
    direction:
        Normalised input direction vector (magnitude <= 1).
    """

    def __init__(
        self,
        radius: int = PLAYER_RADIUS,
        color: tuple[int, int, int] = PLAYER_COLOR_P1,
        speed: float = PLAYER_SPEED,
        start_position: Vector2 | None = None,
    ) -> None:
        super().__init__()
        self.radius: int = radius
        self.color: tuple[int, int, int] = color
        self.speed: float = speed
        self.direction: Vector2 = Vector2(0.0, 0.0)
        self.velocity: Vector2 = Vector2(0.0, 0.0)
        self.position: Vector2 = (
            start_position.copy() if start_position is not None else _pitch_center().copy()
        )

    # -- Input --------------------------------------------------------------

    def set_direction(self, dx: float, dy: float) -> None:
        """Store a normalised movement direction from keyboard input.

        Parameters
        ----------
        dx, dy:
            Raw input components (typically -1, 0, or +1).  The vector
            is normalised so that diagonal movement is not faster than
            axis-aligned movement.  Passing ``(0, 0)`` stops the player.
        """
        v = Vector2(dx, dy)
        if v.length_squared() > 0.0:
            v = v.normalize()
        self.direction = v

    # -- Per-frame hooks ----------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance the player by *dt* seconds.

        Steps:
        1. Derive velocity from the stored direction and fixed speed.
        2. Integrate position: ``position += velocity * dt``.
        3. Clamp to the pitch bounds, zeroing the out-of-bounds
           velocity component (no bounce).

        Parameters
        ----------
        dt:
            Elapsed time in seconds since the previous frame.
        """
        # 1. Velocity from direction
        self.velocity = self.direction * self.speed

        # 2. Integrate
        self.position += self.velocity * dt

        # 3. Boundary clamping (no bounce)
        left, top, right, bottom = _pitch_bounds()

        if self.position.x - self.radius < left:
            self.position.x = left + self.radius
            self.velocity.x = 0.0
        elif self.position.x + self.radius > right:
            self.position.x = right - self.radius
            self.velocity.x = 0.0

        if self.position.y - self.radius < top:
            self.position.y = top + self.radius
            self.velocity.y = 0.0
        elif self.position.y + self.radius > bottom:
            self.position.y = bottom - self.radius
            self.velocity.y = 0.0

    def draw(self, surface: pygame.Surface) -> None:
        """Render the player as a filled circle on *surface*."""
        pygame.draw.circle(
            surface,
            self.color,
            (int(self.position.x), int(self.position.y)),
            self.radius,
        )

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid
        return (
            f"Player(position={self.position}, velocity={self.velocity}, "
            f"radius={self.radius}, color={self.color})"
        )
