"""Pygame Soccer -- scoring and physics simulation.

This module provides two focused, dependency-light classes:

* :class:`Score` -- a tiny value object tracking the two players' scores,
  with a :meth:`draw` helper that renders ``"P1: X  P2: Y"`` centred at the
  top of the window.
* :class:`PhysicsEngine` -- per-frame simulation of ball/player collision
  (separation, velocity reflection, momentum transfer, speed clamping) and
  goal detection (left/right goal boxes), awarding points and resetting the
  ball on a goal.

The module is **headless-safe**: importing it does NOT require a display
surface.  It imports only from :mod:`config` and :mod:`entities`, so there
is no circular dependency with :mod:`main`.
"""

from __future__ import annotations

import pygame
from pygame.math import Vector2

from config import (
    FIELD_MARGIN,
    GOAL_BOX_HEIGHT,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from entities import Ball, Player

# ---------------------------------------------------------------------------
# Tunable physics constants (kept local to the physics domain)
# ---------------------------------------------------------------------------

BALL_MAX_SPEED: float = 800.0     # hard clamp on ball speed (px/s)
PLAYER_MOMENTUM_FACTOR: float = 0.5  # fraction of player velocity added on hit


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

class Score:
    """Track the two players' scores.

    Attributes
    ----------
    p1:
        Score for player 1 (the left / blue side).
    p2:
        Score for player 2 (the right / red side).
    """

    _font: pygame.font.Font | None = None

    def __init__(self) -> None:
        """Start both scores at zero."""
        self.p1: int = 0
        self.p2: int = 0

    @classmethod
    def _get_font(cls) -> pygame.font.Font:
        """Return the shared default font, creating it on first use.

        The font is cached on the class so that every :class:`Score`
        instance (and every frame) reuses a single ``pygame.font.Font``
        object rather than re-instantiating one.
        """
        if cls._font is None:
            cls._font = pygame.font.Font(None, 36)
        return cls._font

    # -- Mutators -----------------------------------------------------------

    def goal_for(self, side: str) -> None:
        """Award a goal to the given *side*.

        Parameters
        ----------
        side:
            ``"p1"`` or ``"p2"`` (case-insensitive).  Any other value is
            ignored, so a malformed side never corrupts the score.
        """
        key = side.lower()
        if key == "p1":
            self.p1 += 1
        elif key == "p2":
            self.p2 += 1

    def reset(self) -> None:
        """Reset both scores back to zero."""
        self.p1 = 0
        self.p2 = 0

    # -- Rendering ----------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        """Render the score line centred at the top of *surface*.

        The text is drawn as ``"P1: X  P2: Y"`` and horizontally centred
        near the top edge of the window.  A default font is created lazily
        and cached on the class so repeated calls do not re-instantiate it.

        Parameters
        ----------
        surface:
            The target ``pygame.Surface``.
        """
        font = Score._get_font()
        text = f"P1: {self.p1}  P2: {self.p2}"
        rendered = font.render(text, True, (255, 255, 255))
        rect = rendered.get_rect(
            midtop=(WINDOW_WIDTH // 2, FIELD_MARGIN // 2)
        )
        surface.blit(rendered, rect)

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid
        return f"Score(p1={self.p1}, p2={self.p2})"


# ---------------------------------------------------------------------------
# PhysicsEngine
# ---------------------------------------------------------------------------

class PhysicsEngine:
    """Per-frame physics: ball/player collision and goal detection.

    The engine is a thin stateless coordinator over the shared
    :class:`~entities.Ball` and :class:`~entities.Player` instances plus a
    :class:`Score`.  It does not own the entities; it mutates them in place.

    Attributes
    ----------
    ball:
        The ball whose velocity/position are simulated.
    player:
        The player that can strike the ball.
    score:
        The score object updated on goal detection.
    """

    def __init__(self, ball: Ball, player: Player, score: Score) -> None:
        """Bind the engine to its entities and score.

        Parameters
        ----------
        ball:
            The :class:`~entities.Ball` to simulate.
        player:
            The :class:`~entities.Player` that can collide with the ball.
        score:
            The :class:`Score` updated when a goal is scored.
        """
        self.ball: Ball = ball
        self.player: Player = player
        self.score: Score = score

    # -- Per-frame hook -----------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance physics by *dt* seconds.

        Steps:
        1. Resolve ball/player collision (separation, reflection, momentum).
        2. Detect a goal in either goal box; award a point and reset the ball.

        Parameters
        ----------
        dt:
            Elapsed time in seconds since the previous frame.
        """
        self._resolve_ball_player_collision()
        self._detect_goal()

    # -- Collision ----------------------------------------------------------

    def _resolve_ball_player_collision(self) -> None:
        """Separate and reflect the ball on contact with the player.

        If the distance between the two centres is less than the sum of
        their radii, the ball is pushed out along the collision normal, its
        velocity is reflected about that normal, and half of the player's
        velocity is added (momentum transfer).  The resulting speed is
        clamped to :data:`BALL_MAX_SPEED`.
        """
        ball = self.ball
        player = self.player

        delta: Vector2 = ball.position - player.position
        dist_sq: float = delta.length_squared()
        min_dist: float = float(ball.radius + player.radius)

        if dist_sq >= min_dist * min_dist:
            return  # no overlap

        # Normal from player centre to ball centre.  Guard against a
        # degenerate zero-length vector (centres exactly coincident).
        if dist_sq > 0.0:
            normal: Vector2 = delta / (dist_sq ** 0.5)
        else:
            normal = Vector2(1.0, 0.0)

        # 1. Separate the ball so it sits just outside the player.
        overlap: float = min_dist - (dist_sq ** 0.5)
        ball.position += normal * overlap

        # 2. Reflect the ball's velocity about the collision normal.
        #    v' = v - 2 * (v . n) * n
        v_dot_n: float = ball.velocity.dot(normal)
        if v_dot_n < 0.0:
            ball.velocity -= normal * (2.0 * v_dot_n)

        # 3. Add player momentum (fraction of the player's velocity).
        ball.velocity += player.velocity * PLAYER_MOMENTUM_FACTOR

        # 4. Clamp the resulting speed.
        speed: float = ball.velocity.length()
        if speed > BALL_MAX_SPEED:
            ball.velocity *= (BALL_MAX_SPEED / speed)

    # -- Goal detection -----------------------------------------------------

    def _goal_box_y_range(self) -> tuple[float, float]:
        """Return the vertical ``(y_min, y_max)`` span of the goal boxes.

        The goal boxes are centred on the pitch's vertical midline and are
        ``GOAL_BOX_HEIGHT`` tall, so the valid y-range is
        ``center ± GOAL_BOX_HEIGHT / 2``.
        """
        center_y: float = WINDOW_HEIGHT / 2.0
        half: float = GOAL_BOX_HEIGHT / 2.0
        return center_y - half, center_y + half

    def _detect_goal(self) -> None:
        """Award a point and reset the ball if it crossed a goal line.

        A **left** goal (scored by P2) is detected when the ball's x is
        below ``FIELD_MARGIN`` and its y lies within the goal box range.
        A **right** goal (scored by P1) is detected when the ball's x is
        above ``WINDOW_WIDTH - FIELD_MARGIN`` and its y is in range.
        """
        ball = self.ball
        y_min, y_max = self._goal_box_y_range()

        in_box_y: bool = y_min <= ball.position.y <= y_max

        if ball.position.x < FIELD_MARGIN and in_box_y:
            # Ball crossed the left goal line -> P2 scores.
            self.score.goal_for("p2")
            ball.reset()
        elif ball.position.x > WINDOW_WIDTH - FIELD_MARGIN and in_box_y:
            # Ball crossed the right goal line -> P1 scores.
            self.score.goal_for("p1")
            ball.reset()

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid
        return (
            f"PhysicsEngine(ball={self.ball!r}, player={self.player!r}, "
            f"score={self.score!r})"
        )
