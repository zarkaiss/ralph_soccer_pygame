"""Pytest suite for ``physics.py`` (Score, PhysicsEngine) and the
goal-mouth fix in ``entities.py`` (Ball.update).

Coverage
--------
* Ball goal-mouth behaviour (passes through the mouth, bounces outside it).
* Ball top/bottom wall bounces.
* ``Score.goal_for`` / ``Score.reset`` / ``Score.draw``.
* ``PhysicsEngine`` ball/player collision separation + speed clamp.
* ``PhysicsEngine`` goal detection (left -> P2, right -> P1) with ball reset.

Run with::

    SDL_VIDEODRIVER=dummy python -m pytest test_physics.py -v --tb=short
"""

from __future__ import annotations

import os

# Headless SDL driver so no real window is ever attempted.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
from pygame.math import Vector2  # noqa: E402

import config  # noqa: E402  (dependency-free; import-safe by design)
from entities import Ball, Player  # noqa: E402
from physics import BALL_MAX_SPEED, PhysicsEngine, Score  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_ball(x: float, y: float, vx: float, vy: float) -> Ball:
    """Create a ball at ``(x, y)`` with velocity ``(vx, vy)``."""
    ball = Ball()
    ball.position = Vector2(x, y)
    ball.velocity = Vector2(vx, vy)
    return ball


def _make_player(x: float, y: float) -> Player:
    """Create a player centred at ``(x, y)`` with zero velocity."""
    player = Player(start_position=Vector2(x, y))
    player.velocity = Vector2(0.0, 0.0)
    return player


def _engine(ball: Ball, player: Player) -> tuple[PhysicsEngine, Score]:
    """Build a ``PhysicsEngine`` with a fresh ``Score`` and return both."""
    score = Score()
    return PhysicsEngine(ball, player, score), score


def _goal_mouth_y_range() -> tuple[float, float]:
    """Mirror the goal-mouth span used by ``entities.Ball.update``."""
    top = config.WINDOW_HEIGHT / 2 - config.GOAL_BOX_HEIGHT / 2
    bottom = config.WINDOW_HEIGHT / 2 + config.GOAL_BOX_HEIGHT / 2
    return top, bottom


# ---------------------------------------------------------------------------
# 1-4. Goal-mouth fix (entities.Ball.update)
# ---------------------------------------------------------------------------

def test_ball_passes_left_goal_mouth() -> None:
    """Ball inside the goal mouth crosses the left goal line (no bounce)."""
    ball = _make_ball(50, 300, -100, 0)
    ball.update(1 / 60)
    assert ball.position.x < 50  # crossed the goal line
    assert ball.velocity.x < 0  # NOT reflected


def test_ball_bounces_left_wall_outside_mouth() -> None:
    """Ball outside the goal mouth bounces off the left wall."""
    ball = _make_ball(50, 100, -100, 0)
    ball.update(1 / 60)
    assert ball.velocity.x > 0  # bounced (reflected)


def test_ball_passes_right_goal_mouth() -> None:
    """Ball inside the goal mouth crosses the right goal line (no bounce)."""
    ball = _make_ball(750, 300, 100, 0)
    ball.update(1 / 60)
    assert ball.position.x > 750  # crossed the goal line
    assert ball.velocity.x > 0  # NOT reflected


def test_ball_bounces_right_wall_outside_mouth() -> None:
    """Ball outside the goal mouth bounces off the right wall."""
    ball = _make_ball(750, 500, 100, 0)
    ball.update(1 / 60)
    assert ball.velocity.x < 0  # bounced (reflected)


# ---------------------------------------------------------------------------
# 5-6. Top / bottom wall bounces
# ---------------------------------------------------------------------------

def test_ball_bounces_top() -> None:
    """Ball hitting the top wall reflects its vertical velocity."""
    ball = _make_ball(400, 50, 0, -100)
    ball.update(1 / 60)
    assert ball.velocity.y > 0


def test_ball_bounces_bottom() -> None:
    """Ball hitting the bottom wall reflects its vertical velocity."""
    ball = _make_ball(400, 550, 0, 100)
    ball.update(1 / 60)
    assert ball.velocity.y < 0


# ---------------------------------------------------------------------------
# 7-8. Score
# ---------------------------------------------------------------------------

def test_score_goal_for_and_reset() -> None:
    """``goal_for`` increments the correct side; ``reset`` zeroes both."""
    s = Score()
    s.goal_for("p1")
    s.goal_for("P2")  # case-insensitive
    assert s.p1 == 1
    assert s.p2 == 1

    s.goal_for("bogus")  # malformed side must be ignored
    assert s.p1 == 1
    assert s.p2 == 1

    s.reset()
    assert s.p1 == 0
    assert s.p2 == 0


def test_score_draw() -> None:
    """``Score.draw`` renders without raising on a valid surface.

    A ``pygame.Surface`` and ``pygame.font`` do NOT require a display
    surface, so this stays headless-safe even when the SDL ``dummy``
    video driver is unavailable in the environment.
    """
    if not pygame.font.get_init():
        pygame.font.init()

    s = Score()
    s.p1, s.p2 = 3, 2
    surf = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    s.draw(surf)  # should not raise


# ---------------------------------------------------------------------------
# 9. PhysicsEngine ball/player collision separation + speed clamp
# ---------------------------------------------------------------------------

def test_physics_collision_separates_and_clamps() -> None:
    """Overlapping ball/player are separated and speed is clamped."""
    ball = _make_ball(400, 300, 0, 0)
    player = _make_player(410, 300)  # 10px apart < ball.radius + player.radius
    engine, _ = _engine(ball, player)

    before = Vector2(ball.position)
    engine.update(1 / 60)

    assert ball.position != before  # separation happened
    # Ball must now sit at least (sum of radii) away from the player.
    min_dist = ball.radius + player.radius
    assert (ball.position - player.position).length() >= min_dist - 1e-6
    # Speed must never exceed the hard clamp.
    assert ball.velocity.length() <= BALL_MAX_SPEED + 1e-6


# ---------------------------------------------------------------------------
# 10-11. PhysicsEngine goal detection + reset
# ---------------------------------------------------------------------------

def test_goal_detection_left_scores_p2() -> None:
    """Ball past the left goal line in the mouth scores P2 and resets."""
    ball = _make_ball(30, 300, 0, 0)  # x < FIELD_MARGIN, y in goal mouth
    player = _make_player(400, 300)
    engine, score = _engine(ball, player)

    engine.update(1 / 60)

    assert score.p2 == 1
    assert score.p1 == 0
    # Ball reset to pitch centre with zero velocity.
    assert ball.position == Vector2(
        config.WINDOW_WIDTH / 2, config.WINDOW_HEIGHT / 2
    )
    assert ball.velocity == Vector2(0, 0)


def test_goal_detection_right_scores_p1() -> None:
    """Ball past the right goal line in the mouth scores P1 and resets."""
    ball = _make_ball(770, 300, 0, 0)  # x > WINDOW_WIDTH - FIELD_MARGIN
    player = _make_player(400, 300)
    engine, score = _engine(ball, player)

    engine.update(1 / 60)

    assert score.p1 == 1
    assert score.p2 == 0
    # Ball reset to pitch centre with zero velocity.
    assert ball.position == Vector2(
        config.WINDOW_WIDTH / 2, config.WINDOW_HEIGHT / 2
    )
    assert ball.velocity == Vector2(0, 0)


def test_no_goal_outside_mouth() -> None:
    """Ball past a goal line but OUTSIDE the mouth does not score."""
    y_out = _goal_mouth_y_range()[0] - 5  # just above the goal mouth
    ball = _make_ball(30, y_out, 0, 0)
    player = _make_player(400, 300)
    engine, score = _engine(ball, player)

    engine.update(1 / 60)

    assert score.p1 == 0
    assert score.p2 == 0
    # Ball was NOT reset (still where we placed it).
    assert ball.position.x == 30
