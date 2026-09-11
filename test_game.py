"""Pytest + OpenCV integration test suite for the Pygame Soccer game.

This suite exercises the fully-wired :class:`main.Game` object (entities +
physics + scoring) in a **headless** environment and verifies both the
behavioural contract (update/draw cycle, bouncing, collisions, goal
detection, speed clamping) and the *visual* output using OpenCV
(:func:`cv2.inRange` for colour-region analysis and
:func:`cv2.HoughCircles` for circle detection).

Design notes
------------
* ``SDL_VIDEODRIVER`` is forced to ``dummy`` **before** pygame is imported
  so no real window is ever attempted.
* All tests are deterministic: no randomness, no wall-clock dependence, and
  float comparisons use :func:`pytest.approx`.
* The suite is written against the *actual* public API of the project:

  * ``Game`` owns a single ``player`` (P1 / blue), a ``ball``, a ``score``
    and a ``physics`` engine (see :mod:`main`).
  * ``PhysicsEngine(ball, player, score)`` with ``update(dt)``.
  * Goal detection: ball crossing the **left** goal line awards **P2**;
    crossing the **right** goal line awards **P1** (see :mod:`physics`).
  * ``BALL_MAX_SPEED`` (800.0 px/s) is the hard speed clamp.

Run with::

    pytest -v test_game.py
"""

from __future__ import annotations

import os

# Headless SDL driver so no real window is ever attempted.  Must be set
# before pygame is imported anywhere in the process.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import numpy as np
import pygame
import pytest

import config
import main
from entities import Ball, Player
from physics import BALL_MAX_SPEED, PhysicsEngine, Score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb(surface: pygame.Surface) -> np.ndarray:
    """Convert a pygame Surface to an ``H x W x 3`` uint8 RGB numpy array.

    Parameters
    ----------
    surface:
        The source pygame surface.

    Returns
    -------
    np.ndarray
        A contiguous ``(height, width, 3)`` array in RGB channel order.
    """
    w, h = surface.get_size()
    raw = pygame.image.tostring(surface, "RGB")
    return np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))


def _render_frame(game: main.Game) -> np.ndarray:
    """Render a full game frame onto a fresh surface and return it as RGB.

    The surface is filled black first so any region the game does not paint
    is unambiguously detectable (used by the "no black holes" test).

    Parameters
    ----------
    game:
        A fully-constructed :class:`main.Game` instance.

    Returns
    -------
    np.ndarray
        The rendered frame as an ``H x W x 3`` RGB array.
    """
    surf = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    surf.fill((0, 0, 0))
    game.draw(surf)
    return _rgb(surf)


def _count_color_pixels(
    arr: np.ndarray,
    target: tuple[int, int, int],
    tol: int = 30,
) -> int:
    """Count pixels within ``tol`` (per channel) of *target* using cv2.

    Parameters
    ----------
    arr:
        An ``H x W x 3`` RGB array.
    target:
        The reference ``(R, G, B)`` colour.
    tol:
        Per-channel tolerance (inclusive) around each target channel.

    Returns
    -------
    int
        Number of pixels whose every channel lies within the tolerance band.
    """
    lower = np.array(
        [max(0, c - tol) for c in target], dtype=np.uint8
    )
    upper = np.array(
        [min(255, c + tol) for c in target], dtype=np.uint8
    )
    mask = cv2.inRange(arr, lower, upper)
    return int(np.count_nonzero(mask))


def _find_circles(arr: np.ndarray) -> list[tuple[int, int, int]]:
    """Detect circles in *arr* via :func:`cv2.HoughCircles`.

    The image is greyscaled and lightly blurred to stabilise the Hough
    transform.  Returns a list of ``(x, y, radius)`` integer tuples (empty
    if nothing is found).

    Parameters
    ----------
    arr:
        An ``H x W x 3`` RGB array.

    Returns
    -------
    list[tuple[int, int, int]]
        Detected circles as ``(x, y, radius)``.
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=80,
        param2=25,
        minRadius=5,
        maxRadius=40,
    )
    if circles is None:
        return []
    return [
        (round(c[0]), round(c[1]), round(c[2]))
        for c in circles[0]
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def display():
    """Initialise the (dummy) video system so pygame surfaces work.

    Yields the display surface and tears it down afterwards.
    """
    if not pygame.display.get_init():
        pygame.display.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    yield pygame.display.get_surface()
    pygame.display.quit()


@pytest.fixture
def game(display) -> main.Game:
    """Provide a fresh, fully-constructed :class:`main.Game` instance."""
    return main.Game()


# ---------------------------------------------------------------------------
# 1. Game integration
# ---------------------------------------------------------------------------

class TestGameIntegration:
    """Behavioural contract of the fully-wired Game object."""

    def test_game_initialization(self, game: main.Game) -> None:
        """A fresh Game is running and owns all core subsystems."""
        assert game.running is True
        assert isinstance(game.ball, Ball)
        assert isinstance(game.player, Player)
        assert isinstance(game.score, Score)
        assert isinstance(game.physics, PhysicsEngine)

        # The physics engine is bound to the same shared entities.
        assert game.physics.ball is game.ball
        assert game.physics.player is game.player
        assert game.physics.score is game.score

        # Scores start at zero.
        assert game.score.p1 == 0
        assert game.score.p2 == 0

    def test_update_draw_cycle(self, game: main.Game, display) -> None:
        """Ten update+draw cycles complete without raising."""
        dt = 1.0 / 60.0
        surf = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        for _ in range(10):
            game.update(dt)
            game.draw(surf)
        # Reaching here without an exception is the assertion.
        assert game.running is True

    def test_ball_bounce(self, game: main.Game) -> None:
        """After a wall hit the ball stays within the pitch bounds."""
        left = float(config.FIELD_MARGIN)
        top = float(config.FIELD_MARGIN)
        right = float(config.WINDOW_WIDTH - config.FIELD_MARGIN)
        bottom = float(config.WINDOW_HEIGHT - config.FIELD_MARGIN)
        r = game.ball.radius

        # Drive the ball hard into the top-left corner region.
        game.ball.position = pygame.math.Vector2(left + r, top + r)
        game.ball.velocity = pygame.math.Vector2(-500.0, -500.0)

        dt = 1.0 / 60.0
        for _ in range(5):
            game.ball.update(dt)

        # The ball must be clamped back inside the playable pitch.
        tol = 1e-6
        assert game.ball.position.x >= left + r - tol
        assert game.ball.position.x <= right - r + tol
        assert game.ball.position.y >= top + r - tol
        assert game.ball.position.y <= bottom - r + tol

        # Bouncing off the top wall must reverse the vertical velocity.
        assert game.ball.velocity.y >= 0.0


# ---------------------------------------------------------------------------
# 2. Visual verification (OpenCV)
# ---------------------------------------------------------------------------

class TestVisualVerification:
    """Pixel-level verification of the rendered frame using OpenCV."""

    def test_field_green(self, game: main.Game, display) -> None:
        """More than 50% of the frame is near PITCH_GREEN."""
        arr = _render_frame(game)
        total = arr.shape[0] * arr.shape[1]
        green = _count_color_pixels(arr, config.PITCH_GREEN, tol=30)
        assert green / total > 0.50

    def test_field_white_lines(self, game: main.Game, display) -> None:
        """The rendered frame contains a substantial number of white pixels."""
        arr = _render_frame(game)
        white = _count_color_pixels(arr, config.LINE_WHITE, tol=20)
        assert white > 100

    def test_ball_visible(self, game: main.Game, display) -> None:
        """A white circle is detected near the pitch centre."""
        arr = _render_frame(game)
        circles = _find_circles(arr)
        assert circles, "expected at least one detected circle (the ball)"

        cx = config.WINDOW_WIDTH / 2.0
        cy = config.WINDOW_HEIGHT / 2.0
        # The ball starts at the pitch centre; some circle should be close.
        near_center = [
            (x, y, r) for (x, y, r) in circles
            if abs(x - cx) < 40 and abs(y - cy) < 40
        ]
        assert near_center, "expected a circle near the pitch centre"

    def test_players_visible(self, game: main.Game, display) -> None:
        """PLAYER_COLOR_P1 (blue) pixels are present in the frame."""
        arr = _render_frame(game)
        blue = _count_color_pixels(arr, config.PLAYER_COLOR_P1, tol=40)
        assert blue > 0

    def test_frame_dimensions(self, game: main.Game, display) -> None:
        """The rendered frame is exactly 600x800x3 (H x W x 3)."""
        arr = _render_frame(game)
        assert arr.shape == (
            config.WINDOW_HEIGHT,
            config.WINDOW_WIDTH,
            3,
        )
        assert arr.shape == (600, 800, 3)
        assert arr.dtype == np.uint8

    def test_no_black_holes(self, game: main.Game, display) -> None:
        """The central field region contains no pure-black pixels."""
        arr = _render_frame(game)
        h, w, _ = arr.shape

        # A central band well inside the pitch (avoids the black border).
        y0, y1 = h // 4, (3 * h) // 4
        x0, x1 = w // 4, (3 * w) // 4
        region = arr[y0:y1, x0:x1]

        black = _count_color_pixels(region, (0, 0, 0), tol=0)
        assert black == 0


# ---------------------------------------------------------------------------
# 3. Physics integration
# ---------------------------------------------------------------------------

class TestPhysicsIntegration:
    """Ball/player collision, goal detection, and speed clamping."""

    def test_ball_player_collision(self, display) -> None:
        """The ball's velocity changes after a player collision."""
        ball = Ball()
        player = Player()
        score = Score()
        engine = PhysicsEngine(ball=ball, player=player, score=score)

        # Place the player just left of the ball so they overlap.
        player.position = pygame.math.Vector2(400.0, 300.0)
        player.velocity = pygame.math.Vector2(300.0, 0.0)
        player.direction = pygame.math.Vector2(1.0, 0.0)

        ball.position = pygame.math.Vector2(410.0, 300.0)
        ball.velocity = pygame.math.Vector2(0.0, 0.0)

        before = pygame.math.Vector2(ball.velocity)
        engine.update(1.0 / 60.0)
        after = pygame.math.Vector2(ball.velocity)

        assert after != before
        # Momentum transfer should push the ball in the player's direction.
        assert after.x > before.x

    def test_goal_detection(self, display) -> None:
        """A ball inside the left goal box awards a point to P2."""
        ball = Ball()
        player = Player()
        score = Score()
        engine = PhysicsEngine(ball=ball, player=player, score=score)

        # Keep the player far away so it does not interfere.
        player.position = pygame.math.Vector2(600.0, 100.0)
        player.velocity = pygame.math.Vector2(0.0, 0.0)

        # Ball just past the left goal line, within the goal box y-range.
        ball.position = pygame.math.Vector2(
            config.FIELD_MARGIN - 5.0, config.WINDOW_HEIGHT / 2.0
        )
        ball.velocity = pygame.math.Vector2(-100.0, 0.0)

        assert score.p2 == 0
        engine.update(1.0 / 60.0)
        assert score.p2 == 1

        # A goal resets the ball to the pitch centre.
        assert ball.position.x == pytest.approx(config.WINDOW_WIDTH / 2.0)
        assert ball.position.y == pytest.approx(config.WINDOW_HEIGHT / 2.0)

    def test_ball_speed_clamp(self, display) -> None:
        """After a collision the ball speed never exceeds BALL_MAX_SPEED."""
        ball = Ball()
        player = Player()
        score = Score()
        engine = PhysicsEngine(ball=ball, player=player, score=score)

        # Overlap the two and give the player a large velocity.
        player.position = pygame.math.Vector2(400.0, 300.0)
        player.velocity = pygame.math.Vector2(1000.0, 1000.0)
        player.direction = pygame.math.Vector2(1.0, 1.0).normalize()

        ball.position = pygame.math.Vector2(410.0, 300.0)
        ball.velocity = pygame.math.Vector2(1000.0, 1000.0)

        engine.update(1.0 / 60.0)

        speed = ball.velocity.length()
        assert speed <= BALL_MAX_SPEED + 1e-6
        assert speed == pytest.approx(min(speed, BALL_MAX_SPEED), abs=1e-6)
