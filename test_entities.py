"""Pytest suite for ``entities.py`` (Pygame Soccer entities).

Covers the four required checks:
  1. Import isolation -- ``import entities`` in a *fresh* subprocess must
     succeed WITHOUT importing ``main`` first; ``Ball`` and ``Player``
     classes must be accessible.
  2. Ball wall bounce -- colliding with a pitch edge inverts the relevant
     velocity component (elastic bounce).
  3. Player direction -- ``set_direction`` normalises the input vector, and
     ``update`` clamps the player at the wall (position pinned, velocity
     component zeroed).
  4. Game ownership -- ``main.Game()`` exposes ``.player`` and ``.ball``.

Run with::

    pytest -v test_main.py test_entities.py
"""

from __future__ import annotations

import os
import subprocess
import sys

# Headless SDL driver so no real window is ever attempted.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest  # noqa: E402
import pygame  # noqa: E402
from pygame.math import Vector2  # noqa: E402

import config  # noqa: E402  (dependency-free; import-safe by design)
import entities  # noqa: E402  (import-safe by design; verified below)
import main  # noqa: E402  (import-safe by design; verified below)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_fresh(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a fresh interpreter with the dummy video driver."""
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


@pytest.fixture
def display():
    """Initialise the (dummy) video system so pygame primitives work."""
    if not pygame.display.get_init():
        pygame.display.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    yield


# ---------------------------------------------------------------------------
# 1. Import isolation (subprocess)
# ---------------------------------------------------------------------------

def test_import_entities_standalone_subprocess() -> None:
    """``import entities`` must work in a fresh interpreter without ``main``.

    The subprocess imports ``entities`` first and asserts that ``main`` was
    NOT pulled in as a side effect, and that both entity classes are exposed.
    """
    code = (
        "import sys\n"
        "import entities\n"
        "assert 'main' not in sys.modules, 'entities must not import main'\n"
        "assert hasattr(entities, 'Ball'), 'entities.Ball missing'\n"
        "assert hasattr(entities, 'Player'), 'entities.Player missing'\n"
        "print('OK')\n"
    )
    result = _run_fresh(code)
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# 2. Ball wall bounce inverts velocity
# ---------------------------------------------------------------------------

def test_ball_bounce_inverts_velocity(display) -> None:
    """Hitting a wall must invert the corresponding velocity component."""
    left, top, right, bottom = (
        float(config.FIELD_MARGIN),
        float(config.FIELD_MARGIN),
        float(config.WINDOW_WIDTH - config.FIELD_MARGIN),
        float(config.WINDOW_HEIGHT - config.FIELD_MARGIN),
    )
    r = config.BALL_RADIUS

    # --- Right wall: moving right (+x) must bounce to moving left (-x) ---
    ball = entities.Ball()
    ball.position = Vector2(right - r - 1.0, (top + bottom) / 2.0)
    ball.velocity = Vector2(500.0, 0.0)
    ball.update(dt=1.0)
    assert ball.velocity.x < 0.0, "right-wall bounce should invert vx to negative"
    assert ball.position.x + r <= right + 1e-6, "ball must be clamped inside right wall"

    # --- Left wall: moving left (-x) must bounce to moving right (+x) ---
    ball = entities.Ball()
    ball.position = Vector2(left + r + 1.0, (top + bottom) / 2.0)
    ball.velocity = Vector2(-500.0, 0.0)
    ball.update(dt=1.0)
    assert ball.velocity.x > 0.0, "left-wall bounce should invert vx to positive"
    assert ball.position.x - r >= left - 1e-6, "ball must be clamped inside left wall"

    # --- Bottom wall: moving down (+y) must bounce to moving up (-y) ---
    ball = entities.Ball()
    ball.position = Vector2((left + right) / 2.0, bottom - r - 1.0)
    ball.velocity = Vector2(0.0, 500.0)
    ball.update(dt=1.0)
    assert ball.velocity.y < 0.0, "bottom-wall bounce should invert vy to negative"
    assert ball.position.y + r <= bottom + 1e-6, "ball must be clamped inside bottom wall"

    # --- Top wall: moving up (-y) must bounce to moving down (+y) ---
    ball = entities.Ball()
    ball.position = Vector2((left + right) / 2.0, top + r + 1.0)
    ball.velocity = Vector2(0.0, -500.0)
    ball.update(dt=1.0)
    assert ball.velocity.y > 0.0, "top-wall bounce should invert vy to positive"
    assert ball.position.y - r >= top - 1e-6, "ball must be clamped inside top wall"


# ---------------------------------------------------------------------------
# 3. Player direction normalises + clamps at wall
# ---------------------------------------------------------------------------

def test_player_set_direction_normalises(display) -> None:
    """``set_direction`` must normalise the input vector to unit length."""
    p = entities.Player()

    # Axis-aligned input stays unit length.
    p.set_direction(1.0, 0.0)
    assert p.direction.length() == pytest.approx(1.0)
    assert p.direction.x == pytest.approx(1.0)
    assert p.direction.y == pytest.approx(0.0)

    # Diagonal input must be normalised (not faster than axis-aligned).
    p.set_direction(1.0, 1.0)
    assert p.direction.length() == pytest.approx(1.0)
    assert p.direction.x == pytest.approx(1.0 / 2**0.5)
    assert p.direction.y == pytest.approx(1.0 / 2**0.5)

    # Zero input stops the player.
    p.set_direction(0.0, 0.0)
    assert p.direction.length() == pytest.approx(0.0)


def test_player_clamps_at_wall(display) -> None:
    """Driving into a wall must clamp position and zero that velocity."""
    left, top, right, bottom = (
        float(config.FIELD_MARGIN),
        float(config.FIELD_MARGIN),
        float(config.WINDOW_WIDTH - config.FIELD_MARGIN),
        float(config.WINDOW_HEIGHT - config.FIELD_MARGIN),
    )
    r = config.PLAYER_RADIUS

    # --- Right wall: move right, must clamp and zero vx ---
    p = entities.Player()
    p.position = Vector2(right - r - 1.0, (top + bottom) / 2.0)
    p.set_direction(1.0, 0.0)
    p.update(dt=1.0)
    assert p.position.x + r <= right + 1e-6, "player must be clamped inside right wall"
    assert p.velocity.x == pytest.approx(0.0), "vx must be zeroed at right wall"

    # --- Left wall: move left, must clamp and zero vx ---
    p = entities.Player()
    p.position = Vector2(left + r + 1.0, (top + bottom) / 2.0)
    p.set_direction(-1.0, 0.0)
    p.update(dt=1.0)
    assert p.position.x - r >= left - 1e-6, "player must be clamped inside left wall"
    assert p.velocity.x == pytest.approx(0.0), "vx must be zeroed at left wall"

    # --- Bottom wall: move down, must clamp and zero vy ---
    p = entities.Player()
    p.position = Vector2((left + right) / 2.0, bottom - r - 1.0)
    p.set_direction(0.0, 1.0)
    p.update(dt=1.0)
    assert p.position.y + r <= bottom + 1e-6, "player must be clamped inside bottom wall"
    assert p.velocity.y == pytest.approx(0.0), "vy must be zeroed at bottom wall"

    # --- Top wall: move up, must clamp and zero vy ---
    p = entities.Player()
    p.position = Vector2((left + right) / 2.0, top + r + 1.0)
    p.set_direction(0.0, -1.0)
    p.update(dt=1.0)
    assert p.position.y - r >= top - 1e-6, "player must be clamped inside top wall"
    assert p.velocity.y == pytest.approx(0.0), "vy must be zeroed at top wall"


# ---------------------------------------------------------------------------
# 4. Game owns .player and .ball
# ---------------------------------------------------------------------------

def test_game_has_player_and_ball(display) -> None:
    """``main.Game()`` must expose ``.player`` and ``.ball`` entities."""
    game = main.Game()
    assert hasattr(game, "player"), "Game must own a .player"
    assert hasattr(game, "ball"), "Game must own a .ball"
    assert isinstance(game.player, entities.Player), ".player must be a Player"
    assert isinstance(game.ball, entities.Ball), ".ball must be a Ball"
