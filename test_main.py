"""Pytest suite for ``main.py`` (Pygame Soccer core).

Covers:
  1. Import safety  -- ``import main`` must NOT open a window / init display.
  2. Constants      -- 800x600 window, 60 FPS, pitch green (34, 139, 34).
  3. draw_field()   -- renders a green pitch + white lines on a Surface.
  4. Game class     -- running=True; update/draw/handle_events run cleanly.
  5. Event handling -- posting QUIT flips ``running`` to False.
  6. Visual (cv2)   -- region/colour analysis of the rendered field frame.

Run with::

    pytest -v test_main.py
"""

from __future__ import annotations

import os
import subprocess
import sys

# Headless SDL driver so no real window is ever attempted.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest
import numpy as np
import pygame
import cv2

import main  # noqa: E402  (import-safe by design; verified below)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb(surface: pygame.Surface) -> np.ndarray:
    """Convert a pygame Surface to an ``H x W x 3`` uint8 RGB numpy array."""
    w, h = surface.get_size()
    raw = pygame.image.tostring(surface, "RGB")
    return np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))


def _render_field() -> pygame.Surface:
    """Render the field onto a fresh 800x600 surface (black background)."""
    surf = pygame.Surface((main.WINDOW_WIDTH, main.WINDOW_HEIGHT))
    surf.fill((0, 0, 0))
    main.draw_field(surf)
    return surf


def _px(surf: pygame.Surface, x: int, y: int) -> tuple[int, int, int]:
    """Return the RGB triple at (x, y)."""
    return tuple(surf.get_at((x, y))[:3])




@pytest.fixture
def display():
    """Initialise the (dummy) video system so the pygame event queue works."""
    if not pygame.display.get_init():
        pygame.display.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((main.WINDOW_WIDTH, main.WINDOW_HEIGHT))
    yield
    pygame.event.clear()


# ---------------------------------------------------------------------------
# 1. Import safety
# ---------------------------------------------------------------------------

def test_import_safety_fresh_interpreter() -> None:
    """``import main`` in a *fresh* interpreter must not init the display.

    We spawn a subprocess with the dummy video driver and assert that
    importing the module leaves ``pygame.display.get_init()`` False and
    that no display surface exists.
    """
    code = (
        "import os\n"
        "os.environ['SDL_VIDEODRIVER']='dummy'\n"
        "import pygame\n"
        "import main\n"
        "print(pygame.display.get_init())\n"
    )
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, f"import failed: {result.stderr}"
    # The last printed line must be "False" (display not initialised).
    assert result.stdout.strip().splitlines()[-1].strip() == "False"


def test_entities_import_fresh_interpreter() -> None:
    """``import entities`` in a *fresh* interpreter must not fail.

    Regression guard for the historical circular import between
    ``main`` and ``entities``.  ``entities`` must be importable on its
    own (without importing ``main`` first) and must expose the two core
    entity classes.
    """
    code = (
        "import os\n"
        "os.environ['SDL_VIDEODRIVER']='dummy'\n"
        "import entities\n"
        "assert hasattr(entities, 'Ball')\n"
        "assert hasattr(entities, 'Player')\n"
        "print('OK')\n"
    )
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, f"import entities failed: {result.stderr}"
    assert result.stdout.strip().splitlines()[-1].strip() == "OK"


def test_import_does_not_set_display_mode() -> None:
    """Within this process, importing main must not have created a window."""
    # main was imported at module load; ensure no display surface exists.
    assert pygame.display.get_surface() is None


# ---------------------------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------------------------

def test_window_dimensions() -> None:
    assert main.WINDOW_WIDTH == 800
    assert main.WINDOW_HEIGHT == 600


def test_fps() -> None:
    assert main.FPS == 60


def test_pitch_green_colour() -> None:
    assert main.PITCH_GREEN == (34, 139, 34)


def test_line_white_colour() -> None:
    assert main.LINE_WHITE == (255, 255, 255)


# ---------------------------------------------------------------------------
# 3. draw_field rendering
# ---------------------------------------------------------------------------

def test_draw_field_renders_green_pitch() -> None:
    """The centre of the pitch must be the configured green colour."""
    surf = _render_field()
    cx, cy = main.WINDOW_WIDTH // 2, main.WINDOW_HEIGHT // 2
    # Centre spot is white; sample slightly offset to hit the green pitch.
    assert _px(surf, cx + 20, cy + 20) == main.PITCH_GREEN


def test_draw_field_renders_white_lines() -> None:
    """Boundary and center lines must be white."""
    surf = _render_field()
    # Outer boundary (top edge of the field rect).
    assert _px(surf, main.WINDOW_WIDTH // 2, main.FIELD_MARGIN) == main.LINE_WHITE
    # Vertical center line.
    cx = main.WINDOW_WIDTH // 2
    assert _px(surf, cx, main.FIELD_MARGIN + 10) == main.LINE_WHITE


def test_draw_field_pitch_inset_by_margin() -> None:
    """Pixels outside the field margin must NOT be pitch green."""
    surf = _render_field()
    # A point well inside the margin band (background, not pitch).
    assert _px(surf, 5, 5) != main.PITCH_GREEN


# ---------------------------------------------------------------------------
# 4. Game class smoke tests
# ---------------------------------------------------------------------------

def test_game_initial_state() -> None:
    game = main.Game()
    assert game.running is True


def test_game_update_runs_cleanly() -> None:
    game = main.Game()
    game.update(0.016)  # should not raise
    assert game.running is True


def test_game_draw_runs_cleanly() -> None:
    game = main.Game()
    surf = pygame.Surface((main.WINDOW_WIDTH, main.WINDOW_HEIGHT))
    surf.fill((0, 0, 0))
    game.draw(surf)  # should not raise
    # draw() delegates to draw_field -> pitch green present.
    assert _px(surf, main.WINDOW_WIDTH // 2 + 20, main.WINDOW_HEIGHT // 2 + 20) == main.PITCH_GREEN


@pytest.mark.usefixtures('display')
def test_game_handle_events_runs_cleanly() -> None:
    game = main.Game()
    pygame.event.clear()
    game.handle_events()  # should not raise
    assert game.running is True


# ---------------------------------------------------------------------------
# 5. Event handling -- QUIT
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures('display')
def test_quit_event_stops_game() -> None:
    game = main.Game()
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    game.handle_events()
    assert game.running is False


@pytest.mark.usefixtures('display')
def test_escape_key_stops_game() -> None:
    game = main.Game()
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    game.handle_events()
    assert game.running is False


# ---------------------------------------------------------------------------
# 6. Visual validation via OpenCV
# ---------------------------------------------------------------------------

def test_visual_field_layout_cv2() -> None:
    """Analyse the rendered frame with cv2 to confirm pitch + lines."""
    surf = _render_field()
    frame = _rgb(surf)  # H x W x 3 RGB
    assert frame.shape == (main.WINDOW_HEIGHT, main.WINDOW_WIDTH, 3)

    # --- Green pitch region ---
    green_mask = (
        (np.abs(frame[..., 0].astype(int) - 34) < 10)
        & (np.abs(frame[..., 1].astype(int) - 139) < 10)
        & (np.abs(frame[..., 2].astype(int) - 34) < 10)
    )
    green_frac = green_mask.mean()
    assert green_frac > 0.5, f"expected a large green pitch, got {green_frac:.2f}"

    # --- White lines region ---
    white_mask = (
        (frame[..., 0] > 240)
        & (frame[..., 1] > 240)
        & (frame[..., 2] > 240)
    )
    white_frac = white_mask.mean()
    assert white_frac > 0.01, f"expected visible white lines, got {white_frac:.4f}"

    # --- Center line: a vertical white stripe near the horizontal centre ---
    cx = main.WINDOW_WIDTH // 2
    center_col = white_mask[:, cx - 2 : cx + 3].any(axis=1)
    assert center_col[main.FIELD_MARGIN : main.WINDOW_HEIGHT - main.FIELD_MARGIN].mean() > 0.5

    # --- cv2 sanity: detect edges (lines) via Canny ---
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    assert edges.sum() > 0, "Canny should detect the field line edges"
