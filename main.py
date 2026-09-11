"""Pygame Soccer -- core window, field rendering, and main event loop.

This module establishes the foundational architecture for a 2D soccer game.
It provides:

* A ``Game`` class that owns all mutable game state and exposes
  ``update(dt)`` / ``draw(surface)`` extension points for future
  entity and physics modules.
* A ``draw_field(surface)`` helper that paints the pitch, boundary lines,
  center circle, and goal boxes.
* A ``main()`` entry point that wires up the window, clock, and event loop.

The module is **headless-safe**: importing it does NOT call
``pygame.display.set_mode`` or open a window.  Those side-effects are
confined to ``main()`` so unit tests can import the module freely.

All read-only configuration constants (window geometry, colours, field
layout) are imported from :mod:`config`, the single source of truth shared
with :mod:`entities`.  This removes the previous circular dependency
between ``main`` and ``entities``.
"""

from __future__ import annotations

import pygame

from config import (
    CENTER_CIRCLE_RADIUS,
    FIELD_MARGIN,
    FPS,
    GOAL_BOX_HEIGHT,
    GOAL_BOX_WIDTH,
    LINE_WHITE,
    LINE_WIDTH,
    PITCH_GREEN,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    WINDOW_WIDTH,
)
from entities import PLAYER_COLOR_P1, Ball, Player
from physics import PhysicsEngine, Score

# ---------------------------------------------------------------------------
# Field rendering
# ---------------------------------------------------------------------------

def draw_field(surface: pygame.Surface) -> None:
    """Draw the soccer pitch and all standard markings onto *surface*.

    The field is drawn as a green rectangle inset by ``FIELD_MARGIN``
    from the window edges, with white boundary lines, a center line,
    a center circle, and two goal boxes (left and right).

    Parameters
    ----------
    surface:
        The target ``pygame.Surface`` (typically the display surface).
    """
    # Pitch background
    field_rect = pygame.Rect(
        FIELD_MARGIN,
        FIELD_MARGIN,
        WINDOW_WIDTH - 2 * FIELD_MARGIN,
        WINDOW_HEIGHT - 2 * FIELD_MARGIN,
    )
    surface.fill(PITCH_GREEN, field_rect)

    # Outer boundary
    pygame.draw.rect(
        surface,
        LINE_WHITE,
        field_rect,
        width=LINE_WIDTH,
    )

    # Center line (vertical)
    cx = field_rect.centerx
    pygame.draw.line(
        surface,
        LINE_WHITE,
        (cx, field_rect.top),
        (cx, field_rect.bottom),
        LINE_WIDTH,
    )

    # Center circle
    pygame.draw.circle(
        surface,
        LINE_WHITE,
        (cx, field_rect.centery),
        CENTER_CIRCLE_RADIUS,
        LINE_WIDTH,
    )

    # Center spot
    pygame.draw.circle(surface, LINE_WHITE, (cx, field_rect.centery), 4)

    # --- Goal boxes (left and right) ---
    box_half = GOAL_BOX_HEIGHT // 2

    # Left goal box
    left_box = pygame.Rect(
        field_rect.left,
        field_rect.centery - box_half,
        GOAL_BOX_WIDTH,
        GOAL_BOX_HEIGHT,
    )
    pygame.draw.rect(surface, LINE_WHITE, left_box, width=LINE_WIDTH)

    # Right goal box
    right_box = pygame.Rect(
        field_rect.right - GOAL_BOX_WIDTH,
        field_rect.centery - box_half,
        GOAL_BOX_WIDTH,
        GOAL_BOX_HEIGHT,
    )
    pygame.draw.rect(surface, LINE_WHITE, right_box, width=LINE_WIDTH)


# ---------------------------------------------------------------------------
# Game class -- owns all mutable state; extension point for entities/physics
# ---------------------------------------------------------------------------

class Game:
    """Top-level game state and per-frame update/draw orchestration.

    All mutable game state lives here (no module-level globals).
    Owns the :class:`~entities.Player`, :class:`~entities.Ball`,
    :class:`~physics.Score`, and :class:`~physics.PhysicsEngine`
    instances and drives their ``update``/``draw`` cycles each frame.
    """

    def __init__(self) -> None:
        """Initialise game state and instantiate core entities."""
        self.running: bool = True

        # --- Core entities ---
        self.player: Player = Player(color=PLAYER_COLOR_P1)
        self.ball: Ball = Ball()

        # --- Scoring & physics ---
        self.score: Score = Score()
        self.physics: PhysicsEngine = PhysicsEngine(
            ball=self.ball,
            player=self.player,
            score=self.score,
        )

        # --- Scoreboard font (created lazily so headless import stays safe) ---
        self.font: pygame.font.Font | None = self._make_font()

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _make_font() -> pygame.font.Font | None:
        """Create the scoreboard font, returning ``None`` if unavailable.

        Font creation requires ``pygame.font`` to be initialised.  In a
        headless test context (no ``pygame.init()``) this degrades
        gracefully to ``None`` and :meth:`draw` simply skips the scoreboard.
        """
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            return pygame.font.SysFont("arial", 32, bold=True)
        except Exception:  # pragma: no cover -- defensive headless fallback
            return None

    # -- Per-frame hooks -----------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance game logic by *dt* seconds.

        Parameters
        ----------
        dt:
            Elapsed time in seconds since the previous frame
            (``clock.tick(FPS) / 1000.0``).
        """
        self.player.update(dt)
        self.ball.update(dt)
        # Resolve ball/player collision and goal detection.
        self.physics.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        """Render the current game state onto *surface*.

        Parameters
        ----------
        surface:
            The target ``pygame.Surface``.
        """
        draw_field(surface)
        self.player.draw(surface)
        self.ball.draw(surface)
        if self.font is not None:
            self.score.draw(surface)

    # -- Event handling ------------------------------------------------------

    def handle_events(self) -> None:
        """Process pending pygame events.

        Sets ``self.running = False`` on ``QUIT`` or ``ESC``.
        Pressing ``R`` resets the ball, player, and score.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset()

    def reset(self) -> None:
        """Reset the ball, player, and score to their initial state."""
        self.ball.reset()
        self.player.position = self.ball.position.copy()
        self.player.velocity = pygame.math.Vector2(0.0, 0.0)
        self.player.direction = pygame.math.Vector2(0.0, 0.0)
        self.score.reset()

    def poll_input(self) -> None:
        """Read current key states and update the player's movement direction.

        Supports WASD and arrow keys.  Diagonal movement is normalised
        so it is not faster than axis-aligned movement.  Releasing all
        keys zeroes the direction (player stops).
        """
        keys = pygame.key.get_pressed()

        dx: float = 0.0
        dy: float = 0.0

        # Horizontal
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += 1.0

        # Vertical
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += 1.0

        self.player.set_direction(dx, dy)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Create the window, run the game loop, and clean up.

    This is the **only** place where ``pygame.display.set_mode`` is called,
    keeping the module import-safe for headless unit tests.
    """
    pygame.init()
    try:
        screen: pygame.Surface = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        pygame.display.set_caption(WINDOW_TITLE)

        clock = pygame.time.Clock()
        game = Game()

        while game.running:
            dt: float = clock.tick(FPS) / 1000.0  # seconds, capped at 60 FPS

            game.handle_events()
            game.poll_input()
            game.update(dt)
            game.draw(screen)

            pygame.display.flip()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
