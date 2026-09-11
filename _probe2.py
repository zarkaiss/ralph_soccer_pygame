import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

print("display init before:", pygame.display.get_init())

# Surface creation without init?
try:
    s = pygame.Surface((10, 10))
    print("Surface OK, get_at:", tuple(s.get_at((0, 0))[:3]))
except Exception as e:
    print("Surface FAIL:", repr(e))

# image.tostring without init?
try:
    s = pygame.Surface((4, 4))
    s.fill((1, 2, 3))
    raw = pygame.image.tostring(s, "RGB")
    print("tostring OK, len:", len(raw))
except Exception as e:
    print("tostring FAIL:", repr(e))

# event.post / get without init?
try:
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    evs = pygame.event.get()
    print("event OK, got:", [e.type for e in evs])
except Exception as e:
    print("event FAIL:", repr(e))

print("display init after (no explicit init):", pygame.display.get_init())
