import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
import main

surf = pygame.Surface((800, 600))
surf.fill((0, 0, 0))
main.draw_field(surf)


def px(x, y):
    return tuple(surf.get_at((x, y))[:3])


print("right edge scan y=300:")
for x in range(754, 766):
    print("  x=%d" % x, px(x, 300))
print("left edge scan y=300:")
for x in range(36, 46):
    print("  x=%d" % x, px(x, 300))
print("top edge scan x=400:")
for y in range(36, 46):
    print("  y=%d" % y, px(400, y))
print("bottom edge scan x=400:")
for y in range(556, 566):
    print("  y=%d" % y, px(400, y))
