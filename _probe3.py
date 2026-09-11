import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

print("before init, display.get_init():", pygame.display.get_init())
pygame.init()
print("after init, display.get_init():", pygame.display.get_init())

pygame.event.clear()
pygame.event.post(pygame.event.Event(pygame.QUIT))
evs = pygame.event.get()
print("events:", [e.type for e in evs])

# KEYDOWN
pygame.event.clear()
pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
evs = pygame.event.get()
print("key events:", [(e.type, e.key) for e in evs])
