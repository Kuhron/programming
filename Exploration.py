import math
import random

import pygame
from pygame.locals import *


class Player(pygame.sprite.Sprite):
    def __init__(self, speed):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([10, 10])
        self.image.fill(pygame.Color(128, 255, 128, 255))
        self.rect = self.image.get_rect()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.speed = speed
        self.state = "still"
        self.movepos = [0, 0]
        self.rect.center = self.area.center

    def update(self):
        newpos = self.rect.move(self.movepos)
        if self.area.contains(newpos):
            self.rect = newpos
        pygame.event.pump()

    def moveup(self):
        self.movepos[1] = self.movepos[1] - (self.speed)
        self.state = "moveup"

    def movedown(self):
        self.movepos[1] = self.movepos[1] + (self.speed)
        self.state = "movedown"

    def moveleft(self):
        self.movepos[0] = self.movepos[0] - (self.speed)
        self.state = "moveleft"

    def moveright(self):
        self.movepos[0] = self.movepos[0] + (self.speed)
        self.state = "moveright"

    def kill(self):
        print("You were killed!\nYou survived {0:.0f} seconds and transmitted It {1} times, resulting in {2} other deaths.".format(
            time.time() - self.start_time, self.infections, self.kills))
        sys.exit()


class Block:
    def __init__(self, x_index, y_index, screen_x_size, screen_y_size):
        self.x_index = x_index
        self.y_index = y_index
        self.topleft = (25*x_index, 25*y_index)


class House(pygame.sprite.Sprite):
    def __init__(self, block):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([20, 20])
        self.image.fill(pygame.Color(255, 255, 0, 255))
        self.rect = self.image.get_rect()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.block = block
        self.rect.topleft = [i + 2 for i in self.block.topleft]


class Road(pygame.sprite.Sprite):
    def __init__(self, block, directions):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([25, 25])
        self.square = pygame.Surface([15, 15])
        self.square.fill(pygame.Color(255, 0, 255, 255))
        self.image.blit(self.square, (5, 5), None, 0)
        if directions[0]:
            self.image.blit(self.square, (5, 0), None, 0)
        if directions[1]:
            self.image.blit(self.square, (10, 5), None, 0)
        if directions[2]:
            self.image.blit(self.square, (5, 10), None, 0)
        if directions[3]:
            self.image.blit(self.square, (0, 5), None, 0)
        self.rect = self.image.get_rect()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.block = block
        self.rect.topleft = [i for i in self.block.topleft]


def tf():
    return random.choice([True, False])

def main():
    pygame.init()
    x_size, y_size = 600, 400
    screen = pygame.display.set_mode((x_size, y_size))

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0, 0, 0))

    player1 = Player(3)

    houses = []
    road_blocks = []
    roads = []
    for i in range(10):
        house_block = Block(random.choice(range(int(x_size/25))), random.choice(range(int(y_size/25))), x_size, y_size)
        road_block = Block(house_block.x_index, house_block.y_index + 1, x_size, y_size)
        houses.append(House(house_block))
        road_blocks.append(road_block)

    for road_block in road_blocks:
        roads.append(Road(road_block, [True, tf(), tf(), tf()]))

    playersprites = pygame.sprite.RenderPlain((player1))
    house_sprites = pygame.sprite.RenderPlain(houses)
    road_sprites = pygame.sprite.RenderPlain(roads)

    screen.blit(background, (0, 0))
    pygame.display.flip()

    clock = pygame.time.Clock()

    while True:
        clock.tick(60) # fps

        for event in pygame.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN: # key pressed
                if event.key == K_RETURN:
                    return
                if event.key == K_UP:
                    player1.moveup()
                if event.key == K_DOWN:
                    player1.movedown()
                if event.key == K_LEFT:
                    player1.moveleft()
                if event.key == K_RIGHT:
                    player1.moveright()
            elif event.type == KEYUP: # key lifted
                if event.key in [K_UP, K_DOWN, K_LEFT, K_RIGHT]:
                    player1.movepos = [0,0]
                    player1.state = "still"

        for house in houses:
            screen.blit(background, house.rect, house.rect)
        screen.blit(background, player1.rect, player1.rect)
        # house_sprites.update(player1, monster)
        playersprites.update()
        house_sprites.draw(screen)
        road_sprites.draw(screen)
        playersprites.draw(screen)
        pygame.display.flip()


main()















