import sys
import random
import math
import time
import os
import getopt
import pygame
# from socket import *
from pygame.locals import *

# def load_png(name):
#     """ Load image and return image object"""
#     # fullname = os.path.join('data', name)
#     fullname = os.path.join('', name)
#     image = pygame.image.load(fullname)
#     if image.get_alpha is None:
#         image = image.convert()
#     else:
#         image = image.convert_alpha()
#     return image, image.get_rect()

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
        self.kills = 0
        self.infections = 0
        self.start_time = time.time()

        self.reinit()

    def reinit(self):
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


class Monster(pygame.sprite.Sprite):
    def __init__(self, speed):
        pygame.sprite.Sprite.__init__(self)
        # self.image, self.rect = load_png('ball.png')
        self.image = pygame.Surface([10, 10])
        self.image.fill(pygame.Color(255, 255, 0, 255))
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.rect = self.image.get_rect()
        self.rect = self.rect.move(random.uniform(0, self.area.width), random.uniform(0, self.area.height))
        self.vector = (0, speed)
        self.hit = 0
        self.queue = []

    def update(self, player, monster):
        newpos = self.calcnewpos(self.rect,self.vector)
        self.rect = newpos
        (angle,z) = self.vector

        victim = self.queue[-1]

        if not self.area.contains(newpos):
            tl = not self.area.collidepoint(newpos.topleft)
            tr = not self.area.collidepoint(newpos.topright)
            bl = not self.area.collidepoint(newpos.bottomleft)
            br = not self.area.collidepoint(newpos.bottomright)
            if tr and tl or (br and bl):
                angle = -angle
            if tl and bl:
                #self.offcourt()
                angle = math.pi - angle
            if tr and br:
                angle = math.pi - angle
                #self.offcourt()
        else:
            # if self.rect.colliderect(player.rect) == 1 and not self.hit:
                # angle = math.pi - angle
                # self.hit = not self.hit
            # elif self.hit:
                # self.hit = not self.hit
            dx_to_player = victim.rect[0] - self.rect[0]
            dy_to_player = victim.rect[1] - self.rect[1]
            if dx_to_player == 0:
                if dy_to_player < 0:
                    angle = -1*math.pi/2.0
                elif dy_to_player > 0:
                    angle = math.pi/2.0
                else:
                    victim.kill()
                    player.kills += 1
                    self.queue.pop()

            else:
                angle = math.atan(dy_to_player * 1.0/dx_to_player) + (0 if dx_to_player >= 0 else math.pi)

        self.vector = (angle, z)

        if self.distance(victim) <= 50:
            self.image.fill(pygame.Color(255, 0, 0, 255))
        elif self.distance(player) <= 50:
            self.image.fill(pygame.Color(255, 128, 0, 255))
        else:
            self.image.fill(pygame.Color(255, 255, 0, 255))

        if self.distance(player) <= 10:
            player.kill()


    def calcnewpos(self, rect, vector):
        (angle,z) = vector
        (dx,dy) = (z*math.cos(angle),z*math.sin(angle))
        return rect.move(round(dx),round(dy))

    def distance(self, player):
        if self.rect.colliderect(player.rect) == 1:
            return 0
        x1, y1 = self.rect[0], self.rect[1]
        x2, y2 = player.rect[0], player.rect[1]
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def add_victim(self, person):
        if person in self.queue:
            i = self.queue.index(person)
            self.queue.pop(i)
            self.queue.append(person)
        else:
            self.queue.append(person)


class Person(pygame.sprite.Sprite):
    def __init__(self, vector):
        pygame.sprite.Sprite.__init__(self)
        # self.image, self.rect = load_png('ball.png')
        self.image = pygame.Surface([10, 10])
        self.image.fill(pygame.Color(255, 255, 0, 255))
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.rect = self.image.get_rect()
        self.rect = self.rect.move(random.uniform(0, self.area.width), random.uniform(0, self.area.height))
        self.vector = vector
        self.hit = 0
        self.dead = False

    def update(self, player, monster):
        newpos = self.calcnewpos(self.rect,self.vector)
        self.rect = newpos
        (angle,z) = self.vector

        if not self.area.contains(newpos):
            tl = not self.area.collidepoint(newpos.topleft)
            tr = not self.area.collidepoint(newpos.topright)
            bl = not self.area.collidepoint(newpos.bottomleft)
            br = not self.area.collidepoint(newpos.bottomright)
            if tr and tl or (br and bl):
                angle = -angle
            if tl and bl:
                #self.offcourt()
                angle = math.pi - angle
            if tr and br:
                angle = math.pi - angle
                #self.offcourt()

        if random.random() < 0.01:
            angle = random.uniform(0, 2*math.pi)

        self.vector = (angle, z)

        if self.distance(player) <= 10 and not self.dead and self not in monster.queue:
            monster.add_victim(self)
            player.infections += 1
            self.image.fill(Color(128, 128, 255, 255))

    def calcnewpos(self, rect, vector):
        (angle,z) = vector
        (dx,dy) = (z*math.cos(angle),z*math.sin(angle))
        return rect.move(dx,dy)

    def distance(self, player):
        if self.rect.colliderect(player.rect) == 1:
            return 0
        x1, y1 = self.rect[0], self.rect[1]
        x2, y2 = player.rect[0], player.rect[1]
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def kill(self):
        self.image.fill(Color(128, 128, 128, 255))
        self.rect.move_ip(-10**6, -10**6) # (-100-self.rect[0], -100-self.rect[1])
        self.vector = (0, 0)
        self.dead = True


def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 400))

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0, 0, 0))

    player1 = Player(3)

    speed = 1.6
    balls = []
    monster = Monster(speed*1.1)
    monster.add_victim(player1)
    balls.append(monster)
    for i in range(12):
        person = Person((random.uniform(0, 2*math.pi),speed))
        balls.append(person)


    playersprites = pygame.sprite.RenderPlain((player1))
    ballsprites = pygame.sprite.RenderPlain(balls)

    screen.blit(background, (0, 0))
    pygame.display.flip()

    clock = pygame.time.Clock()

    while True:
        clock.tick(60) # fps

        for event in pygame.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN: # key pressed
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

        for ball in balls:
            screen.blit(background, ball.rect, ball.rect)
            screen.blit(background, player1.rect, player1.rect)
        ballsprites.update(player1, monster)
        playersprites.update()
        ballsprites.draw(screen)
        playersprites.draw(screen)
        pygame.display.flip()


main()















