
#
#   Weather update server
#   Binds PUB socket to tcp://*:5556
#   Publishes random weather updates
#

import time
import zmq
from random import randrange

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

while True:
    zipcode = randrange(1, 100)
    temperature = randrange(-80, 135)
    relhumidity = randrange(10, 60)

    s = "%i %i %i" % (zipcode, temperature, relhumidity)
    print("sending: {0}".format(s))
    socket.send_string(s)
    print("sent")
    # time.sleep(5)
