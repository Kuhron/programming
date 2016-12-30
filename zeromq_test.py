# source: http://stackoverflow.com/questions/6920858/

import argparse
import zmq

# parser = argparse.ArgumentParser(description='zeromq server/client')
# parser.add_argument('--bar')
# args = parser.parse_args()

# if args.bar:
#     # client
#     context = zmq.Context()
#     socket = context.socket(zmq.PAIR) # originally zmq.REQ
#     socket.connect('tcp://127.0.0.1:5555')
    # socket.send_string(args.bar)
    # msg = socket.recv()
    # msg = msg.decode("utf-8")
    # print(msg)
# else:
#     # server
#     context = zmq.Context()
#     socket = context.socket(zmq.PAIR) # originally zmq.REP
#     socket.bind('tcp://127.0.0.1:5555') # zmq doesn't really care much about bind vs. connect, according to this: http://stackoverflow.com/questions/16109139/
    # while True:
    #     msg = socket.recv()
    #     msg = msg.decode("utf-8") # convert bytes string to normal string
    #     #socket.send_string("msg = {0}".format(msg)) # getting error by sending multiple strings before REQ socket has made next request
    #     if msg == 'zeromq':
    #         socket.send_string('ah ha!')
    #     else:
    #         socket.send_string('...nah')

# now just try to get two identical processes talking to each other (taking turns counting back and forth to 100 in this example)
parser = argparse.ArgumentParser(description='zeromq server/client')
parser.add_argument("--position")
args = parser.parse_args()

if not args.position:
    raise TypeError("Missing --position argument; please specify 'first' or 'second'.")
elif args.position not in ["first","second"]:
    raise ValueError("--position argument must specify 'first' or 'second'.")

context = zmq.Context()
socket = context.socket(zmq.PAIR)
if args.position == "first":
    # bind the thing that is sort of there more permanently, like the server
    # since the first bot is sending the messages first and also sending the last message in this case, it kind of makes sense to bind it
    # however, 
    socket.connect("tcp://127.0.0.1:5555")
else:
    socket.bind("tcp://127.0.0.1:5555")

i = 1
max_count = 10**4

if args.position == "first": # the "first" process sends the first number
    # print("sending first number")
    socket.send_string(str(i))
    # print("sent first number successfully")

while True:
    #if i >= max_count:
        #break
    # print("waiting to receive number")
    a = socket.recv().decode("utf-8")
    try:
        i = int(a) + 1
    except ValueError: # got a message instead of a number, used to signal the first bot that it has been thanked, so counting is over
        # print("received message instead of number:",a)
        break
    print("number received:",i)
    if i >= max_count:
        break
    socket.send_string(str(i))
if args.position == "first":
    # wait_time = 1 # second
    # t0 = time.time()
    # while time.time() <  # this doesn't work because it will wait for the message forever
    # know that the first bot will receive its congratulation first
    # a = socket.recv().decode("utf-8") # so don't need this or it will wait for a message that never comes
    print("received:", a) # see if a makes it into this scope # WORKS!
    print("sending thank you message to second bot")
    socket.send_string("Good job, second bot!")
    print("sent successfully")
else:
    print("sending thank you message to first bot")
    socket.send_string("Good job, first bot!")
    print("sent successfully")
    a = socket.recv().decode("utf-8")
    print("received:", a)

