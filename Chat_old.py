import argparse
import random
import time
import zmq


# def poll_socket(socket, timetick = 100):
#     raise Exception("do not use")
#     poller = zmq.Poller()
#     poller.register(socket, zmq.POLLIN)
#     try:
#         obj = dict(poller.poll(timetick))
#         if socket in obj and obj[socket] == zmq.POLLIN:
#             yield socket.recv()
#     except KeyboardInterrupt:
#         pass


def run_server(context, ip, port):
    socket = context.socket(zmq.REP)
    # socket.RCVTIMEO = 1000
    # del ip
    # ip = "*"
    # ip = "0.0.0.0"
    socket.bind("tcp://{0}:{1}".format(ip, port))
    while True:
        print("waiting to receive request ...")
        received = socket.recv()
        print("received:", received)
        reply = str(random.random())
        socket.send_string(reply)


def run_client(context, ip, port):
    socket = context.socket(zmq.REQ)
    socket.RCVTIMEO = 100000
    socket.connect("tcp://{0}:{1}".format(ip, port))

    while True:
        request = str(random.random())
        socket.send_string(request)
        print("waiting to receive reply ...")
        try:
            received = socket.recv()
            print("received:", received)
        except:
            raise RuntimeError("server failed to reply")
        time.sleep(1)


tessa_ip = "73.177.69.155"  # gotten by googling "what is my ip" which doesn't actually work
localhost_ip = "127.0.0.1"
wesley_ip = "192.168.1.7"  # from `ipconfig /all`
ip_to_use = wesley_ip

context = zmq.Context()

parser = argparse.ArgumentParser()
parser.add_argument("--server", action="store_true")
args = parser.parse_args()

server_port = "5000"

if args.server:
    run_server(context, ip_to_use, server_port)
else:
    run_client(context, ip_to_use, server_port)

