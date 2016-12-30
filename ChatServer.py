import random
import time
import zmq

from Exchange.Exchange import Exchange


class MessageRouter:
    def __init__(self):
        self.chat = Chat()
        self.exchange = Exchange()
        self.story = Story()
        self.flush = {}

        self.default_identifier = "chat"

        self.identifier_to_destination = {
            "flush": None,
            "exit": None,
            "chat": self.chat,
            "exchange": self.exchange,
            "story": self.story,
        }

    def process_incoming_message(self, message):
        header, rest = message.split(">>")
        client_id = header.split("=")[1]
        sub_messages = rest.split(";;")

        replies = []

        for sub_message in sub_messages:
            message_split = sub_message.split("::")

            if len(message_split) == 1:
                identifier = self.default_identifier
                message_to_destination = sub_message
            else:
                identifier = message_split[0]
                message_to_destination = message_split[1]

            if identifier not in self.identifier_to_destination:
                identifier = self.default_identifier

            if identifier == "flush":
                print("flushing for client", client_id)
                flush_replies = self.get_flush_replies(client_id)
                replies += flush_replies
            else:
                print("current identifier:", identifier)
                destination = self.identifier_to_destination[identifier]
                reply = identifier + "::" + destination.process_incoming_message(message_to_destination)
                replies.append(reply)

        return header + ">>" + ";;".join(replies)

    def get_flush_replies(self, client_id):
        replies = self.flush.get(client_id, [])
        return ["flush::" + reply for reply in replies]


class Chat:
    def __init__(self):
        pass

    def process_incoming_message(self, message):
        print("received:", message)
        reply = input("Reply to send to client: ")
        return reply


class Story:
    def __init__(self):
        self.string = ""

    def process_incoming_message(self, message):
        self.string += message
        return self.string


def run_server(context, ip, port, message_router):
    socket = context.socket(zmq.REP)
    socket.bind("tcp://{0}:{1}".format(ip, port))
    while True:
        print("waiting to receive request ...")
        received = socket.recv().decode("utf-8")
        reply = message_router.process_incoming_message(received)
        print("sending reply:", reply)
        socket.send_string(reply)


if __name__ == "__main__":
    localhost_ip = "127.0.0.1"
    wesley_private_ip = "192.168.1.7"  # from `ipconfig /all`
    wesley_public_ip = "71.57.35.2"  # from googling "what is my ip"

    context = zmq.Context()

    server_port = "5000"

    message_router = MessageRouter()
    run_server(context, wesley_private_ip, server_port, message_router)