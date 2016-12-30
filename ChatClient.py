import time
import zmq


class ChatClient:
    def __init__(self, context, ip, port):
        self.socket = context.socket(zmq.REQ)
        self.socket.RCVTIMEO = 100000
        self.socket.connect("tcp://{0}:{1}".format(ip, port))
        self.client_id = self.get_client_id_from_user()

    def get_client_id_from_user(self):
        client_id = input("Your ID for this session (alphanumeric): ").strip()
        if not client_id.isalnum():
            raise ValueError("invalid client id; is not alphanumeric")
        return client_id

    def flush_new_messages(self):
        flush_request = "flush::"
        self.send_message(flush_request)
        replies = self.receive_replies()
        return replies

    def get_message_to_send(self):
        return input("Request to send to server: ")

    def send_message(self, message):
        string_to_send = "client_id={0}>>{1}".format(self.client_id, message)
        self.socket.send_string(string_to_send)

    def receive_replies(self):
        print("waiting to receive reply ...")
        received = self.socket.recv().decode("utf-8")
        header, rest = received.split(">>")
        sub_messages = rest.split(";;")
        return sub_messages

    def run_request_reply(self):
        request = self.get_message_to_send()
        print("sending request:", request)
        self.send_message(request)
        replies = self.receive_replies()
        for reply in replies:
            print("received:", reply)
        time.sleep(0.1)

    def run_flush(self):
        flush_replies = self.flush_new_messages()
        for reply in flush_replies:
            print("flush received:", reply)
        time.sleep(0.1)

    def run(self):
        while True:
            self.run_request_reply()
            # self.run_flush()


def run_client(context, ip, port):
    client = ChatClient(context, ip, port)
    client.run()


if __name__ == "__main__":
    localhost_ip = "127.0.0.1"
    wesley_private_ip = "192.168.1.7"  # from `ipconfig /all`
    wesley_public_ip = "71.57.35.2"  # from googling "what is my ip"

    context = zmq.Context()

    server_port = "5000"

    run_client(context, wesley_public_ip, server_port)