import datetime

def create_block_list_from_file(filepath):
    f = open(filepath,"r")
    f_lines = f.readlines()
    f.close()

    blocks = [[]]
    for line in f_lines:
        if line == "\n":
            blocks.append([])
        else:
            blocks[-1].append(line.strip())

    block_list = []
    for block_text in blocks:
        block_object = Block(block_text)
        block_list.append(block_object)

    return block_list

def unix_time_to_datetime(timestamp):
    timestamp = int(timestamp)
    return datetime.datetime.fromtimestamp(timestamp)

def unix_time_to_string(timestamp):
    return unix_time_to_datetime(timestamp).strftime("%Y-%m-%d-%H:%M:%S")

def datetime_to_string(dt):
    return dt.strftime("%Y-%m-%d-%H:%M:%S")

def split_block_information(block_text):
    return block_text[0],block_text[1],block_text[2:]

class Block:
    def __init__(self,block_text):
        self.block_text = block_text
        location,designation,message_text = split_block_information(block_text)
        self.location = location
        self.designation = designation
        try:
            self.datetime = unix_time_to_datetime(designation)
            self.time = datetime_to_string(self.datetime)
        except ValueError:
            self.datetime = None
            self.time = None

        self.message_text = message_text
        self.message = Message(message_text)

class Message:
    def __init__(self,message_text):
        self.message_text = message_text
        self.body = message_text[:-1]
        self.footer = message_text[-1]

        self.alphabet_04 = self.get_alphabet([0,4])
        self.alphabet_15 = self.get_alphabet([1,5])
        self.alphabet_26 = self.get_alphabet([2,6])
        self.alphabet_37 = self.get_alphabet([3,7])


    def footer_is_different(self):
        return len(self.footer) < 8 or "=" in self.footer or (not self.is_encrypted)

    def is_encrypted(self):
        return " " not in self.message_text[0] and len(self.message_text[0]) == 8

    def get_alphabet(self,indices):
        if not self.is_encrypted():
            return []

        a = indices[0]
        b = indices[1]
        alphabet = set()
        for line in self.body:
            alphabet.add(line[a])
            alphabet.add(line[b])
        try:
            if self.footer[a] != "=":
                alphabet.add(self.footer[a])
            if self.footer[b] != "=":
                alphabet.add(self.footer[b])
        except IndexError:
            pass

        return sorted(alphabet)

class Cipher:
    def __init__(self,filepath):
        self.filepath = filepath
        self.block_list = create_block_list_from_file(filepath)

    def get_alphabets(self,indices):
        result = set()
        for block in self.block_list:
            result.add(tuple(block.message.get_alphabet(indices)))
        return sorted([sorted(list(i)) for i in result])

    def print_alphabets(self,indices):
        for i in self.get_alphabets(indices):
            print(i)



C = Cipher("f04cb41f154db2f05a4a.txt")

#for i in range(len(block_list)):
    #print(block_list[i].message.alphabet_15)

C.print_alphabets([1,5])







