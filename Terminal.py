class Terminal:
    def __init__(self):
        self.prompt = "$ "
        self.command_char = ":"

    def get_input(self):
        return input(self.prompt)

    def process_input(self, user_input):
        if len(user_input) > 0 and user_input[0] == ":":
            return self.process_command(user_input[1:])
        else:
            return self.process_normal_input(user_input)

    def process_normal_input(self, user_input):
        pass

    def display_output(self, output):
        if output is not None:
            print(output)

    def initialize_commands(self):
        self.commands = {}

        self.add_command("h", self.show_help, "Show help.")
        self.add_command("q", self.quit, "Quit.")

    def add_command(self, flag, func, help):
        if flag in self.commands:
            raise Exception("tried to add already-existing command {}".format(flag))
        self.commands[flag] = TerminalCommand(flag, func, help)

    def process_command(self, command):
        s = command.split(" ")
        flag = s[0]
        args = s[1:] if len(s) > 0 else []

        if flag in self.commands:
            return self.commands[flag](*args)
        else:
            return "unrecognized command"


    def show_help(self):
        for flag, command in sorted(self.commands.items()):
            print("{:4s} {1}".format(flag, command.help))
        print()

    def quit(self):
        raise SystemExit

    def change_binary_attribute(self, attr, a=None):
        if a is None:
            return True if getattr(self, attr) else False
        a = int(a)
        if a == 0:
            setattr(self, attr, False)
        elif a == 1:
            setattr(self, attr, True)
        else:
            return "Please specify 0 or 1."

    def run(self):
        while True:
            try:
                user_input = self.get_input()
                output = self.process_input(user_input)
                self.display_output(output)
            except KeyboardInterrupt:
                print("\nInterrupted. Enter :q to quit.")
                continue
            except EOFError:
                print("EOF. Enter :q to quit.")
                continue


class TerminalCommand:
    def __init__(self, flag, func, help):
        self.flag = flag
        self.func = func
        self.help = help

    def __call__(self, *args):
        return self.func(*args)