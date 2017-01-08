import os
import pandas as pd
import numpy as np

from Terminal import Terminal


class DataFrameEditorTerminal(Terminal):
    def __init__(self):
        super().__init__()

        self.filepath = None
        self.df = None
        self.index = None

        self.initialize_commands()
        self.save_status = True
        
    def initialize_commands(self):
        self.commands = {}

        self.add_command("h", self.show_help, "Show help.")
        self.add_command("q", self.quit, "Quit.")
        self.add_command("f", self.process_filepath_input, "Change filepath to arg if given, else show current filepath.")
        self.add_command("s", self.change_save_status, "Turn on saving to file if arg is 1, else turn off if arg is 0.")
        self.add_command("i", self.process_index_input, "Change DataFrame index being edited if arg given, else show current.")
        self.add_command("c", self.print_columns, "Print sorted list of columns in the df.")
        self.add_command("nan", self.print_nan_columns, "Print columns of this row that are NaN.")

    def process_filepath_input(self, s=None):
        if s is None:
            return self.filepath

        if not s.endswith(".csv"):
            print("Only CSV is supported right now.")
            return

        if os.path.exists(s):
            try:
                df = pd.read_csv(s, index_col="Index")
            except Exception as e:
                print("Exception raised in reading this csv:")
                print(e.__class__.__name__, e)
                return
            self.filepath = s
            self.df = df
        else:
            self.filepath = s
            self.df = pd.DataFrame()

    def process_index_input(self, s=None):
        if s is None:
            return self.index
        if self.df is None:
            print("You have no df.")
            return

        if s not in self.df.index:
            print("That row does not exist. New row will be added when you populate a column.")

        self.index = s

    def change_save_status(self, a=None):
        return self.change_binary_attribute("save_status", a)

    def print_columns(self):
        if self.df is None:
            print("You have no df.")
            return

        for col in sorted(self.df.columns.values):
            print(col)

    def print_nan_columns(self):
        if self.df is None:
            print("You have no df.")
            return
        if self.index is None:
            print("You have no row selected.")
            return
        cols = [x for x in self.df.columns.values if np.isnan(self.df.loc[self.index, x])]
        for col in cols:
            print(col)

    def process_normal_input(self, s):
        if "=" not in s:
            if s in self.df.columns:
                return self.df.loc[self.index, s]
            else:
                print("Column not found. You can set new values in the form column_name=value")
                return
        try:
            col, val = s.split("=")
        except ValueError:
            print("input should be in the form column_name=value")
            return

        col = col.strip()
        if col not in self.df.columns:
            print("Column not found. It will be created when you set a value.")
        try:
            val = float(val)
        except ValueError:
            print("invalid value for float: {}".format(val))
            return

        self.df.loc[self.index, col] = val
        self.save()

    def save(self):
        self.df = self.df.set_index(self.df.index.rename("Index"))
        self.df.to_csv(self.filepath)


if __name__ == "__main__":
    terminal = DataFrameEditorTerminal()
    terminal.run()