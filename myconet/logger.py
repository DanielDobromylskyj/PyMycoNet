import builtins
import time
import os
import re


def remove_ansi_escape_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    GRAY = '\033[90m'
    WHITE = '\033[37m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class Logger:
    def __init__(self, log_level=0):
        self.colour_lookup = {
            "INFO": bcolors.BOLD,
            "WARNING": bcolors.WARNING + bcolors.BOLD,
            "ERROR": bcolors.FAIL + bcolors.BOLD,
            "DEBUG": bcolors.GRAY + bcolors.BOLD,
            "SOMETHING WRONG": bcolors.GRAY + bcolors.BOLD,
        }

        self.log_level = log_level

        if self.log_level == -1:
            return

        self.path = f"logs/{time.time()}.log"

        self.default_print = builtins.print
        self.enable()

        self.__prep_file()

    def __prep_file(self):
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

        open(self.path, "w").close()

    def __log(self, text):
        if self.log_level == -1:
            return

        self.default_print(text, end="")

        with open(self.path, "a") as f:
            f.write(remove_ansi_escape_codes(text))

    def __create_message(self, state, text):
        colour = self.colour_lookup[state]

        if state != "DEBUG":
            return f"{colour}[{state}]{bcolors.ENDC} {text}"

        return f"{colour}[{state}] {text}{bcolors.ENDC}"

    def disable(self):
        if self.log_level == -1:
            return

        builtins.print = self.default_print

    def enable(self):
        if self.log_level == -1:
            return

        builtins.print = self.print

    def print(self, *args, sep=" ", end="\n"):
        if self.log_level > 0:
            text = sep.join([str(arg) for arg in args]) + end
            self.__log(self.__create_message("INFO", text))

    def debug(self, *args, sep=" ", end="\n"):
        if self.log_level > 1:
            text = sep.join([str(arg) for arg in args]) + end
            self.__log(self.__create_message("DEBUG", text))

    def true_debug(self, *args, sep=" ", end="\n"):
        if self.log_level > 2:
            text = sep.join([str(arg) for arg in args]) + end
            self.__log(self.__create_message("SOMETHING WRONG", text))

    def close(self):
        if self.log_level == -1:
            return

        builtins.print = self.default_print
        self.log_level = -1




