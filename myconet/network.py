from .logger import Logger


class Network:
    def __init__(self, layout):
        self.__layout = layout

        self.__log = Logger()
        self.__log.log("Initializing network")