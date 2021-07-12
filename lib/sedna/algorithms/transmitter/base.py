from abc import ABC, abstractmethod


class Transmitter(ABC):

    @abstractmethod
    def recv(self):
        pass

    @abstractmethod
    def send(self):
        pass

    @abstractmethod
    def compress(self):  # for compressing weights, feature data, data after distillation
        pass

    @abstractmethod
    def decompress(self):
        pass
