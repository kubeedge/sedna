from abc import ABC, abstractmethod


class Transmitter(ABC):

    @abstractmethod
    def recv(self):
        pass

    @abstractmethod
    def send(self):
        pass

    @abstractmethod
    def compress(self):  # 传输的内容可能有：weights，压缩后的weights， 特征向量，蒸馏后的数据
        pass

    @abstractmethod
    def decompress(self):
        pass
