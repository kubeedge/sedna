from plato.servers import mistnet

from .base import BaseAggregation


class MistNet(BaseAggregation):

    def aggregate(self, weights, size=0):
        # do aggregate using the function provide by plato.
        mistnet.Server.aggregate_weights()

        # the following function
        # mistnet.Server.send()
        # mistnet.Server.run()
        # mistnet.Server.start()

    def exit_check(self):
        # do exit_check using the function provide by plato
        pass

    def client_choose(self):
        # do client_choose using the function provide by plato
        mistnet.Server.choose_clients()
        pass
