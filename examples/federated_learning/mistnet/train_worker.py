from sedna.core.federated_learning import MistWorker
import asyncio

if __name__ == '__main__':
    client = MistWorker()
    client.configure()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.start_client())