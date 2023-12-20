from __future__ import print_function
import logging, configparser, grpc
import autocomp_pb2_grpc, autocomp_pb2


def run():
    print("Connecting to server ...")
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    pi_host = config['SERVER']['Host']
    port = config['SERVER']['Port']

    with grpc.insecure_channel(f"{pi_host}:{port}") as channel:
        stub = autocomp_pb2_grpc.AutoCompleteStub(channel)
        msg = input('>')
        while msg:
            response = stub.FetchNext(autocomp_pb2.InferenceRequest(input=msg))
            print(response.output)
            msg = input('>')


if __name__ == "__main__":
    logging.basicConfig()
    run()