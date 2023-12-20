from concurrent import futures
import logging

import grpc
import autocomp_pb2_grpc, autocomp_pb2



class AutoComplete(autocomp_pb2_grpc.AutoCompleteServicer):
    def FetchNext(self, request, context):
        return autocomp_pb2.InferenceResponse(output="%s, back at you!" % request.input)


def serve():
    host = "0.0.0.0"
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    autocomp_pb2_grpc.add_AutoCompleteServicer_to_server(AutoComplete(), server)
    server.add_insecure_port(host + ":" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()