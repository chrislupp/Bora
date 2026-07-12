/// Philote discipline server exposing NeuralFoil analyses over gRPC.
///
/// Usage: bora_server [address]  (default localhost:50051)

#include "bora/philote_discipline.h"

#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

int main(int argc, char* argv[]) try {
    std::string address = "localhost:50051";
    if (argc > 1) {
        address = argv[1];
    }

    auto discipline = std::make_shared<bora::BoraDiscipline>();

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    discipline->RegisterServices(builder);

    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server) {
        std::cerr << "Failed to start Bora discipline server on " << address
                  << '\n';
        return 1;
    }
    std::cout << "Bora discipline server listening on " << address << '\n';
    server->Wait();

    return 0;
} catch (const std::exception& e) {
    std::cerr << "Bora discipline server failed: " << e.what() << '\n';
    return 1;
}
