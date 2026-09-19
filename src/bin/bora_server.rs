//! Philote discipline server exposing NeuralFoil analyses over gRPC.
//!
//! Usage: `bora_server [address]` (default `localhost:50051`)

use std::process::ExitCode;
use std::sync::Arc;

use bora::philote::BoraDiscipline;
use philote_mdo::philote_info::{
    discipline_service_server::DisciplineServiceServer,
    explicit_service_server::ExplicitServiceServer,
};
use philote_mdo::server::ExplicitServer;
use tonic::transport::Server;

async fn serve(address: &str) -> Result<(), Box<dyn std::error::Error>> {
    let socket = tokio::net::lookup_host(address)
        .await?
        .next()
        .ok_or_else(|| format!("cannot resolve address {address}"))?;
    let server = Arc::new(ExplicitServer::new(BoraDiscipline::new()));

    println!("Bora discipline server listening on {address}");
    Server::builder()
        .add_service(DisciplineServiceServer::from_arc(server.clone()))
        .add_service(ExplicitServiceServer::from_arc(server))
        .serve(socket)
        .await?;
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    let address = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "localhost:50051".to_string());
    match serve(&address).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("Bora discipline server failed: {err}");
            ExitCode::FAILURE
        }
    }
}
