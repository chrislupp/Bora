/*
    Bora: Rust Port of NeuralFoil

    Copyright 2026 Christopher A. Lupp

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    This work has been cleared for public release, distribution unlimited, case
    number: AFRL-2026-2087.

    The views expressed are those of the authors and do not reflect the
    official guidance or position of the United States Government, the
    Department of Defense or of the United States Air Force.

    Statement from DoD: The Appearance of external hyperlinks does not
    constitute endorsement by the United States Department of Defense (DoD) of
    the linked websites, of the information, products, or services contained
    therein. The DoD does not exercise any editorial, security, or other
    control over the information you may find at these locations.
*/

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
