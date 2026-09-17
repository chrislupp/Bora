mod common;

use bora::{InputDistribution, Network};
use common::WEIGHTS_DATA_DIR;

fn weights_path(name: &str) -> String {
    format!("{WEIGHTS_DATA_DIR}/{name}")
}

#[test]
fn network_from_bytes_matches_load() {
    let path = weights_path("nn-xxsmall.h5");
    let from_disk = Network::load(&path).unwrap();
    let from_memory = Network::from_bytes(&std::fs::read(&path).unwrap()).unwrap();

    assert_eq!(from_memory.layers(), from_disk.layers());
    assert_eq!(from_disk.input_size(), 25);
    assert_eq!(from_disk.output_size(), 198);
}

#[test]
fn network_from_bytes_rejects_garbage() {
    assert!(Network::from_bytes(&[0x42; 256]).is_err());
}

#[test]
fn input_distribution_from_bytes_matches_load() {
    let path = weights_path("scaled_input_distribution.h5");
    let from_disk = InputDistribution::load(&path).unwrap();
    let from_memory = InputDistribution::from_bytes(&std::fs::read(&path).unwrap()).unwrap();

    assert_eq!(from_memory.size(), from_disk.size());
    let x = vec![0.5; from_disk.size()];
    assert_eq!(
        from_memory.squared_mahalanobis(&x),
        from_disk.squared_mahalanobis(&x)
    );
}

#[test]
fn input_distribution_load_rejects_network_file() {
    assert!(InputDistribution::load(weights_path("nn-xxsmall.h5")).is_err());
}
