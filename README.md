cargo run --bin node configs/global_config.yml configs/config0.yml KEY_GEN 100
cargo run --bin node configs/global_config.yml configs/config0.yml PRE_SIGN
cargo run --bin server configs/global_config.yml configs/config0.yml
cargo run --bin client configs/global_config.yml
