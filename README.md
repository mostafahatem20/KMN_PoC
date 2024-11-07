# Node Binary

This binary executes the `main.rs` file and is used to pre-generate keys, pre-signatures, or run related tests.

### Usage

```bash
cargo run --bin node <global_config_file_path> <config_file_path> <operation> <args>
```

### Example

To run 3 nodes and generate 100 keys each, use the following commands:

```bash
cargo run --bin node configs/global_config.yml configs/config0.yml KEY_GEN 100
cargo run --bin node configs/global_config.yml configs/config1.yml KEY_GEN 100
cargo run --bin node configs/global_config.yml configs/config2.yml KEY_GEN 100
```

---

# Server Binary

This binary runs the `server.rs` file, starting a node service that handles requests defined in `server.proto`.

### Usage

```bash
cargo run --bin server <global_config_file_path> <config_file_path>
```

---

# Client Binary

This binary runs the `client.rs` file, starting a client server that processes requests defined in `client.proto`. It communicates with nodes started by the `server` binary to aggregate results. This binary acts as the main entry point for receiving requests from the CBDC system.

### Usage

```bash
cargo run --bin client <global_config_file_path>
```
