use std::error::Error;
use kmn_poc::{ set_number_of_parties, set_threshold, utils::client_server };
use kmn_poc::utils::config;
use std::collections::HashMap;

pub mod proto {
    tonic::include_proto!("kmn_poc");
}
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_global_config>", args[0]);
        std::process::exit(1);
    }

    // Read setup config from the file
    let setup_path = args[1].clone();
    let setup_config = config::read_setup_config(&setup_path).unwrap();

    let port = setup_config.port.ok_or("Port is not provided")?;

    // Set threshold and number of parties
    set_threshold(setup_config.threshold);
    set_number_of_parties(setup_config.number_of_parties);

    // Use the URLs from the config
    let urls = setup_config.urls.ok_or("Urls are not provided")?;

    let keys_map: HashMap<(String, String), Option<String>> = HashMap::new();
    
    let _ = client_server::start_client_server(
        port,
        keys_map,
        urls.clone() // Use all URLs
    ).await;

    Ok(())
}