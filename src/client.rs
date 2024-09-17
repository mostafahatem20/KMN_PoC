use proto::key_management_client::KeyManagementClient;
use tonic::transport::Channel;
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

    // Assuming there's at least one URL for the first client
    let channel = Channel::builder(urls[0].clone().parse().unwrap()).connect().await.unwrap();
    let limit = 20 * 1024 * 1024;
    let mut client1 = KeyManagementClient::new(channel).max_decoding_message_size(limit);
    let response = get_keys_request(&mut client1).await?;
    println!("{:?}", response.keys);

    // Convert the response into a HashMap with (key_id, pub_key) tuples as the key
    let mut keys_map: HashMap<(String, String), Option<String>> = HashMap::new();
    for key_info in response.keys {
        keys_map.insert((key_info.key_id.clone(), key_info.pub_key.clone()), None);
    }

    // Start the client-server with URLs from the config and the new keys structure
    let _ = client_server::start_client_server(
        port,
        keys_map,
        urls.clone() // Use all URLs
    ).await;

    Ok(())
}

async fn get_keys_request(
    client: &mut KeyManagementClient<Channel>
) -> Result<proto::GetKeysResponse, Box<dyn Error>> {
    let request = tonic::Request::new(proto::GetKeysRequest {});
    let response = client.get_keys(request).await?;
    Ok(response.into_inner())
}
