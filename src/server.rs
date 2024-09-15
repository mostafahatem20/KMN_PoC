use std::env;
use anyhow::Error;
use kmn_poc::{
    network::node_server, set_database_url, set_index, set_topic, utils::config::{ self, create_postgres_url }
};

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <path_to_global_config> <path_to_local_config>", args[0]);
        std::process::exit(1);
    }
    let setup_path = args[1].clone();
    let path = args[2].clone();
    let config = config::read_config(&path).unwrap();
    let index = config.index;
    let setup_config = config::read_setup_config(&setup_path).unwrap();
    let number_of_parties = setup_config.number_of_parties;
    let database_url = create_postgres_url(&config);
    set_database_url(database_url.clone());
    set_index(index);
    set_topic(setup_config.topic);
    let _ = node_server::start_node_server(config.server_port, number_of_parties).await;

    Ok(())
}
