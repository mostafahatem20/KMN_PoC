use std::{ env, sync::Arc };
use futures::future::join_all;
use tokio::{ self, sync::{ mpsc::unbounded_channel, Semaphore } };
use anyhow::Error;
use kmn_poc::{
    database::{
        database::{ create_tables_if_not_exists, establish_connection },
        models::{ Key, PreSignature },
    },
    get_aux_info,
    get_database_url,
    network::node::Node,
    protocol::{ auxinfo, keygen, keyshare, sign::{ self } },
    set_aux_info,
    set_database_url,
    set_index,
    utils::{
        config::{ self, create_postgres_url },
        utils::{ generate_array_i16, generate_array_u16, generate_uuid_v5_from_execution_id },
    },
};
use num_cpus;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();

    // Validate the number of arguments based on the operation
    if args.len() < 4 || (args[2] == "KEY_GEN" && args.len() != 5) {
        eprintln!(
            "Usage: {} <path_to_global_config> <path_to_local_config> <operation> [<number_of_keys>]",
            args[0]
        );
        std::process::exit(1);
    }
    let setup_path = args[1].clone();
    let path = args[2].clone();
    let operation = args[3].clone();

    // Validate the operation
    if operation != "KEY_GEN" && operation != "PRE_SIGN" {
        eprintln!("Invalid operation: {}. Please use 'KEY_GEN' or 'PRE_SIGN'.", operation);
        std::process::exit(1);
    }

    let config = config::read_config(&path).unwrap();
    let setup_config = config::read_setup_config(&setup_path).unwrap();
    let threshold = setup_config.threshold;
    let number_of_parties = setup_config.number_of_parties;
    let index = config.index;
    let database_url = create_postgres_url(&config);
    let topic = setup_config.topic;

    set_database_url(database_url.clone());
    set_index(index);

    // Ensure the keys table exists
    let connection = &mut establish_connection(database_url.clone());
    if let Err(err) = create_tables_if_not_exists(connection) {
        eprintln!("Failed to create table: {}", err);
        std::process::exit(1);
    }

    // Handle operations
    if operation == "KEY_GEN" {
        let number_of_keys: usize = args[4].clone().parse().unwrap(); // Only parse number_of_keys for KEY_GEN
        let _ = generate_keys(number_of_keys, index, threshold, number_of_parties, topic).await?;
    } else if operation == "PRE_SIGN" {
        let _ = generate_pre_signatures(index, threshold, topic).await?;
    }

    Ok(())
}

pub async fn generate_keys(
    number_of_keys: usize,
    index: usize,
    threshold: usize,
    number_of_parties: usize,
    topic: String
) -> Result<(), Error> {
    // Initialize the node
    let (connection_tx, mut connection_rx) = unbounded_channel();
    let mut node = Node::new(index, Some(connection_tx), number_of_parties, topic)?;

    let (receiver1, sender1) = node.add_receiver_sender(0);

    let receiver_senders_keygen: Vec<_> = (0..number_of_keys)
        .map(|i| node.add_receiver_sender(i + 1))
        .collect();

    let node_task = tokio::spawn(async move { node.run().await });
    connection_rx.recv().await;

    // Spawn auxinfo task
    let auxinfo_task = tokio::spawn(async move {
        let eid: [u8; 32] = [0u8; 32];
        let data = auxinfo
            ::run(sender1, receiver1, index as u16, number_of_parties as u16, &eid, 0).await
            .unwrap();
        set_aux_info(&data);
    });

    // Wait for auxinfo and node tasks to complete
    let _ = auxinfo_task.await?;

    let mut keygen_tasks = Vec::new();
    let semaphore = Arc::new(Semaphore::new(num_cpus::get()));
    let aux_info = auxinfo::load_from_string(get_aux_info().to_string()).unwrap();
    for (iteration, (receiver2, sender2)) in receiver_senders_keygen.into_iter().enumerate() {
        let permit = Arc::clone(&semaphore).acquire_owned().await.unwrap();
        let eid = generate_eid(iteration); // Generate EID for the task
        let database_url = get_database_url().to_string();
        let auxinfo_clone = aux_info.clone();
        let keygen_task = tokio::spawn(async move {
            let keygen_data = keygen
                ::run(
                    sender2,
                    receiver2,
                    index as u16,
                    threshold as u16,
                    number_of_parties as u16,
                    &eid,
                    iteration + 1
                ).await
                .unwrap();
            let key_share = keyshare::run(keygen_data, auxinfo_clone).unwrap();
            let connection = &mut establish_connection(database_url);
            let _ = Key::create_key(
                connection,
                generate_uuid_v5_from_execution_id(&eid),
                &serde_json::to_string(&key_share).unwrap()
            );

            drop(permit); // Release the permit when the task is done
        });

        keygen_tasks.push(keygen_task);
    }

    let _ = join_all(keygen_tasks).await;

    let _ = node_task.await?;

    Ok(())
}

pub async fn generate_pre_signatures(
    index: usize,
    threshold: usize,
    topic: String
) -> Result<(), Error> {
    let database_url = get_database_url().to_string();
    let conn = &mut establish_connection(database_url);
    let keys = Key::get_keys_with_no_presignatures(conn);
    let number_of_keys = keys.len();
    let (connection_tx, mut connection_rx) = unbounded_channel();
    let mut node = Node::new(index, Some(connection_tx), threshold, topic)?;

    // Set up receiver and sender pairs for presignature tasks
    let receiver_senders_presign: Vec<_> = (0..number_of_keys)
        .map(|i| node.add_receiver_sender(i + number_of_keys + 1))
        .collect();

    // Spawn the node task
    let node_task = tokio::spawn(async move { node.run().await });

    // Wait for the node to initialize
    connection_rx.recv().await;

    // Set up the semaphore to limit concurrent tasks
    let semaphore = Arc::new(Semaphore::new(num_cpus::get()));
    let mut presign_tasks = Vec::new();
    for (iteration, (receiver, sender)) in receiver_senders_presign.into_iter().enumerate() {
        let permit = Arc::clone(&semaphore).acquire_owned();
        let key = keys.get(iteration).unwrap().clone(); // Clone key for task
        let eid = generate_eid(iteration); // Generate EID for the task
        let key_share = serde_json::from_str(&key.key_share).unwrap(); // Deserialize key_share

        let presign_task = tokio::spawn(async move {
            let permit = permit.await.unwrap(); // Await the permit acquisition
            let signers: &[u16] = &generate_array_u16(threshold);
            let database_url = get_database_url().to_string();
            let connection = &mut establish_connection(database_url);

            if signers.contains(&(index as u16)) {
                let presign_data = sign
                    ::pre_sign(
                        sender,
                        receiver,
                        index as u16,
                        key_share,
                        signers,
                        &eid,
                        iteration + number_of_keys + 1
                    ).await
                    .unwrap();

                let _ = PreSignature::create_presignature(
                    connection,
                    key.id,
                    &serde_json::to_string(&presign_data).unwrap(),
                    &generate_array_i16(threshold)
                );
            }
            drop(permit); // Release the permit when the task is done
        });
        presign_tasks.push(presign_task);
    }

    // Wait for all presignature tasks to complete
    let _ = join_all(presign_tasks).await;

    // Wait for the node task to complete
    let _ = node_task.await?;

    Ok(())
}

// Helper function to generate EID from iteration
fn generate_eid(iteration: usize) -> [u8; 32] {
    let mut eid = [0u8; 32];
    for (i, byte) in iteration.to_be_bytes().iter().enumerate() {
        eid[i] = *byte;
    }
    eid
}
