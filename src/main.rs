use std::{ env, sync::Arc };
use cggmp21::{ supported_curves::Secp256r1, KeyShare };
use futures::future::join_all;
use tokio::{ self, sync::{ mpsc::unbounded_channel, Semaphore } };
use anyhow::{ Error, anyhow };
use kmn_poc::{
    database::{
        database::{ create_tables_if_not_exists, establish_connection },
        models::{ Key, PreSignature },
    },
    get_aux_info,
    get_database_url,
    network::node::{ Node, NodeReceiver, NodeSender },
    protocol::{ auxinfo, keygen, keyshare, sign::{ self } },
    set_aux_info,
    set_database_url,
    set_index,
    utils::{
        config::{ self, create_postgres_url },
        utils::{
            generate_array_i16,
            generate_array_u16,
            generate_uuid_v5_from_execution_id,
            MyPerfReport,
        },
    },
};
use std::fs::OpenOptions;
use num_cpus;
use std::io::Write;
use tokio::fs;

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Initialize logging to log errors to a file
    let args: Vec<String> = env::args().collect();
    let log_file_name = format!("application_errors_{}.log", args[1..].join("_").replace("/", "_"));
    println!("{:?}", log_file_name);
    let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_name)?;

    if let Err(err) = actual_main(args).await {
        writeln!(log_file, "Error: {:?}", err).unwrap();
        println!("An error occurred: {:?}", err);
        return Ok(());
    }

    Ok(())
}

async fn actual_main(args: Vec<String>) -> Result<(), Error> {
    // Your existing main logic goes here
    // Wrap the main code and return errors, so they can be logged
    if
        args.len() < 4 ||
        ((args[3] == "KEY_GEN" || args[3] == "PRE_SIGN_TEST" || args[3] == "SIGN_TEST") &&
            args.len() != 5)
    {
        return Err(anyhow!("Invalid number of arguments"));
    }

    let setup_path = args[1].clone();
    let path = args[2].clone();
    let operation = args[3].clone();

    // Validate the operation
    if
        operation != "KEY_GEN" &&
        operation != "PRE_SIGN" &&
        operation != "PRE_SIGN_TEST" &&
        operation != "SIGN_TEST"
    {
        return Err(anyhow!("Invalid operation: {}", operation));
    }

    let config = config::read_config(&path)?;
    let setup_config = config::read_setup_config(&setup_path)?;
    let threshold = setup_config.threshold;
    let number_of_parties = setup_config.number_of_parties;
    let index = config.index;
    let database_url = create_postgres_url(&config);
    let topic = setup_config.topic;

    set_database_url(database_url.clone());
    set_index(index);

    let connection = &mut establish_connection(database_url.clone());
    if let Err(err) = create_tables_if_not_exists(connection) {
        return Err(anyhow!("Failed to create table: {}", err));
    }

    if operation == "KEY_GEN" {
        let number_of_keys: usize = args[4].parse()?;
        let _ = generate_keys(number_of_keys, index, threshold, number_of_parties, topic).await?;
    } else if operation == "PRE_SIGN" {
        let _ = generate_pre_signatures(index, threshold, topic).await?;
    } else if operation == "PRE_SIGN_TEST" {
        let number_of_iterations: usize = args[4].parse()?;
        let _ = generate_pre_signatures_with_one_key(
            number_of_iterations,
            index,
            threshold,
            number_of_parties,
            topic
        ).await?;
    } else if operation == "SIGN_TEST" {
        let number_of_iterations: usize = args[4].parse()?;
        let _ = generate_signatures_with_one_key(
            number_of_iterations,
            index,
            threshold,
            number_of_parties,
            topic
        ).await?;
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
            let (keygen_data, report) = keygen
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
            report
        });

        keygen_tasks.push(keygen_task);
    }

    // Wait for all tasks to complete and collect the reports
    let reports: Vec<MyPerfReport> = join_all(keygen_tasks).await
        .into_iter()
        .map(|res| res.unwrap()) // Unwrap the result to get the report
        .collect();

    let _ = node_task.await?;

    let json_reports = serde_json::to_string_pretty(&reports).unwrap();
    let filename = format!("keygen_reports_{}.json", index); // Include the task index in the filename

    fs::write(filename, json_reports).await?;

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

            // Initialize report as empty
            let report = if signers.contains(&(index as u16)) {
                let (presign_data, report) = sign
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

                // Return the report
                Some(report)
            } else {
                None
            };

            drop(permit); // Release the permit when the task is done
            report // Return the report (or None if there is no report)
        });

        presign_tasks.push(presign_task);
    }

    // Wait for all presignature tasks to complete and collect the reports
    let reports: Vec<_> = join_all(presign_tasks).await
        .into_iter()
        .filter_map(|res| res.unwrap()) // Unwrap the result and filter out None values
        .collect();

    // Serialize reports to JSON and write to a file
    let json_reports = serde_json::to_string_pretty(&reports).unwrap();
    let filename = format!("presignature_reports_{}.json", index); // Include the task index in the filename

    tokio::fs::write(filename, json_reports).await?;

    // Wait for the node task to complete
    let _ = node_task.await?;

    Ok(())
}

pub async fn generate_key(
    index: usize,
    threshold: usize,
    number_of_parties: usize
) -> Result<KeyShare<Secp256r1>, Error> {
    // Initialize the node
    let (connection_tx, mut connection_rx) = unbounded_channel();
    let mut node = Node::new(
        index,
        Some(connection_tx),
        number_of_parties,
        "dummy_key".to_string()
    )?;

    let (receiver1, sender1) = node.add_receiver_sender(0);
    let (receiver2, sender2) = node.add_receiver_sender(1);

    let node_task = tokio::spawn(async move { node.run().await });
    connection_rx.recv().await;

    // Spawn auxinfo task
    let auxinfo_task = tokio::spawn(async move {
        let eid: [u8; 32] = [0u8; 32];
        let data = auxinfo
            ::run(sender1, receiver1, index as u16, number_of_parties as u16, &eid, 0).await
            .unwrap();
        data
    });

    // Wait for auxinfo and node tasks to complete
    let aux_data = auxinfo_task.await?;

    // Spawn key_gen task
    let keygen_task = tokio::spawn(async move {
        let eid: [u8; 32] = [0u8; 32];
        let data = keygen
            ::run(
                sender2,
                receiver2,
                index as u16,
                threshold as u16,
                number_of_parties as u16,
                &eid,
                1
            ).await
            .unwrap();
        data
    });
    let (dirty_key, _) = keygen_task.await?;
    let key_share = keyshare::run(dirty_key, aux_data)?;

    let _ = node_task.await?;
    Ok(key_share)
}

pub async fn generate_pre_signatures_with_one_key(
    number_of_iterations: usize,
    index: usize,
    threshold: usize,
    number_of_parties: usize,
    topic: String
) -> Result<(), Error> {
    let key_share = generate_key(index, threshold, number_of_parties).await?;
    let signers = generate_array_u16(threshold);

    if signers.contains(&(index as u16)) {
        // Initialize the node
        let (connection_tx, mut connection_rx) = unbounded_channel();
        let mut node = Node::new(index, Some(connection_tx), threshold, topic.clone())?;

        // Include the original iteration number with each receiver and sender pair
        let mut receiver_senders_presign: Vec<_> = (0..number_of_iterations)
            .map(|i| (i, node.add_receiver_sender(i))) // Store the iteration index along with (receiver, sender)
            .collect();

        // Spawn the node task
        let node_task = tokio::spawn(async move { node.run().await });

        // Wait for the node to initialize
        connection_rx.recv().await;

        let mut reports = Vec::new(); // Collect reports from all batches

        let mut chunks: Vec<Vec<(usize, (NodeReceiver, NodeSender))>> = Vec::new();

        // Move chunks out of the original vector using drain and preserve iteration numbers
        while !receiver_senders_presign.is_empty() {
            let chunk: Vec<(usize, (NodeReceiver, NodeSender))> = receiver_senders_presign
                .drain(..num_cpus::get().min(receiver_senders_presign.len()))
                .collect();
            chunks.push(chunk);
        }

        // Process tasks in batches based on the number of available CPUs
        for chunk in chunks {
            let mut batch_tasks = Vec::new();

            // Use into_iter() to move (original_iteration, (receiver, sender)) out of the chunk
            for (original_iteration, (receiver, sender)) in chunk.into_iter() {
                let key_share_clone = key_share.clone();
                let signers_clone = signers.clone();

                // Move receiver and sender into the async task and pass the original_iteration
                let presign_task = tokio::spawn(async move {
                    let eid = generate_eid(original_iteration); // Use the original_iteration to generate EID
                    let (_, report) = sign
                        ::pre_sign(
                            sender, // Moved sender
                            receiver, // Moved receiver
                            index as u16,
                            key_share_clone,
                            &signers_clone,
                            &eid,
                            original_iteration // Pass original iteration to pre_sign
                        ).await
                        .unwrap();

                    report // Return the report for this task
                });

                batch_tasks.push(presign_task);
            }

            // Wait for all tasks in the current batch to complete
            let batch_reports: Vec<_> = join_all(batch_tasks).await
                .into_iter()
                .map(|res| res.unwrap()) // Unwrap the result to get the report
                .collect();

            // Collect reports from the current batch
            reports.extend(batch_reports);
        }

        // Serialize reports to JSON and write to a file
        let json_reports = serde_json::to_string_pretty(&reports).unwrap();
        let filename = format!("presignature_reports_test_{}.json", index); // Include the task index in the filename
        tokio::fs::write(filename, json_reports).await?;

        let _ = node_task.await?;
    }

    Ok(())
}

pub async fn generate_signatures_with_one_key(
    number_of_iterations: usize,
    index: usize,
    threshold: usize,
    number_of_parties: usize,
    topic: String
) -> Result<(), Error> {
    let message: &[u8] = b"Hello, world! This is a random message in byte array format.";
    // Initialize the node
    let key_share = generate_key(index, threshold, number_of_parties).await?;
    let signers = generate_array_u16(threshold);

    if signers.contains(&(index as u16)) {
        // Initialize the node
        let (connection_tx, mut connection_rx) = unbounded_channel();
        let mut node = Node::new(index, Some(connection_tx), threshold, topic.clone())?;

        // Include the original iteration number with each receiver and sender pair
        let mut receiver_senders_sign: Vec<_> = (0..number_of_iterations)
            .map(|i| (i, node.add_receiver_sender(i))) // Store the iteration index along with (receiver, sender)
            .collect();

        // Spawn the node task
        let node_task = tokio::spawn(async move { node.run().await });

        // Wait for the node to initialize
        connection_rx.recv().await;

        let mut reports = Vec::new(); // Collect reports from all batches

        let mut chunks: Vec<Vec<(usize, (NodeReceiver, NodeSender))>> = Vec::new();

        // Move chunks out of the original vector using drain and preserve iteration numbers
        while !receiver_senders_sign.is_empty() {
            let chunk: Vec<(usize, (NodeReceiver, NodeSender))> = receiver_senders_sign
                .drain(..num_cpus::get().min(receiver_senders_sign.len()))
                .collect();
            chunks.push(chunk);
        }

        // Process tasks in batches based on the number of available CPUs
        for chunk in chunks {
            let mut batch_tasks = Vec::new();

            // Use into_iter() to move (original_iteration, (receiver, sender)) out of the chunk
            for (original_iteration, (receiver, sender)) in chunk.into_iter() {
                let key_share_clone = key_share.clone();
                let signers_clone = signers.clone();

                // Move receiver and sender into the async task and pass the original_iteration
                let sign_task = tokio::spawn(async move {
                    let eid = generate_eid(original_iteration); // Use the original_iteration to generate EID
                    let (_, report) = sign
                        ::run(
                            sender, // Moved sender
                            receiver, // Moved receiver
                            index as u16,
                            key_share_clone,
                            &signers_clone,
                            message,
                            &eid,
                            original_iteration // Pass original iteration to pre_sign
                        ).await
                        .unwrap();

                    report // Return the report for this task
                });

                batch_tasks.push(sign_task);
            }

            // Wait for all tasks in the current batch to complete
            let batch_reports: Vec<_> = join_all(batch_tasks).await
                .into_iter()
                .map(|res| res.unwrap()) // Unwrap the result to get the report
                .collect();

            // Collect reports from the current batch
            reports.extend(batch_reports);
        }

        // Serialize reports to JSON and write to a file
        let json_reports = serde_json::to_string_pretty(&reports).unwrap();
        let filename = format!("signature_reports_{}.json", index); // Include the task index in the filename
        tokio::fs::write(filename, json_reports).await?;

        let _ = node_task.await?;
    }

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
