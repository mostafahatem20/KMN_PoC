use cggmp21::generic_ec::{ NonZero, SecretScalar, Point };
use cggmp21::key_share::Validate;
use cggmp21::KeyShare;
use cggmp21::{ supported_curves::Secp256r1, Presignature, IncompleteKeyShare };
use tonic::{ Request, Response, Status };
use kmn_poc::key_management_server::{ KeyManagement, KeyManagementServer };
use kmn_poc::{
    GetKeysRequest,
    GetKeysResponse,
    GetKeyRequest,
    GetKeyResponse,
    KeyUpdateRequest,
    KeyUpdateResponse,
    SignRequest,
    SignResponse,
    SignOnlineRequest,
    SignOnlineResponse,
    GenerateKeyRequest,
    GenerateKeyResponse,
    KeyInfo,
};

use crate::database::database::establish_connection;
use crate::database::models::{ Key, PreSignature };
use crate::network::node::Node;
use crate::utils::utils::{ generate_array_u16, generate_uuid_v5_from_execution_id };
use crate::{ get_aux_info, get_database_url, get_index, get_topic, set_aux_info };
use crate::protocol::{ auxinfo, keyshare, sign, keygen };
use crate::protocol::sign::partial_sign;
use uuid::Uuid;
use tokio::{ self, sync::mpsc::unbounded_channel };
pub mod kmn_poc {
    tonic::include_proto!("kmn_poc"); // The string specified here must match the proto package name

    // Include the generated file descriptor set
    pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!(
        "kmn_poc_descriptor"
    );
}

#[derive(Debug, Default)]
pub struct NodeServer {}

// Implement the gRPC service by implementing the KeyManagement trait
#[tonic::async_trait]
impl KeyManagement for NodeServer {
    async fn sign(&self, request: Request<SignRequest>) -> Result<Response<SignResponse>, Status> {
        let req = request.into_inner();
        println!("Received Partial Sign request: {:?}", req);

        let msg = req.msg;
        let key_id = req.key_id;

        // Establish database connection
        let connection = &mut establish_connection(get_database_url().to_string());

        // Parse the UUID
        let uuid = match Uuid::parse_str(&key_id) {
            Ok(uuid) => uuid,
            Err(e) => {
                return Err(Status::invalid_argument(format!("Invalid key_id: {}", e)));
            }
        };

        // Fetch presignatures from the database
        let presignatures = match PreSignature::find_by_key_id(connection, uuid) {
            Ok(presignatures) => presignatures,
            Err(e) => {
                return Err(Status::internal(format!("Failed to find presignatures: {}", e)));
            }
        };

        // Ensure we have at least one presignature
        if presignatures.is_empty() {
            return Err(Status::not_found("No presignature found for the provided key_id"));
        }

        // Extract the presignature from the first entry in the list
        let pre_signature = &presignatures[0].pre_signature; // assuming `pre_signature` is a field in your struct

        // Deserialize the presignature
        let extracted_pre_signature: Presignature<Secp256r1> = match
            serde_json::from_str(pre_signature)
        {
            Ok(pre_sig) => pre_sig,
            Err(e) => {
                return Err(Status::internal(format!("Failed to decode presignature: {}", e)));
            }
        };

        // Perform partial signing
        let partial_signature = match partial_sign(extracted_pre_signature, &msg) {
            Ok(signature) => signature,
            Err(e) => {
                return Err(Status::internal(format!("Failed to perform partial signing: {}", e)));
            }
        };
        let response = SignResponse {
            partial_signature: serde_json::to_string(&partial_signature).unwrap(),
        };

        Ok(Response::new(response))
    }

    async fn sign_online(
        &self,
        request: Request<SignOnlineRequest>
    ) -> Result<Response<SignOnlineResponse>, Status> {
        let req = request.into_inner();
        println!("Received Online Sign request: {:?}", req);
        let (connection_tx, mut connection_rx) = unbounded_channel();
        let index = get_index();
        let room_id = req.room_id;
        let msg = &req.msg;
        let key_id = req.key_id;
        let eid = &req.eid;

        // Establish database connection
        let connection = &mut establish_connection(get_database_url().to_string());

        // Parse the UUID
        let uuid = match Uuid::parse_str(&key_id) {
            Ok(uuid) => uuid,
            Err(e) => {
                return Err(Status::invalid_argument(format!("Invalid key_id: {}", e)));
            }
        };
        let key_share: KeyShare<Secp256r1> = serde_json
            ::from_str(&Key::get_by_id(connection, uuid).unwrap())
            .unwrap();
        let threshold = key_share.core.key_info.vss_setup.as_ref().unwrap().min_signers as usize;
        let mut node = Node::new(index, Some(connection_tx), threshold, key_id).unwrap();
        let (receiver1, sender1) = node.add_receiver_sender(room_id as usize);
        let node_task = tokio::spawn(async move { node.run().await });
        connection_rx.recv().await;

        let (signature, _) = sign
            ::run(
                sender1,
                receiver1,
                index as u16,
                key_share,
                &generate_array_u16(threshold),
                msg,
                eid,
                room_id as usize
            ).await
            .unwrap();

        let _ = node_task.await.unwrap();

        let response = SignOnlineResponse {
            signature: serde_json::to_string(&signature).unwrap(),
        };

        Ok(Response::new(response))
    }

    async fn generate_key(
        &self,
        request: Request<GenerateKeyRequest>
    ) -> Result<Response<GenerateKeyResponse>, Status> {
        let req = request.into_inner();
        println!("Received Generate Key request: {:?}", req);
        let (connection_tx, mut connection_rx) = unbounded_channel();
        let index = get_index();
        let room_id = req.room_id;
        let number_of_parties = req.number_of_parties;
        let eid = &req.eid;
        let threshold = req.threshold;

        // Establish database connection
        let connection = &mut establish_connection(get_database_url().to_string());
        let mut node = Node::new(
            index,
            Some(connection_tx),
            number_of_parties as usize,
            room_id.to_string()
        ).map_err(|e| { Status::internal(format!("Failed to create node: {:?}", e)) })?;
        let (receiver, sender) = node.add_receiver_sender(room_id as usize);
        let node_task = tokio::spawn(async move { node.run().await });
        connection_rx.recv().await;

        let (incomplete_key_share, _) = keygen
            ::run(
                sender,
                receiver,
                index as u16,
                threshold as u16,
                number_of_parties as u16,
                &eid,
                room_id as usize
            ).await
            .unwrap();

        let _ = node_task.await.unwrap();
        let key_id = generate_uuid_v5_from_execution_id(eid);
        let aux_info = auxinfo::load_from_string(get_aux_info().to_string()).unwrap();
        let key_share = keyshare::run(incomplete_key_share, aux_info).unwrap();
        let _ = Key::create_key(connection, key_id, &serde_json::to_string(&key_share).unwrap());

        let response = GenerateKeyResponse {
            key_id: key_id.to_string(),
            pub_key: serde_json::to_string(&key_share.core.shared_public_key).unwrap(),
        };

        Ok(Response::new(response))
    }

    async fn get_keys(
        &self,
        request: Request<GetKeysRequest>
    ) -> Result<Response<GetKeysResponse>, Status> {
        let req = request.into_inner();
        println!("Received GetKeys request: {:?}", req);

        // Establish database connection
        let connection = &mut establish_connection(get_database_url().to_string());

        // Fetch keys from the database
        let keys = Key::get_all(connection);

        // Parse key_share for each key and extract the public key
        let key_infos: Vec<KeyInfo> = keys
            .into_iter()
            .map(|key| {
                // Deserialize key_share to extract pub_key
                let key_share: KeyShare<Secp256r1> = serde_json::from_str(&key.key_share).unwrap();

                KeyInfo {
                    key_id: key.id.to_string(), // Assuming `id` is the key ID
                    pub_key: serde_json::to_string(&key_share.core.shared_public_key).unwrap(),
                }
            })
            .collect();

        let response = GetKeysResponse {
            keys: key_infos,
        };

        Ok(Response::new(response))
    }

    async fn get_key(
        &self,
        request: Request<GetKeyRequest>
    ) -> Result<Response<GetKeyResponse>, Status> {
        let req = request.into_inner();
        println!("Received GetKey request: {:?}", req);
        // Establish database connection
        let connection = &mut establish_connection(get_database_url().to_string());

        let key_id = req.key_id;
        // Parse the UUID
        let uuid = match Uuid::parse_str(&key_id) {
            Ok(uuid) => uuid,
            Err(e) => {
                return Err(Status::invalid_argument(format!("Invalid key_id: {}", e)));
            }
        };

        let response = GetKeyResponse {
            key_share: Key::get_by_id(connection, uuid).unwrap(),
        };

        Ok(Response::new(response))
    }

    async fn key_update(
        &self,
        request: Request<KeyUpdateRequest>
    ) -> Result<Response<KeyUpdateResponse>, Status> {
        let req = request.into_inner();
        println!("Received KeyUpdate request: {:?}", req);
        let connection = &mut establish_connection(get_database_url().to_string());

        let key_id = req.key_id;
        let new_key_id = req.new_key_id;

        // Parse the UUID
        let uuid = match Uuid::parse_str(&key_id) {
            Ok(uuid) => uuid,
            Err(e) => {
                return Err(Status::invalid_argument(format!("Invalid key_id: {}", e)));
            }
        };

        let new_uuid = match Uuid::parse_str(&new_key_id) {
            Ok(uuid) => uuid,
            Err(e) => {
                return Err(Status::invalid_argument(format!("Invalid key_id: {}", e)));
            }
        };
        // Fetch presignatures from the database
        let key_share_local: KeyShare<Secp256r1> = serde_json
            ::from_str(&Key::get_by_id(connection, uuid).unwrap())
            .unwrap();

        let key_share_to_be_added: IncompleteKeyShare<Secp256r1> = serde_json
            ::from_str(&req.key_share)
            .unwrap();

        let key1 = (&key_share_local.core.x).clone().into_inner();
        let key2 = (&key_share_to_be_added.x).clone().into_inner();
        let scalar1 = key1.as_ref();
        let scalar2 = key2.as_ref();
        let mut added_scalar = scalar1 + scalar2;
        let new_secret_scalar = SecretScalar::new(&mut added_scalar);
        let new_secret = NonZero::<SecretScalar<Secp256r1>>
            ::from_secret_scalar(new_secret_scalar)
            .unwrap();

        let shared_pk_1 = key_share_local.core.key_info.shared_public_key.clone().into_inner();
        let shared_pk_2 = key_share_to_be_added.key_info.shared_public_key.clone().into_inner();
        let shared_pk = NonZero::<Point<Secp256r1>>::from_point(shared_pk_1 + shared_pk_2);

        let public_shares_1 = key_share_local.core.key_info.public_shares.clone();
        let public_shares_2 = key_share_to_be_added.key_info.public_shares.clone();

        // Ensure both vectors have the same length to avoid panics
        if public_shares_1.len() != public_shares_2.len() {
            return Err(Status::invalid_argument("Public share vectors have different lengths."));
        }

        // Sum corresponding public shares
        let new_public_shares: Vec<_> = public_shares_1
            .iter()
            .zip(public_shares_2.iter())
            .map(|(share1, share2)| {
                let pk1 = share1.clone().into_inner();
                let pk2 = share2.clone().into_inner();
                NonZero::<Point<Secp256r1>>
                    ::from_point(pk1 + pk2)
                    .expect("Point addition resulted in an invalid point") // Perform point addition on the two public keys
            })
            .collect();

        let mut key_info = key_share_local.core.key_info.clone();
        key_info.public_shares = new_public_shares;
        if let Some(pk) = shared_pk {
            key_info.shared_public_key = pk;
        }

        let index = get_index() as u16;
        let key_share = IncompleteKeyShare::<Secp256r1>
            ::from_parts((index, key_info, new_secret))
            .unwrap();
        let aux_info = key_share_local.aux.clone().validate().unwrap();

        let final_key_share: KeyShare<Secp256r1> = keyshare::run(key_share, aux_info).unwrap();

        // let _ = PreSignature::delete_by_key_id(connection, uuid);

        let new_key = Key::create_key(
            connection,
            new_uuid,
            &serde_json::to_string(&final_key_share).unwrap()
        );

        Ok(
            Response::new(KeyUpdateResponse {
                key_id: new_key.id.to_string(),
                pub_key: serde_json::to_string(&final_key_share.core.shared_public_key).unwrap(),
            })
        )
    }
}

// Function to create the gRPC server with the NodeServer implementation
pub async fn start_node_server(
    port: usize,
    number_of_parties: usize
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("[::1]:{}", port).parse().unwrap();
    let node_server = NodeServer::default();
    let index = get_index();
    // Initialize the node
    let topic = get_topic().to_string();
    let (connection_tx, mut connection_rx) = unbounded_channel();
    let mut node = Node::new(index, Some(connection_tx), number_of_parties, topic)?;

    let (receiver1, sender1) = node.add_receiver_sender(0);

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

    let _ = node_task.await?;

    println!("NodeServer listening on {}", addr);

    let reflection_service = tonic_reflection::server::Builder
        ::configure()
        .register_encoded_file_descriptor_set(kmn_poc::FILE_DESCRIPTOR_SET)
        .build()?;

    tonic::transport::Server
        ::builder()
        .accept_http1(true)
        .layer(tower_http::cors::CorsLayer::permissive())
        .add_service(reflection_service) // Add the reflection service last
        .add_service(KeyManagementServer::new(node_server)) // Add your gRPC service first
        .serve(addr).await?;

    Ok(())
}
