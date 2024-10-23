use std::collections::HashMap;
use std::sync::Arc;
use cggmp21::generic_ec::{ NonZero, Point };
use cggmp21::Signature;
use tonic::transport::{ Channel, Server };
use tonic::{ Request, Response, Status };
use kmn_poc_client::key_management_service_server::{
    KeyManagementService,
    KeyManagementServiceServer,
};
use kmn_poc_client::{
    AssignKeyRequest,
    AssignKeyResponse,
    ExportKeyRequest,
    ExportKeyResponse,
    GenerateKeyResponse,
    KeyUpdateRequest,
    KeyUpdateResponse,
    SignOnlineRequest,
    SignOnlineResponse,
    SignRequest,
    SignResponse,
    GenerateKeyRequest,
    CombinePublicKeysRequest,
    CombinePublicKeysResponse,
};
use kmn_poc::key_management_client::KeyManagementClient;
use crate::utils::utils::generate_array_u16;
use crate::{ get_number_of_parties, get_threshold };
use crate::network::node_server::kmn_poc::{ self };
use std::error::Error;
use cggmp21::{ supported_curves::Secp256r1, PartialSignature, generic_ec::SecretScalar, KeyShare };
use crate::protocol::{ key_export, key_import, sign };
use rand::Rng; // Add this import for generating random values
use uuid::Uuid;
use std::sync::atomic::{ AtomicI64, Ordering };
use tokio::sync::{ Semaphore, Mutex }; // Import Semaphore from Tokio

pub mod kmn_poc_client {
    tonic::include_proto!("kmn_poc_client"); // The string specified here must match the proto package name

    // Include the generated file descriptor set
    pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!(
        "kmn_poc_descriptor"
    );
}

#[derive(Debug)]
pub struct ClientServer {
    pub keys: Arc<Mutex<HashMap<(String, String), Option<String>>>>,
    pub clients: Vec<Arc<Mutex<KeyManagementClient<Channel>>>>,
    pub index_counter: Arc<AtomicI64>, // Atomic counter for unique indexes
    pub semaphore: Arc<Semaphore>, // Semaphore to allow a maximum of 10 concurrent requests
}

impl ClientServer {
    pub async fn new(
        keys: HashMap<(String, String), Option<String>>, // Updated to hold tuple as the key
        urls: Vec<String>
    ) -> Result<Self, tonic::transport::Error> {
        let mut clients = Vec::new();

        for url in urls {
            let client = KeyManagementClient::connect(url).await?;
            clients.push(Arc::new(Mutex::new(client)));
        }

        Ok(ClientServer {
            keys: Arc::new(Mutex::new(keys)),
            clients,
            index_counter: Arc::new(AtomicI64::new(0)),
            semaphore: Arc::new(Semaphore::new(10)), // Allow only 10 concurrent requests
        })
    }
}

// Implement the gRPC service by implementing the KeyManagementService trait
#[tonic::async_trait]
impl KeyManagementService for ClientServer {
    async fn assign_key(
        &self,
        request: Request<AssignKeyRequest>
    ) -> Result<Response<AssignKeyResponse>, Status> {
        let req = request.into_inner();
        println!("Received AssignKey request");
        let id = req.id;
        // Lock the keys map for safe access
        let mut keys = self.keys.lock().await;

        // Fetch a unique index using atomic counter
        let index = self.index_counter.fetch_add(1, Ordering::SeqCst);

        // Try to assign a key from the first client
        let client = self.clients.get(0).unwrap().clone(); // Get the first client
        let assign_key_request = kmn_poc::AssignKeyRequest { index };

        match assign_key_request_to_client(client, assign_key_request).await {
            Ok(assign_key_response) => {
                // Assign the key and add it to the HashMap
                keys.insert(
                    (assign_key_response.key_id.clone(), assign_key_response.pub_key.clone()),
                    Some(id.clone())
                );
                println!("Len of keys {:?}", keys.len());

                let response = AssignKeyResponse {
                    key_id: assign_key_response.key_id.clone(),
                    pub_key: assign_key_response.pub_key.clone(),
                };

                Ok(Response::new(response))
            }
            Err(_) => {
                println!("No key found, attempting to generate a new key...");
                // Acquire a permit from the semaphore before proceeding
                let permit = self.semaphore.acquire().await.unwrap();
                // If no key is found, attempt to generate a new key
                let room_id = rand::thread_rng().gen();
                let eid: Vec<u8> = (0..32).map(|_| rand::random::<u8>()).collect();

                let generate_req = kmn_poc::GenerateKeyRequest {
                    threshold: get_threshold() as i32,
                    number_of_parties: get_number_of_parties() as i32,
                    room_id,
                    eid,
                };

                // Generate the key using multiple clients
                let futures: Vec<_> = self.clients
                    .iter()
                    .map(|client| generate_key_request(client.clone(), generate_req.clone()))
                    .collect();

                let responses: Vec<kmn_poc::GenerateKeyResponse> = futures::future
                    ::join_all(futures).await
                    .into_iter()
                    .collect::<Result<_, _>>()
                    .map_err(|e| Status::internal(format!("Generate Key request failed: {}", e)))?;

                let new_key_response = responses.get(0).unwrap();

                // Add the newly generated key to the HashMap
                keys.insert(
                    (new_key_response.key_id.clone(), new_key_response.pub_key.clone()),
                    Some(id.clone())
                );
                println!("Len of keys {:?}", keys.len());

                let response = AssignKeyResponse {
                    key_id: new_key_response.key_id.clone(),
                    pub_key: new_key_response.pub_key.clone(),
                };
                drop(permit);

                Ok(Response::new(response))
            }
        }
    }
    async fn sign_online(
        &self,
        request: Request<SignOnlineRequest>
    ) -> Result<Response<SignOnlineResponse>, Status> {
        // Acquire a permit from the semaphore before proceeding
        let permit = self.semaphore.acquire().await.unwrap();
        let req = request.into_inner();
        println!("Received Online Sign request: {:?}", req);

        // Generate a random i32 for room_id
        let room_id = rand::thread_rng().gen();

        // Generate a random &[u8] for eid (using a random byte array of length 32)
        let eid: Vec<u8> = (0..32).map(|_| rand::random::<u8>()).collect();

        let parties_indexes = generate_array_u16(get_threshold());

        let node_req = kmn_poc::SignOnlineRequest {
            msg: req.msg,
            key_id: req.key_id,
            room_id,
            eid,
        };

        // Run all client sign requests in parallel
        // Filter clients if their index is in parties_indexes
        let futures: Vec<_> = self.clients
            .iter()
            .enumerate()
            .filter(|(index, _)| parties_indexes.contains(&(index.clone() as u16)))
            .map(|(_, client)| sign_online_request(client.clone(), node_req.clone()))
            .collect();

        let responses: Vec<kmn_poc::SignOnlineResponse> = futures::future
            ::join_all(futures).await
            .into_iter()
            .collect::<Result<_, _>>()
            .map_err(|e| Status::internal(format!("Sign Online request failed: {}", e)))?;

        let signature: Vec<Signature<Secp256r1>> = responses
            .iter()
            .map(|res| serde_json::from_str(&res.signature).unwrap())
            .collect();

        drop(permit);

        Ok(
            Response::new(SignOnlineResponse {
                signature: serde_json::to_string(&signature.get(0)).unwrap(),
            })
        )
    }

    async fn generate_key(
        &self,
        request: Request<GenerateKeyRequest>
    ) -> Result<Response<GenerateKeyResponse>, Status> {
        // Acquire a permit from the semaphore before proceeding
        let permit = self.semaphore.acquire().await.unwrap();
        let req = request.into_inner();
        println!("Received Generate Key request: {:?}", req);

        // Generate a random i32 for room_id
        let room_id = rand::thread_rng().gen();

        // Generate a random &[u8] for eid (using a random byte array of length 32)
        let eid: Vec<u8> = (0..32).map(|_| rand::random::<u8>()).collect();

        let node_req = kmn_poc::GenerateKeyRequest {
            threshold: get_threshold() as i32,
            number_of_parties: get_number_of_parties() as i32,
            room_id,
            eid,
        };

        let futures: Vec<_> = self.clients
            .iter()
            .enumerate()
            .map(|(_, client)| generate_key_request(client.clone(), node_req.clone()))
            .collect();

        let responses: Vec<kmn_poc::GenerateKeyResponse> = futures::future
            ::join_all(futures).await
            .into_iter()
            .collect::<Result<_, _>>()
            .map_err(|e| Status::internal(format!("Generate Key request failed: {}", e)))?;
        let first = responses.get(0).unwrap();
        drop(permit);

        Ok(
            Response::new(GenerateKeyResponse {
                key_id: first.key_id.clone(),
                pub_key: first.pub_key.clone(),
            })
        )
    }

    async fn sign(&self, request: Request<SignRequest>) -> Result<Response<SignResponse>, Status> {
        let req = request.into_inner();
        println!("Received Sign request: {:?}", req);

        let node_req = kmn_poc::SignRequest {
            msg: req.msg.clone(),
            key_id: req.key_id.clone(),
        };

        let parties_indexes = generate_array_u16(get_threshold());

        // Run all client sign requests in parallel
        // Filter clients if their index is in parties_indexes
        let futures: Vec<_> = self.clients
            .iter()
            .enumerate()
            .filter(|(index, _)| parties_indexes.contains(&(index.clone() as u16)))
            .map(|(_, client)| sign_request(client.clone(), node_req.clone()))
            .collect();

        let responses: Result<Vec<kmn_poc::SignResponse>, _> = futures::future
            ::join_all(futures).await
            .into_iter()
            .collect();

        // If the sign request fails, try running the sign_online
        match responses {
            Ok(responses) => {
                // Partial Sign combine
                let partial_signatures: Vec<PartialSignature<Secp256r1>> = responses
                    .iter()
                    .map(|res| serde_json::from_str(&res.partial_signature).unwrap())
                    .collect();

                let data = sign::combine_sign(&partial_signatures).unwrap();

                let response = SignResponse {
                    signature: serde_json::to_string(&data).unwrap(),
                };

                Ok(Response::new(response))
            }
            Err(_) => {
                // Acquire a permit from the semaphore before proceeding
                let permit = self.semaphore.acquire().await.unwrap();
                println!("Sign request failed, attempting sign_online...");

                // Try the sign_online if the regular sign fails
                let node_req_online = kmn_poc::SignOnlineRequest {
                    msg: req.msg,
                    key_id: req.key_id,
                    room_id: rand::thread_rng().gen(),
                    eid: (0..32).map(|_| rand::random::<u8>()).collect(),
                };
                println!("{:?}", parties_indexes);
                let futures_online: Vec<_> = self.clients
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| parties_indexes.contains(&(index.clone() as u16)))
                    .map(|(_, client)| sign_online_request(client.clone(), node_req_online.clone()))
                    .collect();

                let responses_online: Vec<kmn_poc::SignOnlineResponse> = futures::future
                    ::join_all(futures_online).await
                    .into_iter()
                    .collect::<Result<_, _>>()
                    .map_err(|e| Status::internal(format!("Sign Online request failed: {}", e)))?;

                let signature: Vec<Signature<Secp256r1>> = responses_online
                    .iter()
                    .map(|res| serde_json::from_str(&res.signature).unwrap())
                    .collect();

                drop(permit);

                Ok(
                    Response::new(SignResponse {
                        signature: serde_json::to_string(&signature.get(0)).unwrap(),
                    })
                )
            }
        }
    }

    async fn export_key(
        &self,
        request: Request<ExportKeyRequest>
    ) -> Result<Response<ExportKeyResponse>, Status> {
        let req = request.into_inner();
        println!("Received ExportKey request: {:?}", req);
        let node_req = kmn_poc::GetKeyRequest {
            key_id: req.key_id,
        };

        // Run all client get key requests in parallel
        let futures: Vec<_> = self.clients
            .iter()
            .map(|client| get_key_request(client.clone(), node_req.clone()))
            .collect();

        let responses: Vec<kmn_poc::GetKeyResponse> = futures::future
            ::join_all(futures).await
            .into_iter()
            .collect::<Result<_, _>>()
            .map_err(|e| Status::internal(format!("GetKey request failed: {}", e)))?;

        // Key export
        let key_shares: Vec<KeyShare<Secp256r1>> = responses
            .iter()
            .map(|res| serde_json::from_str(&res.key_share).unwrap())
            .collect();

        let data = key_export::run(&key_shares).await.unwrap();
        let response = ExportKeyResponse {
            key: serde_json::to_string(&data).unwrap(),
        };

        Ok(Response::new(response))
    }

    async fn key_update(
        &self,
        request: Request<KeyUpdateRequest>
    ) -> Result<Response<KeyUpdateResponse>, Status> {
        let req = request.into_inner();
        println!("Received KeyUpdate request: {:?}", req);
        let key: SecretScalar<Secp256r1> = serde_json::from_str(&req.key).unwrap();

        let keys = key_import
            ::run(key, get_threshold() as u16, get_number_of_parties() as u16).await
            .unwrap();

        let mut futures = Vec::new();
        let new_uuid = Uuid::new_v4().to_string();
        for (i, client) in self.clients.iter().enumerate() {
            let key_share = serde_json::to_string(&keys.get(i).unwrap()).unwrap();
            println!("{:?}", key_share);
            let node_req = kmn_poc::KeyUpdateRequest {
                new_key_id: new_uuid.clone(),
                key_id: req.key_id.clone(),
                key_share,
            };
            futures.push(key_update_request(client.clone(), node_req));
        }

        // Await all key update requests
        let results: Vec<kmn_poc::KeyUpdateResponse> = futures::future
            ::join_all(futures).await
            .into_iter()
            .collect::<Result<_, _>>()
            .map_err(|e| Status::internal(format!("KeyUpdate request failed: {}", e)))?;

        let new_key = results.get(0).unwrap();

        Ok(
            Response::new(KeyUpdateResponse {
                key_id: new_key.key_id.clone(),
                pub_key: new_key.pub_key.clone(),
            })
        )
    }

    async fn combine_public_keys(
        &self,
        request: Request<CombinePublicKeysRequest>
    ) -> Result<Response<CombinePublicKeysResponse>, Status> {
        let req = request.into_inner();
        println!("Received CombinePublicKeys request: {:?}", req);
        let pk_1: Point<Secp256r1> = serde_json::from_str(&req.pub_key).unwrap();
        let node_req = kmn_poc::GetKeyRequest {
            key_id: req.key_id.clone(),
        };

        let key_details = get_key_request(
            self.clients.get(0).unwrap().clone(),
            node_req.clone()
        ).await.unwrap();

        let key_share: KeyShare<Secp256r1> = serde_json::from_str(&key_details.key_share).unwrap();
        let pk_2 = key_share.core.key_info.shared_public_key.clone().into_inner();
        let combined_pk = NonZero::<Point<Secp256r1>>::from_point(pk_1 + pk_2);

        Ok(
            Response::new(CombinePublicKeysResponse {
                pub_key: serde_json::to_string(&combined_pk).unwrap(),
            })
        )
    }
}

pub async fn start_client_server(
    port: usize,
    keys_map: HashMap<(String, String), Option<String>>, // Use the tuple (key_id, pub_key) as the key
    urls: Vec<String>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = format!("[::1]:{}", port).parse().unwrap();

    // Initialize the ClientServer with the updated keys_map
    let client_server = ClientServer::new(keys_map, urls).await?;

    println!("Client Server listening on {}", addr);

    let reflection_service = tonic_reflection::server::Builder
        ::configure()
        .register_encoded_file_descriptor_set(kmn_poc_client::FILE_DESCRIPTOR_SET)
        .build()?;

    Server::builder()
        .accept_http1(true)
        .layer(tower_http::cors::CorsLayer::permissive())
        .add_service(reflection_service) // Add the reflection service last
        .add_service(KeyManagementServiceServer::new(client_server)) // Add your gRPC service first
        .serve(addr).await?;

    Ok(())
}
// Function to send a sign request to a client
async fn sign_request(
    client: Arc<Mutex<KeyManagementClient<Channel>>>,
    req: kmn_poc::SignRequest
) -> Result<kmn_poc::SignResponse, Box<dyn Error + Send + Sync>> {
    let mut client = client.lock().await;
    let request = tonic::Request::new(req);
    let response = client.sign(request).await?;
    Ok(response.into_inner())
}

// Function to send a get key request to a client
async fn get_key_request(
    client: Arc<Mutex<KeyManagementClient<Channel>>>,
    req: kmn_poc::GetKeyRequest
) -> Result<kmn_poc::GetKeyResponse, Box<dyn Error + Send + Sync>> {
    let mut client = client.lock().await;
    let request = tonic::Request::new(req);
    let response = client.get_key(request).await?;
    Ok(response.into_inner())
}

// Function to send a key update request to a client
async fn key_update_request(
    client: Arc<Mutex<KeyManagementClient<Channel>>>,
    req: kmn_poc::KeyUpdateRequest
) -> Result<kmn_poc::KeyUpdateResponse, Box<dyn Error + Send + Sync>> {
    let mut client = client.lock().await;
    let request = tonic::Request::new(req);
    let response = client.key_update(request).await?;
    Ok(response.into_inner())
}

// Function to send a sign online request to a client
async fn sign_online_request(
    client: Arc<Mutex<KeyManagementClient<Channel>>>,
    req: kmn_poc::SignOnlineRequest
) -> Result<kmn_poc::SignOnlineResponse, Box<dyn Error + Send + Sync>> {
    let mut client = client.lock().await;
    let request = tonic::Request::new(req);
    let response = client.sign_online(request).await?;
    Ok(response.into_inner())
}

// Function to send a generate key online request to a client
async fn generate_key_request(
    client: Arc<Mutex<KeyManagementClient<Channel>>>,
    req: kmn_poc::GenerateKeyRequest
) -> Result<kmn_poc::GenerateKeyResponse, Box<dyn Error + Send + Sync>> {
    let mut client = client.lock().await;
    let request = tonic::Request::new(req);
    let response = client.generate_key(request).await?;
    Ok(response.into_inner())
}

async fn assign_key_request_to_client(
    client: Arc<Mutex<KeyManagementClient<Channel>>>,
    req: kmn_poc::AssignKeyRequest
) -> Result<kmn_poc::AssignKeyResponse, Status> {
    let mut client = client.lock().await;
    let request = tonic::Request::new(req);
    let response = client.assign_key(request).await?;
    Ok(response.into_inner())
}
