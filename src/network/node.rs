use std::{ collections::HashMap, time::Duration };
use cggmp21::round_based::{ Incoming, MessageType as DfnsMessageType, MsgId, PartyIndex };
use futures::{ channel::mpsc::{ self }, FutureExt, SinkExt, StreamExt };
use libp2p::{
    mdns,
    noise,
    request_response::{ self, json, ProtocolSupport },
    swarm::{ NetworkBehaviour, SwarmEvent },
    tcp,
    yamux,
    Multiaddr,
    PeerId,
    StreamProtocol,
    Swarm,
    gossipsub,
    identify,
};
use anyhow::{ Error, Context };
use serde::{ de::DeserializeOwned, Serialize };
use tokio::{
    io,
    select,
    sync::mpsc::{ unbounded_channel, UnboundedReceiver, UnboundedSender },
    time::sleep,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{ Hash, Hasher };

use crate::utils::utils::all_unique;
use log::{ error, info };
#[derive(Debug, serde::Serialize, serde::Deserialize, Eq, PartialEq, Clone)]
pub enum MessageType {
    /// Message was broadcasted
    Broadcast,
    /// P2P message
    P2P,
}
#[derive(Debug, serde::Serialize, serde::Deserialize, Eq, PartialEq, Clone)]
pub struct RequestWithReceiver {
    request: Request,
    room_id: usize,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Eq, PartialEq, Clone)]
pub struct Request {
    /// Index of a message
    pub id: MsgId,
    /// Index of a party who sent the message
    pub sender: PartyIndex,
    /// Indicates whether it's a broadcast message (meaning that this message is received by all the
    /// parties), or p2p (private message sent by `sender`)
    pub msg_type: MessageType,
    /// Received message
    pub msg: Vec<u8>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Response {
    pub msg_id: MsgId,
    pub room_id: usize,
    pub peer_id: PeerId,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct NodeInfo {
    pub index: usize,
    pub peer_id: PeerId,
}

#[derive(NetworkBehaviour)]
struct MyBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
    request_response: request_response::json::Behaviour<RequestWithReceiver, Response>,
    identify: identify::Behaviour,
}
#[derive(Debug)]
pub struct NodeRequest {
    pub request: Request,
    pub index: u16,
    pub room_id: usize,
}
pub struct Node {
    party_index: usize,
    number_of_parties: usize,
    peers_connected: usize,
    sorted_peers: Vec<PeerId>,
    consensus_topic: gossipsub::IdentTopic,
    swarm: Swarm<MyBehaviour>,
    receivers: HashMap<usize, UnboundedReceiver<NodeRequest>>,
    senders: HashMap<usize, UnboundedSender<Request>>,
    peer_addresses: HashMap<PeerId, Vec<Multiaddr>>,
    connection_signal: Option<UnboundedSender<()>>,
    signal_sent: bool,
}
pub struct NodeReceiver {
    from_worker: UnboundedReceiver<Request>,
}

pub struct NodeSender {
    to_worker: UnboundedSender<NodeRequest>,
}

impl Node {
    pub fn new(
        index: usize,
        connection_signal: Option<UnboundedSender<()>>,
        number_of_parties: usize,
        topic: String
    ) -> Result<Node, Error> {
        let mut swarm = libp2p::SwarmBuilder
            ::with_new_identity()
            .with_tokio()
            .with_tcp(tcp::Config::default(), noise::Config::new, yamux::Config::default)?
            .with_behaviour(|key| {
                // To content-address message, we can take the hash of message and use it as an ID.
                let message_id_fn = |message: &gossipsub::Message| {
                    let mut s = DefaultHasher::new();
                    message.data.hash(&mut s);
                    gossipsub::MessageId::from(s.finish().to_string())
                };

                // Set a custom gossipsub configuration
                let gossipsub_config = gossipsub::ConfigBuilder
                    ::default()
                    .heartbeat_interval(Duration::from_secs(2)) // This is set to aid debugging by not cluttering the log space
                    .validation_mode(gossipsub::ValidationMode::Strict) // This sets the kind of message validation. The default is Strict (enforce message signing)
                    .message_id_fn(message_id_fn) // content-address messages. No two messages of the same content will be propagated.
                    .build()
                    .map_err(|msg| io::Error::new(io::ErrorKind::Other, msg))?; // Temporary hack because `build` does not return a proper `std::error::Error`.

                // build a gossipsub network behaviour
                let gossipsub = gossipsub::Behaviour::new(
                    gossipsub::MessageAuthenticity::Signed(key.clone()),
                    gossipsub_config
                )?;

                let mdns_config = mdns::Config {
                    query_interval: Duration::from_secs(2), // Increase frequency
                    ..Default::default()
                };

                let mdns = mdns::tokio::Behaviour::new(mdns_config, key.public().to_peer_id())?;

                let request_response = json::Behaviour::<RequestWithReceiver, Response>::new(
                    [(StreamProtocol::new("/my-json-protocol"), ProtocolSupport::Full)],
                    request_response::Config::default()
                );

                let identify = identify::Behaviour::new(
                    identify::Config::new("/ipfs/id/1.0.0".to_string(), key.public())
                );
                Ok(MyBehaviour { mdns, request_response, gossipsub, identify })
            })?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
            .build();

        // Create a Gossipsub topic
        let topic = gossipsub::IdentTopic::new(topic);
        // subscribes to our topic
        swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        let mut sorted_peers: Vec<PeerId> = Vec::new();
        sorted_peers.resize(number_of_parties, swarm.local_peer_id().clone());
        let worker = Node {
            party_index: index,
            number_of_parties,
            peers_connected: 0,
            sorted_peers,
            consensus_topic: topic.clone(),
            peer_addresses: HashMap::new(),
            swarm,
            receivers: HashMap::new(),
            senders: HashMap::new(),
            connection_signal,
            signal_sent: false,
        };

        Ok(worker)
    }

    pub fn add_receiver_sender(&mut self, room_id: usize) -> (NodeReceiver, NodeSender) {
        let (network_sender_in, network_receiver_in) = unbounded_channel();
        let (network_sender_out, network_receiver_out) = unbounded_channel();

        let receiver = NodeReceiver {
            from_worker: network_receiver_out,
        };
        let sender = NodeSender {
            to_worker: network_sender_in,
        };
        self.receivers.insert(room_id, network_receiver_in);
        self.senders.insert(room_id, network_sender_out);

        (receiver, sender)
    }
    pub async fn run(mut self) -> Result<(), Error> {
        let mut room_id_to_remove: Option<usize> = None;
        let mut pending_requests: Vec<(PeerId, MsgId, usize)> = Vec::new();

        loop {
            let peers_connected = self.swarm
                .behaviour_mut()
                .gossipsub.all_peers()
                .filter(|(_, topics)| topics.contains(&&self.consensus_topic.hash()))
                .count();
            info!("peers {:?}, self {:?}", peers_connected, self.peers_connected);
            if
                peers_connected == self.number_of_parties - 1 &&
                peers_connected != self.peers_connected
            {
                let node_info = NodeInfo {
                    index: self.party_index,
                    peer_id: self.swarm.local_peer_id().clone(),
                };
                info!("Publish index and id");
                if
                    let Err(e) = self.swarm
                        .behaviour_mut()
                        .gossipsub.publish(
                            self.consensus_topic.clone(),
                            serde_json::to_vec(&node_info).unwrap()
                        )
                {
                    info!("Publish error: {e:?}");
                }
                self.peers_connected = peers_connected;
            }
            let is_sorted_peers_unique = all_unique(&self.sorted_peers);
            if
                is_sorted_peers_unique &&
                self.sorted_peers.len() == self.number_of_parties &&
                !self.signal_sent
            {
                if let Some(connection_signal) = &self.connection_signal {
                    connection_signal.send(()).unwrap();
                    self.signal_sent = true; // Set the flag to true after sending the signal
                }
            }
            match room_id_to_remove {
                Some(v) => {
                    self.receivers.remove(&v);
                    room_id_to_remove = None;
                }
                None => {}
            }
            let receiver_futures: Vec<_> = self.receivers
                .iter_mut()
                .map(|(&room_id, rx)| {
                    let fut = rx.recv();
                    (room_id, fut)
                })
                .collect();

            info!("Receivers {:?}", receiver_futures.len());
            info!("Pending Requests {:?}", pending_requests);

            if receiver_futures.is_empty() && pending_requests.is_empty() {
                sleep(Duration::from_millis(100)).await;
                break;
            }

            if !receiver_futures.is_empty() {
                select! {
                    event = self.swarm.select_next_some() => match event {
                        SwarmEvent::Behaviour(MyBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                            for (peer_id, multiaddr) in list {
                                self.peer_addresses.entry(peer_id).or_insert_with(Vec::new).push(multiaddr.clone());
                                self.swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                            }
                        },
                        SwarmEvent::Behaviour(MyBehaviourEvent::Mdns(mdns::Event::Expired(list))) => {
                            for (peer_id, _multiaddr) in list {
                                self.peer_addresses.remove(&peer_id);
                                self.swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                            }
                        },
                        SwarmEvent::Behaviour(MyBehaviourEvent::RequestResponse(request_response::Event::Message { peer: _peer, message })) => {
                            match message{
                                request_response::Message::Request { request_id: _request_id, request, channel } => {
                                    info!("Received request room {:?}, msg id {:?}, sender {:?}", request.room_id, request.request.id, request.request.sender);
                                    let msg_id = request.request.id;
                                    self.senders.get_mut(&request.room_id).unwrap().send(request.request)?;
                                    let response = Response {
                                        msg_id,
                                        room_id: request.room_id,
                                        peer_id: self.swarm.local_peer_id().clone()
                                    };
                                    
                                    if let Err(e) = self.swarm.behaviour_mut().request_response.send_response(channel, response) {
                                        // Handle the error gracefully
                                        error!("Failed to send response: {:?}", e);
                                        // You might also want to take some recovery actions here
                                    }
                                },
                                request_response::Message::Response { request_id: _request_id, response } => {
                                    info!("Received response peer_id {:?} msg id {:?}, room {:?}",  response.peer_id, response.msg_id, response.room_id);
                                    // Find the position of the first occurrence of (msg_id, room_id)
                                    if let Some(pos) = pending_requests.iter().position(|x| *x == (response.peer_id, response.msg_id, response.room_id)) {
                                        pending_requests.remove(pos); // Remove the first match
                                    }
                                },
                            }
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                            propagation_source: _peer_id,
                            message_id: _id,
                            message,
                        })) => {
                            if !self.signal_sent {
                                let peer_info: NodeInfo = serde_json::from_slice(&message.data).unwrap();
                                info!("Received publish {:?}", peer_info);
                                self.sorted_peers[peer_info.index] = peer_info.peer_id;
                            }
                        } 
                        SwarmEvent::Behaviour(MyBehaviourEvent::RequestResponse(request_response::Event::ResponseSent { peer: _peer, request_id:_request_id })) => {
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::RequestResponse(request_response::Event::InboundFailure { peer: _peer, request_id: _request_id, error: _error })) => {
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::RequestResponse(request_response::Event::OutboundFailure { peer: _peer, request_id: _request_id, error: _error })) => {
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Identify(identify::Event::Sent { peer_id: _peer_id, .. })) => {
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Identify(identify::Event::Received { info: _info, .. })) => {
                        }
                        SwarmEvent::NewListenAddr { address, .. } => {
                            info!("Listening on {:?}", address)
                        },
                        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                            info!("Connection Established {:?}", peer_id);
                        },
                        SwarmEvent::ConnectionClosed { peer_id, .. } => {
                            info!("Connection Closed {:?}", peer_id);
                        }
                        _ => {}
                 },
                    request = futures::future::select_all(receiver_futures.into_iter().map(|(room_id, fut)| {
                        Box::pin(async move {
                            let result = fut.await;
                            (room_id, result)
                        })
                    })).fuse() => {
                    match request {
                        ((room_id, Some(request)), _, _) => {
                            info!(
                                "local room {:?} msg to be sent (Node) remote room {:?}, msg id: {:?}, receiver: {:?}, msg type {:?}",
                                room_id,
                                request.room_id,
                                request.request.id,
                                request.index,
                                request.request.msg_type
                            );
                            let behaviour = self.swarm.behaviour_mut();
                            let peers: Vec<_> = self.sorted_peers.clone();
                            if request.request.msg_type == MessageType::Broadcast {
                                for (i, peer_id) in peers.iter().enumerate() {
                                    if i != self.party_index {
                                    pending_requests.push((peer_id.clone(), request.request.id, request.room_id));
                                    behaviour.request_response.send_request(peer_id, RequestWithReceiver{
                                        room_id: request.room_id,
                                        request: request.request.clone()
                                    });
                                }
                                }
                            } else {
                                if let Some(peer_id) = self.sorted_peers.get(request.index as usize) {
                                    pending_requests.push((peer_id.clone(), request.request.id, request.room_id));
                                    behaviour.request_response.send_request(peer_id, RequestWithReceiver{
                                        room_id: request.room_id,
                                        request: request.request.clone()
                                    });
                                }
                            }
                        },
                        ((room_id, None), _, _) => {
                            // The receiver for the given room_id is closed, remove it from the map
                            room_id_to_remove = Some(room_id);
                        },
                    }
                }
                }
            } else {
                select! {
                    event = self.swarm.select_next_some() => match event {
                        SwarmEvent::Behaviour(MyBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                            for (peer_id, multiaddr) in list {
                                self.peer_addresses.entry(peer_id).or_insert_with(Vec::new).push(multiaddr.clone());
                                self.swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                            }
                        },
                        SwarmEvent::Behaviour(MyBehaviourEvent::Mdns(mdns::Event::Expired(list))) => {
                            for (peer_id, _multiaddr) in list {
                                self.peer_addresses.remove(&peer_id);
                                self.swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                            }
                        },
                        SwarmEvent::Behaviour(MyBehaviourEvent::RequestResponse(request_response::Event::Message { peer: _peer, message })) => {
                            match message{
                                request_response::Message::Request { request_id: _request_id, request, channel } => {
                                    info!("Received request room {:?}, msg id {:?}, sender {:?}", request.room_id, request.request.id, request.request.sender);
                                    let msg_id = request.request.id;
                                    self.senders.get_mut(&request.room_id).unwrap().send(request.request)?;
                                    let response = Response {
                                        msg_id,
                                        room_id: request.room_id,
                                        peer_id: self.swarm.local_peer_id().clone()
                                    };
                                    
                                    if let Err(e) = self.swarm.behaviour_mut().request_response.send_response(channel, response) {
                                        // Handle the error gracefully
                                        error!("Failed to send response: {:?}", e);
                                        // You might also want to take some recovery actions here
                                    }
                                },
                                request_response::Message::Response { request_id: _request_id, response } => {
                                    info!("Received response peer_id {:?} msg id {:?}, room {:?}",  response.peer_id, response.msg_id, response.room_id);
                                    // Find the position of the first occurrence of (msg_id, room_id)
                                    if let Some(pos) = pending_requests.iter().position(|x| *x == (response.peer_id, response.msg_id, response.room_id)) {
                                        pending_requests.remove(pos); // Remove the first match
                                    }
                                },
                            }
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                            propagation_source: _peer_id,
                            message_id: _id,
                            message,
                        })) => {
                            if !self.signal_sent {
                                let peer_info: NodeInfo = serde_json::from_slice(&message.data).unwrap();
                                info!("Received publish {:?}", peer_info);
                                self.sorted_peers[peer_info.index] = peer_info.peer_id;
                            }
                        } 
                        SwarmEvent::Behaviour(MyBehaviourEvent::RequestResponse(request_response::Event::ResponseSent { peer: _peer, request_id:_request_id })) => {
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::RequestResponse(request_response::Event::InboundFailure { peer: _peer, request_id: _request_id, error: _error })) => {
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::RequestResponse(request_response::Event::OutboundFailure { peer: _peer, request_id: _request_id, error: _error })) => {
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Identify(identify::Event::Sent { peer_id: _peer_id, .. })) => {
                        }
                        SwarmEvent::Behaviour(MyBehaviourEvent::Identify(identify::Event::Received { info: _info, .. })) => {
                        }
                        SwarmEvent::NewListenAddr { address, .. } => {
                            info!("Listening on {:?}", address)
                        },
                        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                            info!("Connection Established {:?}", peer_id);
                        },
                        SwarmEvent::ConnectionClosed { peer_id, .. } => {
                            info!("Connection Closed {:?}", peer_id);
                        }
                        _ => {}
                 },
                }
            }
        }
        info!("node done");
        Ok(())
    }
}

impl NodeSender {
    pub fn send_request(&self, message: NodeRequest) -> Result<(), Error> {
        self.to_worker.send(message)?;
        Ok(())
    }
}

impl NodeReceiver {
    pub async fn listen<M>(
        mut self,
        mut tx_to_protocol: mpsc::UnboundedSender<Result<Incoming<M>, std::io::Error>>,
        room_id: usize
    ) -> Result<(), Error>
        where M: Clone + Send + 'static + Serialize + DeserializeOwned
    {
        loop {
            select! {
                rpc_message = self.from_worker.recv() => match rpc_message {
                    // Inbound requests
                    Some(request) => {
                        info!("Received msg in room {:?}, msg id {:?}", room_id, request.id);
                        let msg = serde_json::from_slice::<M>(&request.msg).context("failed to deserialize Msg")?;
                        let incoming = Incoming {
                            id: request.id,
                            sender: request.sender,
                            msg_type: if request.msg_type == MessageType::P2P {
                                DfnsMessageType::P2P
                            } else {
                                DfnsMessageType::Broadcast
                            },
                            msg,
                        };
                
                    tx_to_protocol.send(Ok(incoming)).await?;
                    }
                    None => {break;}
                }
            }
        }
        print!("done listening");
        Ok(())
    }
}
