use crate::network::node::{ MessageType, NodeRequest, Request };

use super::{
    messages::{ NextMessageId, TssDelivery, TssOutgoing },
    node::NodeReceiver,
    node::NodeSender,
};
use anyhow::Error;
use cggmp21::round_based::{ Incoming, MessageDestination, Outgoing };
use futures::{ channel::mpsc, StreamExt };
use log::info;
use serde::{ de::DeserializeOwned, Serialize };
use tokio::task::JoinHandle;

pub async fn join_computation<M>(
    node_sender: NodeSender,
    node_receiver: NodeReceiver,
    index: u16,
    room_id: usize
) -> Result<
        (
            TssDelivery<M>,
            JoinHandle<Result<(), anyhow::Error>>,
            JoinHandle<Result<(), anyhow::Error>>,
        ),
        Error
    >
    where M: Clone + Send + 'static + Serialize + DeserializeOwned
{
    let (tx_to_protocol, rx_from_protocol) = mpsc::unbounded();
    let (tx_to_preserver, rx_from_delivery): (
        mpsc::UnboundedSender<Outgoing<Incoming<M>>>,
        mpsc::UnboundedReceiver<Outgoing<Incoming<M>>>,
    ) = mpsc::unbounded();

    let listening = tokio::spawn(node_receiver.listen(tx_to_protocol, room_id));
    let sending = tokio::spawn(send(rx_from_delivery, node_sender, room_id));

    let delivery = TssDelivery {
        incoming: rx_from_protocol,
        outgoing: TssOutgoing {
            party_idx: index,
            sender: tx_to_preserver,
            next_msg_id: NextMessageId::default(),
        },
    };
    Ok((delivery, listening, sending))
}

async fn send<M>(
    mut rx_from_delivery: mpsc::UnboundedReceiver<Outgoing<Incoming<M>>>,
    node_service: NodeSender,
    room_id: usize
) -> Result<(), Error>
    where M: Clone + Send + 'static + Serialize + DeserializeOwned
{
    while let Some(outgoing) = rx_from_delivery.next().await {
        let (msg_type, receiver) = match outgoing.recipient {
            MessageDestination::AllParties => (MessageType::Broadcast, 0),
            MessageDestination::OneParty(id) => (MessageType::P2P, id),
        };

        info!(
            "msg to be sent room {:?}, msg id: {:?}, receiver: {:?}, msg type {:?}",
            room_id,
            outgoing.msg.id,
            receiver,
            msg_type
        );

        let _ = node_service.send_request(NodeRequest {
            request: Request {
                id: outgoing.msg.id,
                sender: outgoing.msg.sender,
                msg_type,
                msg: serde_json::to_vec(&outgoing.msg.msg).unwrap(),
            },
            index: receiver,
            room_id,
        });
    }

    Ok(())
}
