use std::{ pin::Pin, sync::atomic::AtomicU64, task::{ Context, Poll } };

use cggmp21::round_based::{
    Delivery,
    Incoming,
    MessageDestination,
    MessageType,
    MsgId,
    Outgoing,
    PartyIndex,
};
use futures::{ channel::mpsc, executor::block_on, Sink, SinkExt };
use cggmp21::security_level::SecurityLevel128;
use cggmp21::keygen::msg::threshold::Msg as KGMsg;
use cggmp21::signing::msg::Msg as SMsg;
use cggmp21::key_refresh::AuxOnlyMsg;
use sha2::Sha256;
use cggmp21::supported_curves::Secp256r1;

pub type AuxMsg = AuxOnlyMsg<Sha256, SecurityLevel128>;
pub type KeyGenMsg = KGMsg<Secp256r1, SecurityLevel128, Sha256>;
pub type SigningMsg = SMsg<Secp256r1, Sha256>;

pub struct TssDelivery<M> {
    pub incoming: mpsc::UnboundedReceiver<Result<Incoming<M>, std::io::Error>>,
    pub outgoing: TssOutgoing<M>,
}

impl<M> Delivery<M> for TssDelivery<M> where M: Clone + Send + Unpin + 'static {
    type Receive = mpsc::UnboundedReceiver<Result<Incoming<M>, std::io::Error>>;
    type ReceiveError = std::io::Error;
    type Send = TssOutgoing<M>;
    type SendError = std::io::Error;

    fn split(self) -> (Self::Receive, Self::Send) {
        (self.incoming, self.outgoing)
    }
}

pub struct TssOutgoing<M> {
    pub party_idx: PartyIndex,
    pub sender: mpsc::UnboundedSender<Outgoing<Incoming<M>>>,
    pub next_msg_id: NextMessageId,
}

impl<M> Sink<Outgoing<M>> for TssOutgoing<M> where M: Clone + Send + 'static {
    type Error = std::io::Error;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(mut self: Pin<&mut Self>, msg: Outgoing<M>) -> Result<(), Self::Error> {
        let msg_type = match msg.recipient {
            MessageDestination::AllParties => MessageType::Broadcast,
            MessageDestination::OneParty(_) => MessageType::P2P,
        };

        let id = self.next_msg_id.next();
        let sender = self.party_idx;

        let future = self.sender
            .send(
                msg.map(|m| Incoming {
                    id,
                    sender,
                    msg_type,
                    msg: m,
                })
            );
            block_on(future).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
}

#[derive(Default)]
pub struct NextMessageId(AtomicU64);

impl NextMessageId {
    pub fn next(&self) -> MsgId {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}
