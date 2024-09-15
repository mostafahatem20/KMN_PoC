use anyhow::{ Context, Error };
use cggmp21::{
    key_share::IncompleteKeyShare,
    progress::PerfProfiler,
    round_based::MpcParty,
    supported_curves::Secp256r1,
    ExecutionId,
};
use rand::rngs::OsRng;
use anyhow::anyhow;
use crate::network::{ network::join_computation, node::{ NodeReceiver, NodeSender } };

pub async fn run(
    sender: NodeSender,
    receiver: NodeReceiver,
    index: u16,
    threshold: u16,
    number_of_parties: u16,
    eid: &[u8],
    room_id: usize
) -> Result<IncompleteKeyShare<Secp256r1>, Error> {
    let (delivery, listening, sending) = join_computation(
        sender,
        receiver,
        index,
        room_id
    ).await.context("join computation")?;

    let party = MpcParty::connected(delivery);

    let eid = ExecutionId::new(&eid);

    let mut profiler = PerfProfiler::new();

    let keygen = cggmp21
        ::keygen::<Secp256r1>(eid, index, number_of_parties)
        .set_progress_tracer(&mut profiler)
        .set_threshold(threshold);

    let output = keygen.start(&mut OsRng, party).await.expect("keygen failed");

    let _ = sending.await?;
    listening.abort();

    let report = profiler.get_report().context("get perf report")?;
    println!("Key Generation {}", report.display_io(false));

    Ok(output)
}

pub async fn load(filename: String) -> Result<IncompleteKeyShare<Secp256r1>, Error> {
    let file = tokio::fs
        ::read_to_string(filename).await
        .map_err(|e| anyhow!("reading dirtykey terminated with err: {}", e))?;
    serde_json::from_str(&file).map_err(|e| anyhow!("decoding dirtykey terminated with err: {}", e))
}
