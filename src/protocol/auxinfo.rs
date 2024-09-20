use anyhow::{ Context, Error };
use cggmp21::{ key_share::AuxInfo, progress::PerfProfiler, round_based::MpcParty, ExecutionId };
use rand::rngs::OsRng;

use crate::{
    network::{ network::join_computation, node::{ NodeReceiver, NodeSender } },
    utils::utils::MyPerfReport,
};
use anyhow::anyhow;
use log::info;

pub async fn run(
    sender: NodeSender,
    receiver: NodeReceiver,
    index: u16,
    number_of_parties: u16,
    eid: &[u8],
    room_id: usize
) -> Result<(AuxInfo, MyPerfReport), Error> {
    let (delivery, listening, sending) = join_computation(
        sender,
        receiver,
        index,
        room_id
    ).await.context("join computation")?;
    let pregenerated_primes = cggmp21::PregeneratedPrimes::generate(&mut OsRng);

    let party = MpcParty::connected(delivery);

    let eid = ExecutionId::new(&eid);

    let mut profiler = PerfProfiler::new();

    let aux_info = cggmp21
        ::aux_info_gen(eid, index, number_of_parties, pregenerated_primes)
        .set_progress_tracer(&mut profiler);

    let output = aux_info.start(&mut OsRng, party).await.expect("aux info failed");

    let _ = sending.await?;
    listening.abort();

    let report = profiler.get_report().context("get perf report")?;
    info!("Aux Info {}", report);

    Ok((output, MyPerfReport(report)))
}

pub async fn load(filename: String) -> Result<AuxInfo, Error> {
    let file = tokio::fs
        ::read_to_string(filename).await
        .map_err(|e| anyhow!("reading auxinfo terminated with err: {}", e))?;
    serde_json::from_str(&file).map_err(|e| anyhow!("decoding auxinfo terminated with err: {}", e))
}
pub fn load_from_string(aux_info: String) -> Result<AuxInfo, Error> {
    serde_json
        ::from_str(&aux_info)
        .map_err(|e| anyhow!("decoding auxinfo terminated with err: {}", e))
}
