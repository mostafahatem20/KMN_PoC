use anyhow::{ Context, Error };
use cggmp21::{
    Signature,
    progress::PerfProfiler,
    round_based::MpcParty,
    supported_curves::Secp256r1,
    ExecutionId,
    DataToSign,
    KeyShare,
    Presignature,
    PartialSignature,
};
use rand::rngs::OsRng;
use sha2::Sha256;
use crate::network::{ network::join_computation, node::{ NodeReceiver, NodeSender } };
use anyhow::anyhow;

pub async fn run(
    sender: NodeSender,
    receiver: NodeReceiver,
    index: u16,
    key_share: KeyShare<Secp256r1>,
    parties_indexes_at_keygen: &[u16],
    message: &[u8],
    eid: &[u8],
    room_id: usize
) -> Result<Signature<Secp256r1>, Error> {
    let (delivery, listening, sending) = join_computation(
        sender,
        receiver,
        index,
        room_id
    ).await.context("join computation")?;

    let party = MpcParty::connected(delivery);

    let eid = ExecutionId::new(&eid);

    let mut profiler = PerfProfiler::new();

    let data_to_sign: DataToSign<Secp256r1> = DataToSign::digest::<Sha256>(message);

    let signature = cggmp21
        ::signing(eid, index, &parties_indexes_at_keygen, &key_share)
        .set_progress_tracer(&mut profiler);

    let output = signature.sign(&mut OsRng, party, data_to_sign).await.expect("sign failed");

    let _ = sending.await?;
    listening.abort();

    let report = profiler.get_report().context("get perf report")?;
    println!("Sign {}", report.display_io(false));

    Ok(output)
}

pub async fn pre_sign(
    sender: NodeSender,
    receiver: NodeReceiver,
    index: u16,
    key_share: KeyShare<Secp256r1>,
    parties_indexes_at_keygen: &[u16],
    eid: &[u8],
    room_id: usize
) -> Result<Presignature<Secp256r1>, Error> {
    let (delivery, listening, sending) = join_computation(
        sender,
        receiver,
        index,
        room_id
    ).await.context("join computation")?;

    let party = MpcParty::connected(delivery);

    let eid = ExecutionId::new(&eid);

    let mut profiler = PerfProfiler::new();

    let pre_signature = cggmp21
        ::signing(eid, index, &parties_indexes_at_keygen, &key_share)
        .set_progress_tracer(&mut profiler);

    let output = pre_signature.generate_presignature(&mut OsRng, party).await.expect("sign failed");

    let _ = sending.await?;
    listening.abort();

    let report = profiler.get_report().context("get perf report")?;
    println!("Presign {}", report.display_io(false));

    Ok(output)
}

pub async fn load_pre_sign(filename: String) -> Result<Presignature<Secp256r1>, Error> {
    let file = tokio::fs
        ::read_to_string(filename).await
        .map_err(|e| anyhow!("reading presign terminated with err: {}", e))?;
    serde_json::from_str(&file).map_err(|e| anyhow!("decoding presign terminated with err: {}", e))
}

pub fn partial_sign(
    pre_signuature: Presignature<Secp256r1>,
    message: &[u8]
) -> Result<PartialSignature<Secp256r1>, Error> {
    let data_to_sign: DataToSign<Secp256r1> = DataToSign::digest::<Sha256>(message);

    let partial_signuature = Presignature::issue_partial_signature(pre_signuature, data_to_sign);

    Ok(partial_signuature)
}

pub async fn load_partial_sign(filename: String) -> Result<PartialSignature<Secp256r1>, Error> {
    let file = tokio::fs
        ::read_to_string(filename).await
        .map_err(|e| anyhow!("reading partial sign terminated with err: {}", e))?;
    serde_json
        ::from_str(&file)
        .map_err(|e| anyhow!("decoding partial sign terminated with err: {}", e))
}

pub fn combine_sign(
    partial_signatures: &[PartialSignature<Secp256r1>]
) -> Result<Signature<Secp256r1>, Error> {
    PartialSignature::combine(partial_signatures).ok_or_else(||
        anyhow!("combining partial terminated")
    )
}
