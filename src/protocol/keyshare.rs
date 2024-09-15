use anyhow::Error;
use cggmp21::{
    key_share::{ AuxInfo, IncompleteKeyShare },
    supported_curves::Secp256r1,
    key_share::KeyShare,
};
use anyhow::anyhow;

pub fn run(
    incomplete_key_share: IncompleteKeyShare<Secp256r1>,
    aux_info: AuxInfo
) -> Result<KeyShare<Secp256r1>, Error> {
    let key_share = cggmp21::KeyShare::from_parts((incomplete_key_share, aux_info))?;
    Ok(key_share)
}

pub async fn load(filename: String) -> Result<KeyShare<Secp256r1>, Error> {
    let file = tokio::fs
        ::read_to_string(filename).await
        .map_err(|e| anyhow!("reading dirtykey terminated with err: {}", e))?;
    serde_json::from_str(&file).map_err(|e| anyhow!("decoding dirtykey terminated with err: {}", e))
}
