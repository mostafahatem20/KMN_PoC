use cggmp21::{
    generic_ec::SecretScalar,
    key_share::{ reconstruct_secret_key, ReconstructError },
    supported_curves::Secp256r1,
    KeyShare,
};
use anyhow::{ anyhow, Error };

pub async fn run(
    key_shares: &[KeyShare<Secp256r1>]
) -> Result<SecretScalar<Secp256r1>, ReconstructError> {
    reconstruct_secret_key(key_shares)
}

pub async fn load(filename: String) -> Result<SecretScalar<Secp256r1>, Error> {
    let file = tokio::fs
        ::read_to_string(filename).await
        .map_err(|e| anyhow!("reading secret scalar terminated with err: {}", e))?;
    serde_json
        ::from_str(&file)
        .map_err(|e| anyhow!("decoding secret scalar terminated with err: {}", e))
}
