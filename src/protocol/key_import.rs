use cggmp21::{
    generic_ec::{ NonZero, SecretScalar },
    security_level::SecurityLevel128,
    supported_curves::Secp256r1,
    IncompleteKeyShare,
};
use rand::rngs::OsRng;
use anyhow::anyhow;
use anyhow::Error;

pub async fn run(
    secret_key_to_be_imported: SecretScalar<Secp256r1>,
    threshold: u16,
    number_of_parties: u16
) -> Result<Vec<IncompleteKeyShare<Secp256r1>>, Error> {
    let mut rng = OsRng;

    let secret_key = NonZero::<SecretScalar<Secp256r1>>
        ::from_secret_scalar(secret_key_to_be_imported)
        .ok_or_else(|| anyhow!("NonZero secret from secret scalar failed"))?;

    let key_shares = cggmp21::trusted_dealer
        ::builder::<Secp256r1, SecurityLevel128>(number_of_parties)
        .enable_crt(true)
        .set_threshold(Some(threshold))
        .set_shared_secret_key(secret_key)
        .generate_core_shares(&mut rng)?;

    Ok(key_shares)
}
