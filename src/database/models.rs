use diesel::prelude::*;
use diesel::pg::Pg;
use uuid::Uuid;
use crate::database::schema::{ keys, presignatures };

#[derive(Queryable, Selectable, Debug, Clone)]
#[diesel(table_name = keys)]
#[diesel(check_for_backend(Pg))]
pub struct Key {
    pub id: Uuid,
    pub key_share: String,
}

#[derive(Insertable)]
#[diesel(table_name = keys)]
pub struct NewKey<'a> {
    pub id: Uuid,
    pub key_share: &'a str,
}

#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = presignatures)]
#[diesel(check_for_backend(Pg))]
pub struct PreSignature {
    pub id: Uuid,
    pub key_id: Uuid,
    pub pre_signature: String,
    pub parties_indexes: Vec<i16>,
}

#[derive(Insertable)]
#[diesel(table_name = presignatures)]
pub struct NewPresignature<'a> {
    pub key_id: Uuid,
    pub pre_signature: &'a str,
    pub parties_indexes: &'a [i16],
}
