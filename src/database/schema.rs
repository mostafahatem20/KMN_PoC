// src/schema.rs

diesel::table! {
    keys (id) {
        id -> Uuid,
        key_share -> Varchar,
    }
}

diesel::table! {
    presignatures (id) {
        id -> Uuid,
        key_id -> Uuid,
        pre_signature -> Varchar,
        parties_indexes -> Array<SmallInt>,
    }
}

diesel::allow_tables_to_appear_in_same_query!(
    keys,
    presignatures
);
