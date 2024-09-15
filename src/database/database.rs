use diesel::pg::PgConnection;
use diesel::prelude::*;
use uuid::Uuid;
use crate::database::models::NewPresignature;

use super::models::{ Key, NewKey, PreSignature };
use diesel::sql_query;
use diesel::result::Error;

pub fn establish_connection(database_url: String) -> PgConnection {
    PgConnection::establish(&database_url).unwrap_or_else(|_|
        panic!("Error connecting to {}", database_url)
    )
}
pub fn create_tables_if_not_exists(conn: &mut PgConnection) -> Result<(), Error> {
    let create_keys_table_sql =
        r#"
        CREATE TABLE IF NOT EXISTS keys (
            id UUID PRIMARY KEY,
            key_share VARCHAR NOT NULL
        );
    "#;

    let create_presignatures_table_sql =
        r#"
        CREATE TABLE IF NOT EXISTS presignatures (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            key_id UUID REFERENCES keys(id) ON DELETE CASCADE,
            pre_signature VARCHAR NOT NULL,
            parties_indexes SMALLINT[] NOT NULL
        );
    "#;

    // Execute the SQL queries to create the tables if they don't exist
    sql_query(create_keys_table_sql).execute(conn)?;
    sql_query(create_presignatures_table_sql).execute(conn)?;

    Ok(())
}

impl Key {
    pub fn create_key(conn: &mut PgConnection, id: Uuid, key_share: &str) -> Key {
        use super::schema::keys;

        let new_key = NewKey { id, key_share };

        diesel
            ::insert_into(keys::table)
            .values(&new_key)
            .returning(Key::as_returning())
            .get_result(conn)
            .expect("Error saving new post")
    }
    pub fn get_all(conn: &mut PgConnection) -> Vec<Key> {
        use super::schema::keys::dsl::*;

        keys.order(id.asc()) // Order by id in ascending order
            .load::<Key>(conn)
            .expect("Error loading keys")
    }

    pub fn get_keys_with_no_presignatures(conn: &mut PgConnection) -> Vec<Key> {
        use super::schema::keys::dsl as keys_dsl;
        use super::schema::presignatures::dsl as presignatures_dsl;
        use diesel::prelude::*;

        keys_dsl::keys
            .left_join(
                presignatures_dsl::presignatures.on(keys_dsl::id.eq(presignatures_dsl::key_id))
            )
            .filter(presignatures_dsl::key_id.is_null()) // Keys with no presignatures
            .order(keys_dsl::id.asc()) // Order by key id
            .select((keys_dsl::id, keys_dsl::key_share)) // Select columns from the `keys` table
            .load::<Key>(conn)
            .expect("Error loading keys with no presignatures")
    }

    pub fn get_by_id(conn: &mut PgConnection, key_id: Uuid) -> Option<String> {
        use super::schema::keys::dsl::*;
        use diesel::prelude::*;

        keys.filter(id.eq(key_id))
            .select(key_share)
            .first::<String>(conn)
            .optional()
            .expect("Error loading key share")
    }

    pub fn update_key_share(
        conn: &mut PgConnection,
        key_id: Uuid,
        new_key_share: &str
    ) -> Result<usize, diesel::result::Error> {
        use super::schema::keys::dsl::*;
        use diesel::prelude::*;

        diesel
            ::update(keys.filter(id.eq(key_id)))
            .set(key_share.eq(new_key_share))
            .execute(conn)
    }
}

impl PreSignature {
    pub fn create_presignature<'a>(
        conn: &mut PgConnection,
        key_id: Uuid,
        pre_signature: &'a str,
        parties_indexes: &'a [i16]
    ) -> Self {
        use super::schema::presignatures;

        let new_presignature = NewPresignature {
            key_id,
            pre_signature,
            parties_indexes,
        };

        diesel
            ::insert_into(presignatures::table)
            .values(&new_presignature)
            .get_result(conn)
            .expect("Error saving new presignature")
    }

    pub fn find_by_key_id(
        conn: &mut PgConnection,
        search_key_id: Uuid
    ) -> Result<Vec<PreSignature>, Error> {
        use super::schema::presignatures::dsl::*;

        presignatures.filter(key_id.eq(search_key_id)).load::<PreSignature>(conn)
    }

    pub fn delete_by_key_id(
        conn: &mut PgConnection,
        target_key_id: Uuid
    ) -> Result<usize, diesel::result::Error> {
        use super::schema::presignatures::dsl::*;
        use diesel::prelude::*;

        diesel::delete(presignatures.filter(key_id.eq(target_key_id))).execute(conn)
    }
}
