pub mod protocol;
pub mod network;
pub mod utils;
pub mod database;
use cggmp21::key_share::AuxInfo;
use tokio::sync::OnceCell;

pub static DATABASE_URL: OnceCell<String> = OnceCell::const_new();
pub static TOPIC: OnceCell<String> = OnceCell::const_new();
pub static INDEX: OnceCell<String> = OnceCell::const_new();
pub static AUX_INFO: OnceCell<String> = OnceCell::const_new();
pub static THRESHOLD: OnceCell<String> = OnceCell::const_new();
pub static NUMBER_OF_PARTIES: OnceCell<String> = OnceCell::const_new();

pub fn set_database_url(url: String) {
    DATABASE_URL.set(url).expect("Failed to set DATABASE_URL");
}

pub fn get_database_url() -> &'static str {
    DATABASE_URL.get().expect("DATABASE_URL is not set")
}

pub fn set_topic(topic: String) {
    TOPIC.set(topic).expect("Failed to set TOPIC");
}

pub fn get_topic() -> &'static str {
    TOPIC.get().expect("TOPIC is not set")
}

pub fn set_index(index: usize) {
    INDEX.set(index.to_string()).expect("Failed to set INDEX");
}

pub fn get_index() -> usize {
    INDEX.get().expect("INDEX is not set").parse::<usize>().expect("INDEX to be a valid number")
}

pub fn set_aux_info(aux_info: &AuxInfo) {
    AUX_INFO.set(serde_json::to_string(aux_info).unwrap()).expect("AUX_INFO to set DATABASE_URL");
}

pub fn get_aux_info() -> &'static str {
    AUX_INFO.get().expect("AUX_INFO is not set")
}

pub fn set_threshold(threshold: usize) {
    THRESHOLD.set(threshold.to_string()).expect("Failed to set THRESHOLD");
}

pub fn get_threshold() -> usize {
    THRESHOLD.get()
        .expect("THRESHOLD is not set")
        .parse::<usize>()
        .expect("THRESHOLD to be a valid number")
}

pub fn set_number_of_parties(number: usize) {
    NUMBER_OF_PARTIES.set(number.to_string()).expect("Failed to set NUMBER_OF_PARTIES");
}

pub fn get_number_of_parties() -> usize {
    NUMBER_OF_PARTIES.get()
        .expect("NUMBER_OF_PARTIES is not set")
        .parse::<usize>()
        .expect("NUMBER_OF_PARTIES to be a valid number")
}
