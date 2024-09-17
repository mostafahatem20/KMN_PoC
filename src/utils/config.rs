use serde::Deserialize;
use std::fs;
use anyhow::Error;
#[derive(Deserialize, Debug)]
pub struct Config {
    pub index: usize,
    pub database_username: String,
    pub database_password: String,
    pub database_host: String,
    pub database_port: usize,
    pub database_name: String,
    pub server_port: usize,
}
#[derive(Deserialize, Debug)]
pub struct SetupConfig {
    pub threshold: usize,
    pub number_of_parties: usize,
    pub topic: String,
    pub port: usize,
    pub urls: Vec<String>,  // Add a field for URLs
}

pub fn read_config(path: &str) -> Result<Config, Error> {
    // Read the YAML file from the specified path
    let config_content = fs::read_to_string(path)?;

    // Parse the YAML file into the Config struct
    let config: Config = serde_yaml::from_str(&config_content)?;

    // Return the parsed configuration
    Ok(config)
}

pub fn read_setup_config(path: &str) -> Result<SetupConfig, Error> {
    // Read the YAML file from the specified path
    let config_content = fs::read_to_string(path)?;

    // Parse the YAML file into the Config struct
    let config: SetupConfig = serde_yaml::from_str(&config_content)?;

    // Return the parsed configuration
    Ok(config)
}

pub fn create_postgres_url(config: &Config) -> String {
    format!(
        "postgres://{}:{}@{}:{}/{}?sslmode=disable",
        config.database_username,
        config.database_password,
        config.database_host,
        config.database_port,
        config.database_name
    )
}
