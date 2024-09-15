use std::error::Error;
use std::{ env, path::PathBuf };

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Compile server.proto and client.proto with file descriptor sets
    tonic_build
        ::configure()
        .file_descriptor_set_path(out_dir.join("kmn_poc_descriptor.bin"))
        .compile(
            &[
                "proto/server.proto", // Add server.proto
                "proto/client.proto", // Add client.proto
            ],
            &["proto"] // Specify the proto directory containing both .proto files
        )?;

    // Compile server.proto separately (without file descriptor set)
    tonic_build::compile_protos("proto/server.proto")?;

    // Compile client.proto separately (without file descriptor set)
    tonic_build::compile_protos("proto/client.proto")?;

    Ok(())
}
