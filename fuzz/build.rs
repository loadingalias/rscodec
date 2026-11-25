//! Builds simdjson and its shim `oracle.cpp`. README.md gives the command that fetches simdjson.

use std::path::Path;

fn main() {
    for file in ["simdjson/simdjson.h", "simdjson/simdjson.cpp"] {
        assert!(
            Path::new(file).exists(),
            "fuzz/{file} is missing: fuzz/README.md gives the command that fetches it"
        );
    }
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .define("SIMDJSON_EXCEPTIONS", "0")
        .include("simdjson")
        .file("simdjson/simdjson.cpp")
        .file("oracle.cpp")
        .warnings(false)
        .compile("oracle");
    println!("cargo::rerun-if-changed=oracle.cpp");
    println!("cargo::rerun-if-changed=simdjson");
}
