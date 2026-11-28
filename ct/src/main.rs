//! Check base64, hex, and the JSON writer's secret base64url path.
//! Run `cargo run --locked --release -p rscodec-ct --config 'profile.release.overflow-checks=false' -- <command>`.
//!
//! - `paths [case]` compares hardware instruction and branch counts for each length and outcome. Requires Linux counters.
//! - `scan [binary] [x86_64|aarch64]` checks disassembly. The default binary is this executable.
//! - `stack [public]` searches a painted stack for secret residue. The public control must find residue.
//! - `dudect [samples] [random|zero|aa] [case] [bytes] [calls] [warm|unwarmed] [codec|buffers]`
//!   compares input classes at one decoded length. The A/A control gives both classes random inputs.
//!
//! In a scratch checkout, apply `ct/dudect-extra-call.patch`. Then `dudect 100000 aa hex-encode-secret 4096` must report one timing difference.
//!
//! Exit codes are 0 for a pass, 1 for a finding, and 2 when the host cannot run the check.

#![expect(
    clippy::print_stdout,
    reason = "the harness is a command-line tool, and standard output is its record"
)]

mod lists;
mod scan;

cfg_select! {
    any(target_arch = "x86_64", target_arch = "aarch64") => {
        mod cases;
        mod dudect;
        mod paths;
        mod random;
        // The x86_64 stack switch passes its argument in the System V ABI.
        #[cfg(not(all(windows, target_arch = "x86_64")))]
        mod stack;

        fn run(command: &str, args: &[String]) -> Result<(), Failure> {
            match command {
                "paths" => paths::run(args),
                "dudect" => dudect::run(args),
                #[cfg(not(all(windows, target_arch = "x86_64")))]
                "stack" => stack::run(args),
                _ => scan_or_refuse(command, args),
            }
        }
    }
    _ => {
        use scan_or_refuse as run;
    }
}

use std::{env, process::ExitCode};

const USAGE: &str = "usage: rscodec-ct paths [case] | scan [binary] [isa] | stack [public] | dudect [samples] [random|zero|aa] [case] [bytes] [calls] [warm|unwarmed] [codec|buffers]";

#[derive(Debug)]
enum Failure {
    Check(String),
    Host(String),
}

fn scan_or_refuse(command: &str, args: &[String]) -> Result<(), Failure> {
    match command {
        "scan" => scan::run(args),
        "paths" | "stack" | "dudect" => Err(Failure::Host(format!(
            "this {} build cannot run {command}",
            env::consts::ARCH
        ))),
        _ => Err(Failure::Host(USAGE.to_owned())),
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_default();
    match run(&command, &args.collect::<Vec<_>>()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Check(reason)) => {
            println!("FAIL: {reason}");
            ExitCode::from(1)
        }
        Err(Failure::Host(reason)) => {
            println!("NOT RUN: {reason}");
            ExitCode::from(2)
        }
    }
}
