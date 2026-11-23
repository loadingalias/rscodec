//! The timing evidence of the base-N codecs of rscodec (C103). Run it as
//! `cargo run --release -p rscodec-ct -- <command>`.
//!
//! - `paths [case]`: each call runs under the CPU counters of instructions and branches. Each length
//!   and outcome must give one count over random inputs, in each class and on each kernel of the
//!   CPU, or only for the cases whose names start with `case`. It needs Linux.
//! - `scan [binary] [x86_64|aarch64]`: reads the disassembly of a binary, by default this one. It finds loads from a
//!   constant table with an index register, indirect jumps and calls, instructions of the secret
//!   code outside the DOIT list of Intel or the DIT list of Arm, and writes of `PSTATE.DIT`.
//! - `stack [public]`: runs each secret call on a private stack, and finds no run of 8 bytes of its
//!   secret bytes or 16 characters of its secret text in that stack after the call. With `public`,
//!   the control: the same search over the public calls must find a residue.
//! - `dudect [samples] [random|zero|aa] [case] [bytes] [calls]`: compares the time of one fixed input with
//!   random inputs of the same length, for each secret function. The fixed input is random data, or
//!   zeros, or with `aa` there is none: the A/A control.
//!
//! Each command prints its record. It exits with 1 when a check fails, and with 2 when this host
//! cannot run the check.

#![expect(
    clippy::print_stdout,
    reason = "the harness is a command-line tool, and standard output is its record"
)]

mod cases;
mod dudect;
mod lists;
mod paths;
mod random;
mod scan;
mod stack;

use std::{env, process::ExitCode};

/// The usage of the harness.
const USAGE: &str = "usage: rscodec-ct paths [case] | scan [binary] [isa] | stack [public] | dudect [samples] [random|zero|aa] [case] [bytes] [calls]";

/// Why a command did not pass.
#[derive(Debug)]
enum Failure {
    /// A check found a defect, or a fact that a person must review.
    Check(String),
    /// This host cannot run the check, for example with no CPU counters.
    Host(String),
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let rest = args.get(1..).unwrap_or_default();
    let result = match args.first().map(String::as_str) {
        Some("paths") => paths::run(rest),
        Some("scan") => scan::run(rest),
        Some("stack") => stack::run(rest),
        Some("dudect") => dudect::run(rest),
        _ => Err(Failure::Host(USAGE.to_owned())),
    };
    match result {
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
