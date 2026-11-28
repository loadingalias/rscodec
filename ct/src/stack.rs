//! Run each secret operation on a painted stack. Search for runs of eight decoded bytes or 16 encoded characters.
//! Shorter residues fall below the detection threshold. The public control must find at least one residue.

use std::collections::HashSet;

use rscodec::Dit;

use crate::{
    Failure,
    cases::{Case, Class, Direction, Fill, Outcome},
    random::Random,
};

const STACK: usize = 1 << 18;

const PAINT: u8 = 0xa5;

const LENGTHS: [usize; 6] = [255, 256, 257, 4095, 4096, 4097];

const SEED: u64 = 0x5ac4;

struct Job<'a> {
    case: &'a Case,
    dit: &'a Dit,
    input: &'a [u8],
    out: &'a mut [u8],
}

pub(crate) fn run(args: &[String]) -> Result<(), Failure> {
    let class = match args.first().map(String::as_str) {
        Some("public") => Class::Public,
        _ => Class::Secret,
    };
    let cases: Vec<Case> = Case::all().into_iter().filter(|case| case.class == class).collect();
    let mut stack = vec![PAINT; STACK];
    let mut random = Random::new(SEED);
    let mut calls = 0_usize;
    let mut found = Vec::new();
    rscodec::with_dit(|dit| -> Result<(), Failure> {
        for case in &cases {
            for bytes in (8..=200).chain(LENGTHS) {
                let mut out = vec![0; case.out_len(bytes)?];
                for &outcome in case.outcomes() {
                    let Some(input) = case.input(bytes, outcome, Fill::Random, None, &mut random)? else {
                        continue;
                    };
                    stack.fill(PAINT);
                    switch::run_on(
                        &mut stack,
                        &mut Job {
                            case,
                            dit,
                            input: &input,
                            out: &mut out,
                        },
                    );
                    let (data, text) = match case.direction {
                        Direction::Encode => (&input[..], &out[..]),
                        Direction::Decode if outcome == Outcome::Valid => (&out[..], &input[..]),
                        Direction::Decode => (&[][..], &input[..]),
                    };
                    if let Some(at) = residue(&stack, data, text) {
                        found.push(format!(
                            "{} bytes={bytes} {outcome:?}: a run of the secret {} bytes under the top",
                            case.name,
                            STACK - at
                        ));
                    }
                    calls += 1;
                }
            }
            println!("stack {}: searched", case.name);
        }
        Ok(())
    })
    .map_err(|refusal| Failure::Host(refusal.to_string()))??;
    for line in &found {
        println!("stack residue: {line}");
    }
    println!(
        "stack: {calls} calls of the {class:?} class, {} with a residue",
        found.len()
    );
    if class == Class::Public {
        return if found.is_empty() {
            Err(Failure::Check(
                "the control found no residue of a public call".to_owned(),
            ))
        } else {
            Ok(())
        };
    }
    if found.is_empty() {
        Ok(())
    } else {
        Err(Failure::Check(format!(
            "{} calls left a run of their secret in the stack",
            found.len()
        )))
    }
}

fn residue(stack: &[u8], data: &[u8], text: &[u8]) -> Option<usize> {
    let mixed = |run: &[u8]| run.iter().any(|&byte| Some(&byte) != run.first());
    let data: HashSet<&[u8; 8]> = data.array_windows::<8>().filter(|run| mixed(&run[..])).collect();
    let text: HashSet<&[u8; 16]> = text.array_windows::<16>().filter(|run| mixed(&run[..])).collect();
    let in_data = stack.array_windows::<8>().position(|run| data.contains(run));
    let in_text = stack.array_windows::<16>().position(|run| text.contains(run));
    in_data.into_iter().chain(in_text).min()
}

#[expect(
    unsafe_code,
    reason = "the asm moves the stack pointer to the top of an owned buffer for one C call, and restores it from a callee-saved register"
)]
mod switch {
    use core::{arch::asm, ffi::c_void, ptr};

    use super::Job;

    pub(super) fn run_on(stack: &mut [u8], job: &mut Job<'_>) {
        let end = stack.as_mut_ptr_range().end;
        let top = end.wrapping_sub(end.addr() % 16);
        let job = ptr::from_mut(job).cast::<c_void>();
        let call: extern "C" fn(*mut c_void) = trampoline;
        // SAFETY: `top` is the 16-byte aligned end of `stack`, which is far larger than the stack of
        // the call, and nothing else uses `stack` during it. The callee-saved register keeps the
        // stack pointer of this function, and the asm restores it after the call.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            asm!(
                "mov r12, rsp",
                "mov rsp, {top}",
                "call {call}",
                "mov rsp, r12",
                top = in(reg) top,
                call = in(reg) call,
                in("rdi") job,
                out("r12") _,
                clobber_abi("C"),
            );
        }
        // SAFETY: as above.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            asm!(
                "mov x20, sp",
                "mov sp, {top}",
                "blr {call}",
                "mov sp, x20",
                top = in(reg) top,
                call = in(reg) call,
                in("x0") job,
                out("x20") _,
                clobber_abi("C"),
            );
        }
    }

    extern "C" fn trampoline(job: *mut c_void) {
        // SAFETY: `run_on` passes a pointer to a live, exclusive `Job`.
        let job = unsafe { &mut *job.cast::<Job<'_>>() };
        job.case.call(job.dit, job.input, job.out);
    }
}
