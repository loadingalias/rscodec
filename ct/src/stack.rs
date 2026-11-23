//! The painted stack. The secret functions clear each temporary copy of their input and result
//! before they return (C106). Thus each secret call runs on a private, painted stack, and after it
//! returns, no run of 8 of its secret bytes and no run of 16 of its secret characters is in that
//! stack. A shorter run is not found, so an input of under 8 bytes has no check.
//!
//! `stack public` is the control: the public functions do not clear their copies, so the same
//! search must find a residue of some public call.

use std::collections::HashSet;

use rscodec::Dit;

use crate::{
    Failure,
    cases::{Case, Class, Direction, Fill, Outcome},
    random::Random,
};

/// The bytes of the private stack: far more than a call uses.
const STACK: usize = 1 << 18;

/// The paint of the private stack.
const PAINT: u8 = 0xa5;

/// The lengths of the data in bytes, after the short lengths 8 to 200.
const LENGTHS: [usize; 6] = [255, 256, 257, 4095, 4096, 4097];

/// The seed of the inputs.
const SEED: u64 = 0x5ac4;

/// The work of one call on the private stack.
struct Job<'a> {
    case: &'a Case,
    dit: &'a Dit,
    input: &'a [u8],
    out: &'a mut [u8],
}

/// Runs each secret case, or with `public` each public case, on the private stack, and searches
/// the stack after it.
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
                    // After a refusal, the output holds zeros, which are no secret.
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

/// The offset in `stack` of the first run of 8 bytes of `data` or of 16 characters of `text`. A
/// run of one repeated byte is left out, because the paint or a cleared slot can hold it.
fn residue(stack: &[u8], data: &[u8], text: &[u8]) -> Option<usize> {
    let mixed = |run: &[u8]| run.iter().any(|&byte| Some(&byte) != run.first());
    let data: HashSet<&[u8; 8]> = data.array_windows::<8>().filter(|run| mixed(&run[..])).collect();
    let text: HashSet<&[u8; 16]> = text.array_windows::<16>().filter(|run| mixed(&run[..])).collect();
    let in_data = stack.array_windows::<8>().position(|run| data.contains(run));
    let in_text = stack.array_windows::<16>().position(|run| text.contains(run));
    in_data.into_iter().chain(in_text).min()
}

/// The switch of the stack pointer to the private stack for one call.
#[expect(
    unsafe_code,
    reason = "the asm moves the stack pointer to the top of an owned buffer for one C call, and restores it from a callee-saved register"
)]
mod switch {
    use core::{arch::asm, ffi::c_void, ptr};

    use super::Job;

    /// Runs `job` with the stack pointer at the top of `stack`, rounded down to 16 bytes.
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

    /// Runs the call of the job at `job`.
    extern "C" fn trampoline(job: *mut c_void) {
        // SAFETY: `run_on` passes a pointer to a live, exclusive `Job`.
        let job = unsafe { &mut *job.cast::<Job<'_>>() };
        job.case.call(job.dit, job.input, job.out);
    }
}
