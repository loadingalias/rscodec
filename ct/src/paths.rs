//! The path count. A branch on the data, or an exit at the first bad block, changes the number of
//! instructions or branches that a call runs. Thus each call runs under the CPU counters, in user
//! mode, and each length and outcome must give one count over its inputs: random data, zeros, and
//! ones, and for a refusal, a bad character at each kind of offset.

use core::hint::black_box;

use crate::{
    Failure,
    cases::{Case, Fill},
    random::Random,
};

/// The lengths of the data in bytes: each length up to 200, which covers each short path and two
/// blocks of each kernel with each tail, and the lengths around the larger blocks.
const LENGTHS: [usize; 13] = [
    255, 256, 257, 511, 512, 1023, 1024, 4095, 4096, 4097, 65_535, 65_536, 65_537,
];

/// The inputs of each length and outcome.
const INPUTS: usize = 24;

/// The counts of each input. The smallest count is the count of the call: an interrupt can only
/// add to a count.
const REPEATS: usize = 3;

/// The counts of an input whose count differs, before the check reports it.
const RECOUNTS: usize = 40;

/// The seed of the inputs, fixed so that a run repeats.
const SEED: u64 = 0xc103;

/// Runs the path count on each case of this CPU, or on each case whose name starts with `args[0]`.
pub(crate) fn run(args: &[String]) -> Result<(), Failure> {
    let counters = perf::Counters::open()?;
    let prefix = args.first().map_or("", String::as_str);
    let cases: Vec<Case> = Case::all()
        .into_iter()
        .filter(|case| case.name.starts_with(prefix))
        .collect();
    if cases.is_empty() {
        return Err(Failure::Host(format!("no case starts with {prefix}")));
    }
    let mut random = Random::new(SEED);
    let mut classes = 0_usize;
    let mut failures = Vec::new();
    rscodec::with_dit(|dit| -> Result<(), Failure> {
        for case in &cases {
            let mut checked = 0_usize;
            for bytes in (0..=200).chain(LENGTHS) {
                let mut out = vec![0; case.out_len(bytes)?];
                for &outcome in case.outcomes() {
                    let mut inputs = Vec::with_capacity(INPUTS);
                    for index in 0..INPUTS {
                        let (fill, at) = match index {
                            0 => (Fill::Zero, Some(0)),
                            1 => (Fill::Ones, Some(usize::MAX)),
                            _ => (Fill::Random, None),
                        };
                        if let Some(input) = case.input(bytes, outcome, fill, at, &mut random)? {
                            inputs.push(input);
                        }
                    }
                    let Some(first) = inputs.first() else { continue };
                    // The first call of a function chooses its kernel.
                    black_box(case.call(dit, first, &mut out));
                    let count = |input: &[u8], out: &mut [u8], repeats| {
                        least(&counters, repeats, || case.call(dit, input, out))
                    };
                    let mut counts = Vec::with_capacity(inputs.len());
                    for input in &inputs {
                        counts.push(count(input, &mut out, REPEATS)?);
                    }
                    let mut distinct = counts.clone();
                    distinct.sort_unstable();
                    distinct.dedup();
                    if distinct.len() > 1 {
                        // Count each input again, many times, so that no interrupt stays in its least count.
                        for (input, slot) in inputs.iter().zip(&mut counts) {
                            *slot = count(input, &mut out, RECOUNTS)?;
                        }
                        distinct.clone_from(&counts);
                        distinct.sort_unstable();
                        distinct.dedup();
                    }
                    if distinct.len() > 1 {
                        failures.push(format!("{} bytes={bytes} {outcome:?}: counts {distinct:?}", case.name));
                    }
                    checked += 1;
                }
            }
            println!("paths {} {:?}: {checked} lengths and outcomes", case.name, case.class);
            classes += checked;
        }
        Ok(())
    })
    .map_err(|refusal| Failure::Host(refusal.to_string()))??;
    for failure in &failures {
        println!("paths differ: {failure}");
    }
    println!(
        "paths: {classes} lengths and outcomes, {} with more than one count",
        failures.len()
    );
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Failure::Check(format!(
            "{} lengths and outcomes have more than one path",
            failures.len()
        )))
    }
}

/// The least instructions and branches of `repeats` runs of `call`.
fn least(counters: &perf::Counters, repeats: usize, mut call: impl FnMut() -> usize) -> Result<[u64; 2], Failure> {
    let mut least = [u64::MAX; 2];
    for _ in 0..repeats {
        let counts = counters.count(&mut call)?;
        least = [least[0].min(counts[0]), least[1].min(counts[1])];
    }
    Ok(least)
}

/// The counters of instructions and branches of this thread in user mode, from `perf_event_open`.
#[cfg(target_os = "linux")]
#[expect(
    unsafe_code,
    reason = "the system calls of perf_event_open take pointers to owned, live memory of the sizes that they read and write"
)]
mod perf {
    use core::{
        ffi::{c_int, c_long, c_ulong, c_void},
        hint::black_box,
    };
    use std::{
        io,
        os::fd::{AsRawFd, FromRawFd, OwnedFd},
    };

    use crate::Failure;

    unsafe extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
        fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
        fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
    }

    #[cfg(target_arch = "x86_64")]
    const PERF_EVENT_OPEN: c_long = 298;
    #[cfg(target_arch = "aarch64")]
    const PERF_EVENT_OPEN: c_long = 241;

    /// The start of `struct perf_event_attr`, with zeros to `PERF_ATTR_SIZE_VER7`.
    #[repr(C)]
    struct Attr {
        kind: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        rest: [u64; 10],
    }

    const HARDWARE: u32 = 0;
    const INSTRUCTIONS: u64 = 1;
    const BRANCHES: u64 = 4;
    const DISABLED: u64 = 1;
    const EXCLUDE_KERNEL: u64 = 1 << 5;
    const EXCLUDE_HV: u64 = 1 << 6;
    const FORMAT_GROUP: u64 = 1 << 3;
    const ENABLE: c_ulong = 0x2400;
    const DISABLE: c_ulong = 0x2401;
    const RESET: c_ulong = 0x2403;
    const GROUP: c_ulong = 1;

    /// A group of two counters: instructions, then branches.
    #[derive(Debug)]
    pub(super) struct Counters {
        leader: OwnedFd,
        _member: OwnedFd,
    }

    impl Counters {
        /// The counters of this thread.
        pub(super) fn open() -> Result<Self, Failure> {
            let leader = open(INSTRUCTIONS, -1, DISABLED)?;
            let member = open(BRANCHES, leader.as_raw_fd(), 0)?;
            Ok(Self {
                leader,
                _member: member,
            })
        }

        /// The instructions and the branches of `call` in user mode, with the constant work of the
        /// two system calls around it.
        pub(super) fn count(&self, call: &mut impl FnMut() -> usize) -> Result<[u64; 2], Failure> {
            let fd = self.leader.as_raw_fd();
            // SAFETY: `fd` is an open perf event, and these requests take an integer argument.
            unsafe {
                ioctl(fd, RESET, GROUP);
                ioctl(fd, ENABLE, GROUP);
            }
            black_box(call());
            // SAFETY: as above.
            unsafe { ioctl(fd, DISABLE, GROUP) };
            let mut values = [0_u64; 3];
            // SAFETY: `values` is owned and has the size that the read may write.
            let read = unsafe { read(fd, values.as_mut_ptr().cast(), size_of_val(&values)) };
            if usize::try_from(read).ok() != Some(size_of_val(&values)) || values[0] != 2 {
                return Err(Failure::Host(format!(
                    "read of the counters: {}",
                    io::Error::last_os_error()
                )));
            }
            Ok([values[1], values[2]])
        }
    }

    fn open(config: u64, group: c_int, flags: u64) -> Result<OwnedFd, Failure> {
        let attr = Attr {
            kind: HARDWARE,
            size: u32::try_from(size_of::<Attr>()).unwrap_or(0),
            config,
            sample_period: 0,
            sample_type: 0,
            read_format: FORMAT_GROUP,
            flags: flags | EXCLUDE_KERNEL | EXCLUDE_HV,
            rest: [0; 10],
        };
        let (pid, cpu, no_flags): (c_int, c_int, c_ulong) = (0, -1, 0);
        // SAFETY: `attr` is a live `perf_event_attr` of the size that it states.
        let fd = unsafe { syscall(PERF_EVENT_OPEN, &raw const attr, pid, cpu, group, no_flags) };
        let fd = c_int::try_from(fd).unwrap_or(-1);
        if fd < 0 {
            return Err(Failure::Host(format!(
                "perf_event_open: {} (set kernel.perf_event_paranoid to 2 or less)",
                io::Error::last_os_error()
            )));
        }
        // SAFETY: the system call gave a new descriptor that nothing else owns.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

/// No counters: only Linux has `perf_event_open`.
#[cfg(not(target_os = "linux"))]
mod perf {
    use crate::Failure;

    #[derive(Debug)]
    pub(super) struct Counters;

    impl Counters {
        pub(super) fn open() -> Result<Self, Failure> {
            Err(Failure::Host(
                "the path count needs the perf events of Linux".to_owned(),
            ))
        }

        pub(super) fn count(&self, _: &mut impl FnMut() -> usize) -> Result<[u64; 2], Failure> {
            Err(Failure::Host(
                "the path count needs the perf events of Linux".to_owned(),
            ))
        }
    }
}
