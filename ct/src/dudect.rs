//! DudeCT (Reparaz, Balasch, and Verbauwhede, 2017) for each secret function: the time of one
//! fixed input against random inputs of the same length and outcome, in a random order. Welch's
//! t-test compares the two classes on all samples, on 100 crops of the slow samples, and in the
//! second order. The decision is the one of rscrypto (`ct.toml`): |t| at most 8, and a second
//! measurement with 4 times the samples and a new seed when |t| reaches 0.75 of that threshold.
//!
//! Each fixed/random and A/A sample generates and copies its input before the batch is timed.
//! A fixed random input uses the same seed each time. A/A uses fresh seeds for both classes.
//! The zero diagnostic uses zeros for its fixed input.
//! The case prefix and length select the inputs. The call count sets the minimum for calibration.
//! A minimum above one adds one untimed call. With explicit `warm` or `unwarmed`, the count is exact.
//! The `buffers` control copies over the same input and output ranges as the selected operation.

use core::hint::black_box;

use rscodec::Dit;

use crate::{
    Failure,
    cases::{Case, Class, Direction, Fill, Outcome},
    random::Random,
};

/// The largest |t| that passes.
const THRESHOLD: f64 = 8.0;

/// The part of the threshold at which a pass needs a second measurement.
const REVIEW: f64 = 0.75;

/// The samples of the second measurement, as a multiple of the first.
const CONFIRM: usize = 4;

/// The least median of the timer ticks of one sample. At 100 ticks, one tick is at most 1% of it.
const TICKS_MIN: u64 = 100;

/// The samples of one batch, whose inputs are in one buffer in the order of the samples.
const BATCH: usize = 1024;

/// The crops of the slow samples.
const PERCENTILES: usize = 100;

/// The least samples of each class in a test that counts.
const ENOUGH: f64 = 10_000.0;

/// The lengths of the data in bytes.
const LENGTHS: [usize; 4] = [16, 32, 64, 4096];

/// The length of the data of the refusals.
const REFUSAL_BYTES: usize = 64;

/// The samples of each measurement, unless the command line gives a number.
const SAMPLES: usize = 10_000_000;

/// The seed of the first measurement.
const SEED: u64 = 0xd0de;

#[derive(Clone, Copy)]
enum Calls {
    Calibrate(usize),
    Exact { count: usize, warmups: usize },
}

/// Welch's t-test of two classes, with Welford's sums.
#[derive(Clone, Copy, Debug, Default)]
struct Test {
    n: [f64; 2],
    mean: [f64; 2],
    m2: [f64; 2],
}

impl Test {
    fn push(&mut self, class: usize, x: f64) {
        let class = class & 1;
        self.n[class] += 1.0;
        let delta = x - self.mean[class];
        self.mean[class] += delta / self.n[class];
        self.m2[class] += delta * (x - self.mean[class]);
    }

    /// The standard error of the difference of the means.
    fn error(&self) -> f64 {
        let variance = |class: usize| self.m2[class] / (self.n[class] - 1.0);
        (variance(0) / self.n[0] + variance(1) / self.n[1]).sqrt()
    }

    /// |t|, or zero if a class has too few samples.
    fn t(&self) -> f64 {
        if self.n[0] < ENOUGH || self.n[1] < ENOUGH {
            return 0.0;
        }
        ((self.mean[0] - self.mean[1]) / self.error()).abs()
    }
}

/// The result of one measurement.
#[derive(Debug)]
struct Measure {
    t: f64,
    /// The test that gave the largest absolute t value.
    test: &'static str,
    samples: usize,
    calls: usize,
    median: u64,
    /// The mean ticks per call in each class, before any crop.
    means: [f64; 2],
    /// The difference of the mean ticks of one call that gives the threshold.
    detectable: f64,
}

/// Runs DudeCT on each secret function.
pub(crate) fn run(args: &[String]) -> Result<(), Failure> {
    let number = |index: usize, name: &str| {
        args.get(index)
            .map(|arg| {
                arg.parse::<usize>()
                    .map_err(|_| Failure::Host(format!("invalid {name}: {arg}")))
            })
            .transpose()
    };
    let samples = number(0, "sample count")?.unwrap_or(SAMPLES);
    let confirmation = samples
        .checked_mul(CONFIRM)
        .filter(|n| n.checked_add(BATCH - 1).is_some())
        .ok_or_else(|| Failure::Host(format!("sample count too large: {samples}")))?;
    let fixed = match args.get(1).map(String::as_str) {
        Some("zero") => Some(Fill::Zero),
        Some("aa") => None,
        Some("random") | None => Some(Fill::Random),
        Some(mode) => return Err(Failure::Host(format!("unknown input class: {mode}"))),
    };
    let prefix = args.get(2).map_or("", String::as_str);
    let only = number(3, "length")?;
    let count = number(4, "call count")?.unwrap_or(1);
    if !(1..=1 << 20).contains(&count) {
        return Err(Failure::Host(format!("call count outside 1..=1048576: {count}")));
    }
    let calls = match args.get(5).map(String::as_str) {
        None => Calls::Calibrate(count),
        Some("warm") => Calls::Exact { count, warmups: 1 },
        Some("unwarmed") => Calls::Exact { count, warmups: 0 },
        Some(value) => return Err(Failure::Host(format!("unknown warm-up: {value}"))),
    };
    let buffers = match args.get(6).map(String::as_str) {
        None | Some("codec") => false,
        Some("buffers") => true,
        Some(value) => return Err(Failure::Host(format!("unknown operation: {value}"))),
    };
    if args.len() > 7 {
        return Err(Failure::Host(crate::USAGE.to_owned()));
    }
    let cases: Vec<Case> = Case::all()
        .into_iter()
        .filter(|case| case.class == Class::Secret && case.name.starts_with(prefix))
        .collect();
    let mut failures = 0_usize;
    let mut measured = 0_usize;
    rscodec::with_dit(|dit| -> Result<(), Failure> {
        for mut case in cases {
            if buffers {
                case.use_buffer_control();
            }
            let mut shapes: Vec<(usize, Outcome)> = LENGTHS.iter().map(|&bytes| (bytes, Outcome::Valid)).collect();
            if case.direction == Direction::Decode {
                shapes.push((REFUSAL_BYTES, Outcome::Symbol));
            }
            for (bytes, outcome) in shapes
                .into_iter()
                .filter(|&(bytes, _)| only.is_none_or(|only| only == bytes))
            {
                let first = measure(&case, dit, (bytes, outcome, fixed), (samples, calls), SEED)?;
                let mut line = report(&case.name, bytes, outcome, &first);
                let pass = if first.t > THRESHOLD {
                    false
                } else if first.t >= REVIEW * THRESHOLD {
                    // A new seed gives new inputs and a new order of the classes.
                    let second = measure(
                        &case,
                        dit,
                        (bytes, outcome, fixed),
                        (confirmation, calls),
                        timer::ticks(),
                    )?;
                    line.push_str(&format!("\n  second: {}", report(&case.name, bytes, outcome, &second)));
                    second.t <= THRESHOLD
                } else {
                    true
                };
                println!("{line} {}", if pass { "pass" } else { "FAIL" });
                failures += usize::from(!pass);
                measured += 1;
            }
        }
        Ok(())
    })
    .map_err(|refusal| Failure::Host(refusal.to_string()))??;
    if measured == 0 {
        return Err(Failure::Host(format!(
            "no measurement matches case {prefix:?} and length {only:?}"
        )));
    }
    let mode = fixed.map_or_else(
        || "no fixed input (A/A)".to_owned(),
        |fill| format!("a fixed input of {fill:?} data"),
    );
    println!("dudect: {measured} measurements with {mode}, {failures} over |t| {THRESHOLD}");
    if failures == 0 {
        Ok(())
    } else {
        Err(Failure::Check(format!(
            "{failures} measurements are over |t| {THRESHOLD}"
        )))
    }
}

fn report(name: &str, bytes: usize, outcome: Outcome, measure: &Measure) -> String {
    let [mean0, mean1] = measure.means;
    format!(
        "dudect {name} bytes={bytes} {outcome:?} samples={} calls={} median={} ticks |t|={:.2} test={} \
         mean0={mean0:.3} mean1={mean1:.3} delta(0-1)={:+.3} detectable={:.3} ticks per call",
        measure.samples,
        measure.calls,
        measure.median,
        measure.t,
        measure.test,
        mean0 - mean1,
        measure.detectable
    )
}

/// Generates and copies each input before the batch is timed.
fn fill_batch<'a>(
    random: &mut Random,
    fixed: Option<(Fill, u64)>,
    slots: impl Iterator<Item = &'a mut [u8]>,
    classes: &mut [usize],
    mut input: impl FnMut(Fill, u64) -> Result<Vec<u8>, Failure>,
) -> Result<(), Failure> {
    for (slot, class) in slots.zip(classes) {
        *class = usize::from(random.next() & 1 == 1);
        // Draw both words for every slot. A/A keeps the class order and the class-one inputs.
        let fresh = (Fill::Random, random.next());
        let (fill, seed) = if *class == 0 { fixed.unwrap_or(fresh) } else { fresh };
        slot.copy_from_slice(&input(fill, seed)?);
    }
    Ok(())
}

/// One DudeCT measurement of `case`.
fn measure(
    case: &Case,
    dit: &Dit,
    (bytes, outcome, fixed): (usize, Outcome, Option<Fill>),
    (samples, sampling): (usize, Calls),
    seed: u64,
) -> Result<Measure, Failure> {
    // The schedule and the fixed input start from different generator states.
    let mut random = Random::new(Random::new(seed).next());
    let fixed = fixed.map(|fill| (fill, seed));
    // A refusal keeps its bad character at one offset: its time may depend on the offset.
    let at = Some(bytes);
    let input = |fill, seed| {
        case.input(bytes, outcome, fill, at, &mut Random::new(seed))?
            .ok_or_else(|| Failure::Host(format!("{} has no input of {bytes} bytes with {outcome:?}", case.name)))
    };
    let len = input(Fill::Random, seed)?.len();
    let mut out = vec![0; case.out_len(bytes)?];
    let mut inputs = vec![0; BATCH * len];
    let mut classes = [0_usize; BATCH];
    let mut ticks = [0_u64; BATCH];
    let (mut calls, warmups) = match sampling {
        Calls::Calibrate(min) => (min, usize::from(min > 1)),
        Calls::Exact { count, warmups } => (count, warmups),
    };
    println!("schedule {} seed={seed} warmups={warmups}", case.name);
    let mut time = |calls: usize, inputs: &[u8], ticks: &mut [u64; BATCH]| {
        for (input, slot) in inputs.chunks_exact(len).zip(ticks.iter_mut()) {
            for _ in 0..warmups {
                black_box(case.call(dit, black_box(input), &mut out));
            }
            let start = timer::ticks();
            for _ in 0..calls {
                black_box(case.call(dit, black_box(input), &mut out));
            }
            *slot = timer::ticks().wrapping_sub(start);
        }
    };
    // The calls of one sample: the least power of two whose median sample spans twice the floor.
    fill_batch(&mut random, fixed, inputs.chunks_exact_mut(len), &mut classes, input)?;
    let median = loop {
        time(calls, &inputs, &mut ticks);
        let mut sorted = ticks;
        sorted.sort_unstable();
        let median = sorted[BATCH / 2];
        if matches!(sampling, Calls::Exact { .. }) || median >= 2 * TICKS_MIN || calls >= 1 << 20 {
            break median;
        }
        calls *= 2;
    };
    if median < TICKS_MIN {
        return Err(Failure::Host(format!(
            "{}: a sample of {calls} calls spans {median} ticks",
            case.name
        )));
    }
    // The crops come from one batch of both classes, as in DudeCT.
    time(calls, &inputs, &mut ticks);
    let mut sorted = ticks;
    sorted.sort_unstable();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "each part is in [0, 1), so its product with BATCH is an index in [0, BATCH)"
    )]
    let crops: Vec<u64> = (0..PERCENTILES)
        .map(|index| {
            let part = 1.0 - 0.5_f64.powf(10.0 * (index + 1) as f64 / PERCENTILES as f64);
            sorted[((part * BATCH as f64) as usize).min(BATCH - 1)]
        })
        .collect();
    let mut all = Test::default();
    let mut second = Test::default();
    let mut cropped = vec![Test::default(); PERCENTILES];
    let mut taken = 0_usize;
    while taken < samples {
        fill_batch(&mut random, fixed, inputs.chunks_exact_mut(len), &mut classes, input)?;
        time(calls, &inputs, &mut ticks);
        for (&class, &sample) in classes.iter().zip(&ticks) {
            let x = sample as f64;
            if all.n[class] >= ENOUGH {
                let centered = x - all.mean[class];
                second.push(class, centered * centered);
            }
            all.push(class, x);
            for (test, &crop) in cropped.iter_mut().zip(&crops) {
                if sample < crop {
                    test.push(class, x);
                }
            }
        }
        taken += BATCH;
    }
    summarize(&all, &cropped, &second, taken, calls, median)
}

fn summarize(
    all: &Test,
    cropped: &[Test],
    second: &Test,
    samples: usize,
    calls: usize,
    median: u64,
) -> Result<Measure, Failure> {
    if second.n.iter().any(|&n| n < ENOUGH) {
        return Err(Failure::Host(format!(
            "too few samples for the second-order test: {:?} per class",
            second.n
        )));
    }
    let mut t = all.t();
    if t.is_nan() {
        return Err(Failure::Host(
            "the timer gave the same value for every sample".to_owned(),
        ));
    }
    let mut test = "uncropped";
    for (name, value) in [
        ("cropped", cropped.iter().map(Test::t).fold(0.0, f64::max)),
        ("second-order", second.t()),
    ] {
        if value > t {
            (test, t) = (name, value);
        }
    }
    Ok(Measure {
        t,
        test,
        samples,
        calls,
        median,
        means: all.mean.map(|mean| mean / calls as f64),
        detectable: THRESHOLD * all.error() / calls as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_random_and_aa_prepare_every_slot_in_the_same_order() {
        let batch = |fixed| {
            let mut classes = [0; 32];
            let mut slots = [[0u8; 8]; 32];
            let mut generated = 0;
            fill_batch(
                &mut Random::new(17),
                fixed,
                slots.iter_mut().map(<[u8; 8]>::as_mut_slice),
                &mut classes,
                |_, seed| {
                    generated += 1;
                    Ok(seed.to_le_bytes().to_vec())
                },
            )
            .expect("the generator always returns eight bytes");
            assert_eq!(generated, classes.len());
            (classes, slots)
        };
        let (classes, fixed) = batch(Some((Fill::Random, SEED)));
        let (aa_classes, aa) = batch(None);
        assert_eq!(classes, aa_classes);
        assert!(classes.contains(&0) && classes.contains(&1));
        for ((class, fixed), aa) in classes.into_iter().zip(fixed).zip(aa) {
            if class == 0 {
                assert_eq!(fixed, SEED.to_le_bytes());
                assert_ne!(fixed, aa);
            } else {
                assert_eq!(fixed, aa);
            }
        }
    }

    #[test]
    fn the_timing_report_gives_the_observed_difference_when_the_second_order_test_wins() {
        // Four calls per sample. The sample means differ by four ticks. The variances are 4 and 16.
        let all = Test {
            n: [20_000.0; 2],
            mean: [200.0, 204.0],
            m2: [79_996.0, 319_984.0],
        };
        let second = Test {
            n: [10_000.0; 2],
            mean: [4.0, 16.0],
            m2: [0.0; 2],
        };
        let measurement = summarize(&all, &[], &second, 40_000, 4, 202).expect("both classes have enough samples");
        let line = report("hex-encode-secret", 4096, Outcome::Valid, &measurement);
        assert!(
            line.ends_with(
                "|t|=inf test=second-order mean0=50.000 mean1=51.000 delta(0-1)=-1.000 detectable=0.063 ticks per call"
            ),
            "{line}"
        );
    }
}

/// The timer of the CPU: the time-stamp counter of x86_64, and the virtual counter of aarch64.
#[expect(
    unsafe_code,
    reason = "each asm reads a counter of the CPU that user code may read, and writes only its outputs"
)]
mod timer {
    use std::arch::asm;

    /// The counter, with a barrier before it, so that the instructions before it complete first.
    #[inline(always)]
    pub(super) fn ticks() -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            let (low, high): (u32, u32);
            // SAFETY: `rdtsc` is defined on each x86_64 CPU and reads no memory.
            unsafe { asm!("lfence", "rdtsc", "lfence", out("eax") low, out("edx") high, options(nostack)) };
            (u64::from(high) << 32) | u64::from(low)
        }
        #[cfg(target_arch = "aarch64")]
        {
            let ticks: u64;
            // SAFETY: Linux and macOS let user code read `cntvct_el0`.
            unsafe { asm!("isb", "mrs {ticks}, cntvct_el0", ticks = out(reg) ticks, options(nostack)) };
            ticks
        }
    }
}
