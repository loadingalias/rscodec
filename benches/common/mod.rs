//! Warm each row, measure heap use on one call, check the result, then time the operation.

use core::{
    alloc::{GlobalAlloc, Layout},
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering::Relaxed},
    time::Duration,
};
use std::alloc::System;

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput, measurement::WallTime};

/// Runs `targets` with the measurement plan of each run. A command-line option of Criterion, for
/// example `--measurement-time`, changes the plan.
pub(crate) fn run(targets: &[fn(&mut Criterion)]) {
    let mut criterion = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1))
        .nresamples(10_000)
        .configure_from_args();
    for target in targets {
        target(&mut criterion);
    }
    criterion.final_summary();
}

/// A Criterion group whose rows check their output and report their heap use.
pub(crate) struct Group<'c> {
    name: String,
    group: BenchmarkGroup<'c, WallTime>,
}

impl<'c> Group<'c> {
    pub(crate) fn new(criterion: &'c mut Criterion, name: &str) -> Self {
        Self {
            name: name.to_owned(),
            group: criterion.benchmark_group(name),
        }
    }

    /// Sets the throughput of the next rows, for example the bytes of one input.
    pub(crate) fn throughput(&mut self, throughput: Throughput) {
        self.group.throughput(throughput);
    }

    /// Registers `operation` of `codec` on `input`. Each call gets `state`, for example an output
    /// buffer that the row reuses. `check` gets the output of the second call and must reject a
    /// wrong output.
    pub(crate) fn row<S, T>(
        &mut self,
        codec: &str,
        input: &str,
        mut state: S,
        mut operation: impl FnMut(&mut S) -> T,
        check: impl FnOnce(T, &mut S),
    ) {
        let id = format!("{}/{codec}/{input}", self.name);
        let mut check = Some(check);
        self.group
            .bench_function(BenchmarkId::new(codec, input), move |bencher| {
                if let Some(check) = check.take() {
                    drop(operation(&mut state));
                    let (output, heap) = heap_use(|| operation(&mut state));
                    check(output, &mut state);
                    report(&id, heap);
                }
                bencher.iter(|| operation(black_box(&mut state)));
            });
    }
}

/// The heap use of one call: the allocations and reallocations, the bytes that they requested, and
/// the peak of the bytes that the call added to the heap.
struct HeapUse {
    allocations: u64,
    bytes: u64,
    peak_bytes: i64,
}

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
static PEAK_BYTES: AtomicI64 = AtomicI64::new(0);

fn heap_use<T>(call: impl FnOnce() -> T) -> (T, HeapUse) {
    for counter in [&ALLOCATIONS, &BYTES] {
        counter.store(0, Relaxed);
    }
    for counter in [&LIVE_BYTES, &PEAK_BYTES] {
        counter.store(0, Relaxed);
    }
    COUNTING.store(true, Relaxed);
    let output = call();
    COUNTING.store(false, Relaxed);
    let heap = HeapUse {
        allocations: ALLOCATIONS.load(Relaxed),
        bytes: BYTES.load(Relaxed),
        peak_bytes: PEAK_BYTES.load(Relaxed),
    };
    (output, heap)
}

#[expect(
    clippy::print_stdout,
    reason = "a benchmark binary reports the heap use of each row on standard output, beside Criterion"
)]
fn report(id: &str, heap: HeapUse) {
    println!(
        "heap {id} allocations={} bytes={} peak_bytes={}",
        heap.allocations, heap.bytes, heap.peak_bytes
    );
}

/// Records a new block of `added` bytes, or a reallocation from `removed` to `added` bytes, while
/// [`heap_use`] counts.
fn record(added: usize, removed: usize) {
    if !COUNTING.load(Relaxed) {
        return;
    }
    let size = |bytes: usize| i64::try_from(bytes).unwrap_or(i64::MAX);
    ALLOCATIONS.fetch_add(1, Relaxed);
    BYTES.fetch_add(u64::try_from(added).unwrap_or(u64::MAX), Relaxed);
    let change = size(added).saturating_sub(size(removed));
    let live = LIVE_BYTES.fetch_add(change, Relaxed).saturating_add(change);
    PEAK_BYTES.fetch_max(live, Relaxed);
}

/// The allocator of each benchmark binary: the system allocator, with counters that only
/// [`heap_use`] turns on. Outside it, each call adds one relaxed load.
struct Counting;

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[expect(
    unsafe_code,
    reason = "each method passes its arguments unchanged to the system allocator, so the caller's contract holds"
)]
// SAFETY: each method forwards to `System`, which meets the `GlobalAlloc` contract.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size(), 0);
        // SAFETY: the caller gives a layout of nonzero size, as `System.alloc` needs.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size(), 0);
        // SAFETY: the caller gives a layout of nonzero size, as `System.alloc_zeroed` needs.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING.load(Relaxed) {
            LIVE_BYTES.fetch_sub(i64::try_from(layout.size()).unwrap_or(i64::MAX), Relaxed);
        }
        // SAFETY: the caller gives a block of this allocator with its layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size, layout.size());
        // SAFETY: the caller gives a block of this allocator with its layout, and a valid new size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
