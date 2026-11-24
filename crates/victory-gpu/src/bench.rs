//! Benchmark zones and the benchmark mode (feature `bench`): where a
//! frame's time goes, measured the same way every run.
//!
//! A zone is a stretch of our code named by [`Z`]. Its time is its *self*
//! time: what zones entered inside it on the same thread took is theirs.
//! So on one thread the zones never overlap, and on the thread that
//! presents, frame time minus the sum of its zones is the time the title's
//! own code ran (or stood) between our calls.
//!
//! `VICTORY_BENCH=<file.json>` turns the mode on: once the world is drawn
//! (`VICTORY_BENCH_MIN_DRAWS` draws a frame, 2000) and has been for
//! `VICTORY_BENCH_SETTLE_SECS` (40) with no pipeline being made over the
//! last `VICTORY_BENCH_STEADY_FRAMES` (30), `VICTORY_BENCH_FRAMES` (300)
//! frames are measured, the file is written and the process ends. A run
//! that has not got there after `VICTORY_BENCH_TIMEOUT` seconds (900)
//! writes what it has, with `"error"`, and ends too.
//!
//! Without the feature every function here is an empty inline.

/// The zones. Add new ones before `Count` and to [`NAMES`].
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
#[allow(dead_code)]
pub(crate) enum Z {
    /// A list call of the title recorded into a host list (draw, dispatch,
    /// state), on the title's recording threads.
    Record,
    /// Inside a list call: the pass over the view heap's buffer views.
    HeapBuffers,
    /// Inside that pass: textures picked by index made and staged.
    HeapTextures,
    /// A list call waiting for a host pipeline being made.
    PipelineWait,
    /// The title's own frame wait (`frame::wait`), sleeping.
    Pace,
    /// A submission, whatever is in none of the zones below.
    Submit,
    /// Waiting for the submission lock.
    SubmitLock,
    /// Settling flights: waiting for the thread that lands them.
    Settle,
    /// The sweep of host objects nothing names any more.
    Sweep,
    /// Descriptor slots that waited for a texture's host copy.
    MakeWaiting,
    /// The write-watch asked once for the submission.
    WatchAsk,
    /// The lists' deferred steps and the batch of refreshes, whatever is
    /// not a copy refresh or a texture refresh.
    Deferred,
    /// Title memory brought up to date in a host copy.
    CopyRefresh,
    /// Deferred steps of the lists, by kind: a copy or fill of title
    /// memory; a refresh put on the batch; work of ours recorded to run
    /// at the submission; a range given to shaders as written.
    StepCopy,
    StepRefresh,
    StepRun,
    StepGpuWrite,
    /// A texture read, hashed and detiled into its staging buffer.
    TexRefresh,
    /// Snapshots of the ranges shaders may write.
    Snapshots,
    /// Copies in the card's memory: changes sent up, the compare recorded.
    CardBefore,
    /// `ExecuteCommandLists` and the fence signal.
    Execute,
    /// Standing, waiting for the card.
    GpuWait,
    /// What shaders wrote found and put back into title memory.
    CopyBack,
    /// Read-backs and bottom-of-pipe writes handed to the title.
    ReadBack,
    /// The present: the copy to the back buffer and the flip.
    Present,
    /// The relay thread waiting for the card.
    RelayWait,
    /// The relay thread landing flights (its copy-back).
    RelayLand,
    Count,
}

#[cfg(feature = "bench")]
const NAMES: [&str; Z::Count as usize] = [
    "record",
    "heap_buffers",
    "heap_textures",
    "pipeline_wait",
    "pace_sleep",
    "submit_other",
    "submit_lock_wait",
    "settle",
    "sweep",
    "make_waiting",
    "watch_ask",
    "deferred_other",
    "copy_refresh",
    "step_copy_fill",
    "step_refresh_add",
    "step_run_host",
    "step_gpu_write",
    "texture_refresh",
    "snapshots",
    "card_before",
    "execute",
    "gpu_wait",
    "copy_back",
    "read_back",
    "present",
    "relay_wait",
    "relay_land",
];

#[cfg(not(feature = "bench"))]
#[inline(always)]
pub(crate) fn zone(_: Z) {}

#[cfg(feature = "bench")]
pub(crate) use on::*;

#[cfg(feature = "bench")]
mod on {
    use super::{NAMES, Z};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Instant;

    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::core::Interface;

    const N: usize = Z::Count as usize;

    /// One thread's zone times (nanoseconds) and entries since the last frame.
    struct Thread {
        id: u32,
        ns: [AtomicU64; N],
        calls: [AtomicU64; N],
    }

    static THREADS: Mutex<Vec<Arc<Thread>>> = Mutex::new(Vec::new());

    fn thread_id() -> u32 {
        // SAFETY: a plain query.
        unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
    }

    thread_local! {
        /// This thread's times, and the nanoseconds the zones inside the
        /// one now open have taken.
        static ME: (Arc<Thread>, Cell<u64>) = {
            let me = Arc::new(Thread {
                id: thread_id(),
                ns: [const { AtomicU64::new(0) }; N],
                calls: [const { AtomicU64::new(0) }; N],
            });
            THREADS.lock().unwrap_or_else(|e| e.into_inner()).push(me.clone());
            (me, Cell::new(0))
        };
    }

    pub(crate) struct Guard {
        z: usize,
        start: Instant,
        outer: u64,
    }

    /// Opens zone `z` until the value is dropped.
    #[inline]
    pub(crate) fn zone(z: Z) -> Guard {
        let outer = ME.try_with(|m| m.1.replace(0)).unwrap_or(0);
        Guard { z: z as usize, start: Instant::now(), outer }
    }

    impl Drop for Guard {
        #[inline]
        fn drop(&mut self) {
            let total = self.start.elapsed().as_nanos() as u64;
            let _ = ME.try_with(|m| {
                let inside = m.1.get();
                m.0.ns[self.z].fetch_add(total.saturating_sub(inside), Relaxed);
                m.0.calls[self.z].fetch_add(1, Relaxed);
                m.1.set(self.outer + total);
            });
        }
    }

    struct Config {
        out: std::path::PathBuf,
        frames: usize,
        min_draws: u64,
        settle: std::time::Duration,
        steady: u32,
        timeout: std::time::Duration,
        /// `VICTORY_BENCH_AT=<s>` and `VICTORY_BENCH_SECS=<s>`: measure the
        /// frames presented in that stretch of the clock, counted from the
        /// first frame of the world, instead: the title plays by the clock,
        /// so two runs then measure the same stretch of the scene.
        at: Option<(std::time::Duration, std::time::Duration)>,
    }

    fn config() -> Option<&'static Config> {
        static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
        CONFIG
            .get_or_init(|| {
                let out = std::path::PathBuf::from(std::env::var_os("VICTORY_BENCH")?);
                let n = |name: &str, default: u64| std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default);
                Some(Config {
                    out,
                    frames: n("VICTORY_BENCH_FRAMES", 300) as usize,
                    min_draws: n("VICTORY_BENCH_MIN_DRAWS", 2000),
                    settle: std::time::Duration::from_secs(n("VICTORY_BENCH_SETTLE_SECS", 40)),
                    steady: n("VICTORY_BENCH_STEADY_FRAMES", 30) as u32,
                    timeout: std::time::Duration::from_secs(n("VICTORY_BENCH_TIMEOUT", 900)),
                    at: std::env::var("VICTORY_BENCH_AT").ok().and_then(|v| v.trim().parse::<u64>().ok()).map(|at| {
                        (std::time::Duration::from_secs(at), std::time::Duration::from_secs(n("VICTORY_BENCH_SECS", 60)))
                    }),
                })
            })
            .as_ref()
    }

    /// Whether the benchmark mode is on.
    pub(crate) fn on() -> bool {
        config().is_some()
    }

    /// Totals since the start, read at every present.
    #[derive(Clone, Copy, Default)]
    pub(crate) struct Counters {
        pub draws: u64,
        pub dropped: u64,
        pub submissions: u64,
        pub pipelines_made: u64,
        pub pipelines_loaded: u64,
        /// Not a total: pipelines being made now.
        pub making: u64,
    }

    struct Frame {
        no: u64,
        wall_ns: u64,
        /// Per zone: the presenting thread's time, every thread's, the
        /// busiest single thread's, and entries.
        main: [u64; N],
        all: [u64; N],
        most: [u64; N],
        calls: [u64; N],
        counters: Counters,
    }

    struct State {
        started: Instant,
        last: Option<Instant>,
        before: Counters,
        no: u64,
        world_since: Option<Instant>,
        steady: u32,
        capturing: bool,
        frames: Vec<Frame>,
    }

    static STATE: Mutex<Option<State>> = Mutex::new(None);
    /// Draws and dispatches dropped since the start (the frame line takes
    /// its own count away every second).
    pub(crate) static DROPPED: AtomicU64 = AtomicU64::new(0);

    pub(crate) enum Next {
        Go,
        /// The measurement is complete, or timed out with this reason.
        Done(Option<&'static str>),
    }

    /// A frame was presented by this thread. Returns the frame's number in
    /// the run and what to do next.
    pub(crate) fn frame_end(now_counters: Counters) -> (u64, Next) {
        let Some(config) = config() else {
            return (0, Next::Go);
        };
        let now = Instant::now();
        let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let state = guard.get_or_insert_with(|| State {
            started: now,
            last: None,
            before: now_counters,
            no: 0,
            world_since: None,
            steady: 0,
            capturing: false,
            frames: Vec::new(),
        });
        state.no += 1;
        let me = thread_id();
        let (mut main, mut all, mut most, mut calls) = ([0u64; N], [0u64; N], [0u64; N], [0u64; N]);
        for thread in THREADS.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            for z in 0..N {
                let ns = thread.ns[z].swap(0, Relaxed);
                all[z] += ns;
                most[z] = most[z].max(ns);
                calls[z] += thread.calls[z].swap(0, Relaxed);
                if thread.id == me {
                    main[z] += ns;
                }
            }
        }
        let before = std::mem::replace(&mut state.before, now_counters);
        let counters = Counters {
            draws: now_counters.draws - before.draws,
            dropped: now_counters.dropped - before.dropped,
            submissions: now_counters.submissions - before.submissions,
            pipelines_made: now_counters.pipelines_made - before.pipelines_made,
            pipelines_loaded: now_counters.pipelines_loaded - before.pipelines_loaded,
            making: now_counters.making,
        };
        let wall_ns = state.last.replace(now).map_or(0, |last| (now - last).as_nanos() as u64);
        let no = state.no;
        if let Some((at, long)) = config.at {
            // From the first frame of the world: how long the run took to
            // get there (loading) varies from run to run by seconds.
            if state.world_since.is_none() && counters.draws >= config.min_draws {
                state.world_since = Some(now);
                tracing::info!(frame = no, "benchmark: the world is drawn");
            }
            let Some(since) = state.world_since else {
                if now - state.started > config.timeout {
                    return (no, Next::Done(Some("timed out before the world was drawn")));
                }
                return (no, Next::Go);
            };
            let clock = now - since;
            if clock >= at + long {
                return (no, Next::Done(None));
            }
            if clock >= at {
                if !state.capturing {
                    state.capturing = true;
                    tracing::info!(frame = no, "benchmark: measuring");
                }
                state.frames.push(Frame { no, wall_ns, main, all, most, calls, counters });
            }
            return (no, Next::Go);
        }
        if state.capturing {
            state.frames.push(Frame { no, wall_ns, main, all, most, calls, counters });
            if state.frames.len() >= config.frames {
                return (no, Next::Done(None));
            }
            return (no, Next::Go);
        }
        // Not measuring yet: the world drawn, long enough, nothing compiling.
        if counters.draws >= config.min_draws {
            let since = *state.world_since.get_or_insert(now);
            state.steady = if counters.making == 0 && counters.pipelines_made == 0 { state.steady + 1 } else { 0 };
            if now - since >= config.settle && state.steady >= config.steady {
                tracing::info!(frame = no, "benchmark: measuring {} frames", config.frames);
                state.capturing = true;
            }
        } else {
            state.world_since = None;
            state.steady = 0;
        }
        if now - state.started > config.timeout {
            return (no, Next::Done(Some("timed out before the world was steady")));
        }
        (no, Next::Go)
    }

    // ---- the card's time, by timestamp queries ---------------------------

    const SETS: usize = 3;
    const CAP: u32 = 8192;

    struct Set {
        heap: ID3D12QueryHeap,
        back: ID3D12Resource,
        allocator: ID3D12CommandAllocator,
        used: u32,
        frame: u64,
        /// The signal behind its resolve, 0 while it has none to read.
        resolved: u64,
    }

    struct Stamps {
        sets: Vec<Set>,
        current: usize,
        first: ID3D12GraphicsCommandList,
        last: ID3D12GraphicsCommandList,
        resolve: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        signals: u64,
        frequency: u64,
        /// (frame, ms the card ran our submissions, ms from the first one's
        /// start to the last one's end, submissions).
        results: Vec<(u64, f64, f64, u32)>,
    }

    // SAFETY: Direct3D 12 objects are free-threaded; the lists are used
    // under the mutex alone.
    unsafe impl Send for Stamps {}

    static STAMPS: Mutex<Option<Stamps>> = Mutex::new(None);

    fn make_stamps(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> windows::core::Result<Stamps> {
        // SAFETY: plain object creation on our device.
        unsafe {
            let mut sets = Vec::new();
            for _ in 0..SETS {
                let mut heap: Option<ID3D12QueryHeap> = None;
                device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: CAP, NodeMask: 0 }, &mut heap)?;
                let mut back: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: u64::from(CAP) * 8,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut back,
                )?;
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                sets.push(Set {
                    heap: heap.ok_or_else(windows::core::Error::empty)?,
                    back: back.ok_or_else(windows::core::Error::empty)?,
                    allocator,
                    used: 0,
                    frame: 0,
                    resolved: 0,
                });
            }
            let list = || -> windows::core::Result<ID3D12GraphicsCommandList> {
                let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &sets[0].allocator, None)?;
                list.Close()?;
                Ok(list)
            };
            Ok(Stamps {
                first: list()?,
                last: list()?,
                resolve: list()?,
                fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
                signals: 0,
                frequency: queue.GetTimestampFrequency()?.max(1),
                sets,
                current: 0,
                results: Vec::new(),
            })
        }
    }

    /// Two closed lists for the lists of one submission to run between: a
    /// timestamp before and one after. Called under the submission lock,
    /// which the caller holds until it has sent them.
    pub(crate) fn stamp_pair(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> Option<(ID3D12CommandList, ID3D12CommandList)> {
        if !on() {
            return None;
        }
        let mut guard = STAMPS.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            match make_stamps(device, queue) {
                Ok(stamps) => *guard = Some(stamps),
                Err(e) => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| tracing::warn!(error = %e, "benchmark: no timestamp queries; the card's time stays 0"));
                    return None;
                }
            }
        }
        let stamps = guard.as_mut()?;
        let set = &mut stamps.sets[stamps.current];
        if set.used + 2 > CAP {
            return None;
        }
        // SAFETY: our lists, closed; each recorded again while the other
        // is closed, on the set's allocator.
        unsafe {
            for (list, index) in [(&stamps.first, set.used), (&stamps.last, set.used + 1)] {
                list.Reset(&set.allocator, None).ok()?;
                list.EndQuery(&set.heap, D3D12_QUERY_TYPE_TIMESTAMP, index);
                list.Close().ok()?;
            }
        }
        set.used += 2;
        Some((stamps.first.cast().ok()?, stamps.last.cast().ok()?))
    }

    fn read_set(stamps: &mut Stamps, index: usize) {
        let frequency = stamps.frequency as f64;
        let set = &mut stamps.sets[index];
        if set.resolved != 0 {
            // SAFETY: our fence; a null event makes the call wait.
            unsafe {
                if stamps.fence.GetCompletedValue() < set.resolved {
                    let _ = stamps.fence.SetEventOnCompletion(set.resolved, HANDLE::default());
                }
                let mut data = std::ptr::null_mut();
                if set.back.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
                    let ticks = std::slice::from_raw_parts(data.cast::<u64>(), set.used as usize);
                    let mut ran = 0u64;
                    for pair in ticks.chunks_exact(2) {
                        ran += pair[1].saturating_sub(pair[0]);
                    }
                    let span = match (ticks.first(), ticks.last()) {
                        (Some(&a), Some(&b)) => b.saturating_sub(a),
                        _ => 0,
                    };
                    set.back.Unmap(0, None);
                    stamps.results.push((set.frame, ran as f64 * 1000.0 / frequency, span as f64 * 1000.0 / frequency, set.used / 2));
                }
                let _ = set.allocator.Reset();
            }
        }
        set.used = 0;
        set.resolved = 0;
    }

    /// At a present, under the submission lock: the frame's timestamps are
    /// resolved and the next set is made ready.
    pub(crate) fn stamps_frame(queue: &ID3D12CommandQueue, frame: u64) {
        let mut guard = STAMPS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(stamps) = guard.as_mut() else {
            return;
        };
        let current = stamps.current;
        // SAFETY: our lists and queue.
        unsafe {
            let set = &stamps.sets[current];
            if set.used > 0
                && stamps.resolve.Reset(&set.allocator, None).is_ok()
            {
                stamps.resolve.ResolveQueryData(&set.heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, set.used, &set.back, 0);
                if stamps.resolve.Close().is_ok()
                    && let Ok(list) = stamps.resolve.cast::<ID3D12CommandList>()
                {
                    queue.ExecuteCommandLists(&[Some(list)]);
                }
            }
            stamps.signals += 1;
            let _ = queue.Signal(&stamps.fence, stamps.signals);
        }
        stamps.sets[current].frame = frame;
        stamps.sets[current].resolved = stamps.signals;
        let next = (current + 1) % SETS;
        read_set(stamps, next);
        stamps.current = next;
    }

    // ---- the result -------------------------------------------------------

    fn stats(values: &mut [f64]) -> (f64, f64, f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        values.sort_by(|a, b| a.total_cmp(b));
        let at = |q: f64| values[((values.len() as f64 * q).ceil() as usize).clamp(1, values.len()) - 1];
        (values.iter().sum::<f64>() / values.len() as f64, at(0.5), at(0.99), values[values.len() - 1])
    }

    fn json_stats(values: &mut [f64]) -> String {
        let (mean, p50, p99, max) = stats(values);
        format!("{{\"mean\": {mean:.3}, \"p50\": {p50:.3}, \"p99\": {p99:.3}, \"max\": {max:.3}}}")
    }

    fn escape(text: &str) -> String {
        text.chars().flat_map(|c| if c == '"' || c == '\\' { vec!['\\', c] } else if c.is_control() { vec![' '] } else { vec![c] }).collect()
    }

    /// Writes the file and ends the process.
    pub(crate) fn finish(error: Option<&'static str>) -> ! {
        let config = config().expect("the mode is on");
        {
            let mut guard = STAMPS.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(stamps) = guard.as_mut() {
                for index in 0..SETS {
                    read_set(stamps, index);
                }
            }
        }
        let state = STATE.lock().unwrap_or_else(|e| e.into_inner()).take();
        let frames = state.as_ref().map_or(&[][..], |s| &s.frames[..]);
        // The first frame's wall time and counters start before the measuring.
        let frames = frames.get(1..).unwrap_or(&[]);
        let card: std::collections::HashMap<u64, (f64, f64, u32)> = STAMPS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|s| s.results.iter().map(|&(frame, ran, span, n)| (frame, (ran, span, n))).collect())
            .unwrap_or_default();
        let ms = |ns: u64| ns as f64 / 1e6;
        let mut out = String::from("{\n");
        if let Some(error) = error {
            out += &format!("  \"error\": \"{}\",\n", escape(error));
        }
        out += &format!("  \"frames\": {},\n", frames.len());
        out += &format!("  \"diagnostics_feature\": {},\n", cfg!(feature = "diagnostics"));
        let switches: Vec<String> = {
            let mut all: Vec<(String, String)> = std::env::vars()
                .filter(|(k, _)| k.starts_with("VICTORY_") && !k.ends_with("_DIR") && !k.contains("CAPTURE") && !k.contains("KEYS") && !k.contains("KEY_EXE") && k != "VICTORY_BENCH")
                .collect();
            all.sort();
            all.iter().map(|(k, v)| format!("\"{}\": \"{}\"", escape(k), escape(v))).collect()
        };
        out += &format!("  \"switches\": {{{}}},\n", switches.join(", "));
        let mut wall: Vec<f64> = frames.iter().map(|f| ms(f.wall_ns)).collect();
        let mean_wall = wall.iter().sum::<f64>() / wall.len().max(1) as f64;
        out += &format!("  \"fps\": {:.3},\n", if mean_wall > 0.0 { 1000.0 / mean_wall } else { 0.0 });
        out += &format!("  \"frame_ms\": {},\n", json_stats(&mut wall));
        let of = |name: &str, f: &dyn Fn(&Frame) -> f64| {
            let mut values: Vec<f64> = frames.iter().map(f).collect();
            format!("  \"{name}\": {},\n", json_stats(&mut values))
        };
        out += &of("draws", &|f| f.counters.draws as f64);
        out += &of("draws_dropped", &|f| f.counters.dropped as f64);
        out += &of("submissions", &|f| f.counters.submissions as f64);
        out += &of("pipelines_being_made", &|f| f.counters.making as f64);
        out += &format!(
            "  \"pipelines_compiled\": {}, \"pipelines_loaded\": {},\n",
            frames.iter().map(|f| f.counters.pipelines_made).sum::<u64>(),
            frames.iter().map(|f| f.counters.pipelines_loaded).sum::<u64>()
        );
        let with_card: Vec<&Frame> = frames.iter().filter(|f| card.contains_key(&f.no)).collect();
        let mut ran: Vec<f64> = with_card.iter().map(|f| card[&f.no].0).collect();
        let mut span: Vec<f64> = with_card.iter().map(|f| card[&f.no].1).collect();
        out += &format!("  \"gpu_frames_measured\": {},\n", with_card.len());
        out += &format!("  \"gpu_ms_running_submissions\": {},\n", json_stats(&mut ran));
        out += &format!("  \"gpu_ms_first_start_to_last_end\": {},\n", json_stats(&mut span));
        // On the presenting thread the zones do not overlap: the rest of
        // the frame is the title's own code between our calls.
        let mut outside: Vec<f64> = frames.iter().map(|f| ms(f.wall_ns.saturating_sub(f.main.iter().sum::<u64>()))).collect();
        out += &format!("  \"present_thread_outside_zones_ms\": {},\n", json_stats(&mut outside));
        out += "  \"zones\": {\n";
        for z in 0..N {
            let column = |pick: &dyn Fn(&Frame) -> u64| -> String {
                let mut values: Vec<f64> = frames.iter().map(|f| ms(pick(f))).collect();
                json_stats(&mut values)
            };
            let calls = frames.iter().map(|f| f.calls[z]).sum::<u64>() as f64 / frames.len().max(1) as f64;
            out += &format!(
                "    \"{}\": {{\"present_thread_ms\": {}, \"all_threads_ms\": {}, \"busiest_thread_ms\": {}, \"calls_per_frame\": {calls:.1}}}{}\n",
                NAMES[z],
                column(&|f| f.main[z]),
                column(&|f| f.all[z]),
                column(&|f| f.most[z]),
                if z + 1 < N { "," } else { "" }
            );
        }
        out += "  }\n}\n";
        if let Err(e) = std::fs::write(&config.out, out) {
            tracing::error!(error = %e, file = %config.out.display(), "benchmark: the result could not be written");
        } else {
            tracing::info!(file = %config.out.display(), frames = frames.len(), "benchmark: written; the process ends");
        }
        // SAFETY: ends our own process, as a host failure does.
        unsafe {
            use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
            TerminateProcess(GetCurrentProcess(), if error.is_some() { 3 } else { 0 });
        }
        unreachable!("the process has ended")
    }
}
