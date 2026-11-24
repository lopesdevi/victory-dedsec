//! `victory pso-repro`: why the host refuses a pipeline, without the title.
//!
//! The shaders named by hash (the 16 digits a run's log gives: "host
//! pipeline refused ... shaders=[compute <hash>]") are found in the pack,
//! translated as a run translates them (the same host probe first), checked
//! by the DXIL validator and made into a pipeline on the host with the
//! Direct3D 12 debug layer on, and the host's own words are printed.
//! `--pair <mesh>+<pixel>` makes one pipeline of the two. Reads only.
//!
//! It begins with what the device reports (`CheckFeatureSupport`): what a
//! translation may use natively and what it has to emulate is decided by
//! that, not by the card's name.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, bail};
use victory_shader::HwStage;
use victory_shader::emit::host::{self, Host};
use victory_shader::emit::{self, HostShader, validator::Validator};
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;

use crate::shader::{MAX_FILE, sources};

fn check<T: Default>(device: &ID3D12Device, feature: D3D12_FEATURE) -> Option<T> {
    let mut data = T::default();
    // SAFETY: the structure is the one the feature asks for.
    unsafe { device.CheckFeatureSupport(feature, (&raw mut data).cast(), size_of::<T>() as u32) }
        .ok()
        .map(|()| data)
}

/// What the device reports, printed, and its widest wave.
fn report(device: &ID3D12Device) -> u32 {
    println!("== device (CheckFeatureSupport) ==");
    let mut highest = None;
    for model in [D3D_SHADER_MODEL_6_9, D3D_SHADER_MODEL_6_8, D3D_SHADER_MODEL_6_7, D3D_SHADER_MODEL_6_6] {
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: model };
        // SAFETY: as `check`; the runtime lowers the model to what it has.
        if unsafe { device.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, (&raw mut sm).cast(), size_of_val(&sm) as u32) }.is_ok() {
            highest = Some(sm.HighestShaderModel.0);
            break;
        }
    }
    match highest {
        Some(m) => println!("shader model            {}.{}", m >> 4, m & 15),
        None => println!("shader model            not reported"),
    }
    let mut wave = 0;
    if let Some(o) = check::<D3D12_FEATURE_DATA_D3D12_OPTIONS1>(device, D3D12_FEATURE_D3D12_OPTIONS1) {
        println!("wave ops                {}", o.WaveOps.as_bool());
        println!("wave lanes              min {} max {}", o.WaveLaneCountMin, o.WaveLaneCountMax);
        println!("int64 shader ops        {}", o.Int64ShaderOps.as_bool());
        wave = o.WaveLaneCountMax;
    }
    if let Some(o) = check::<D3D12_FEATURE_DATA_D3D12_OPTIONS4>(device, D3D12_FEATURE_D3D12_OPTIONS4) {
        println!("native 16-bit ops       {}", o.Native16BitShaderOpsSupported.as_bool());
    }
    if let Some(o) = check::<D3D12_FEATURE_DATA_D3D12_OPTIONS9>(device, D3D12_FEATURE_D3D12_OPTIONS9) {
        println!("int64 atomics           typed resource {}, groupshared {}", o.AtomicInt64OnTypedResourceSupported.as_bool(), o.AtomicInt64OnGroupSharedSupported.as_bool());
    }
    if let Some(o) = check::<D3D12_FEATURE_DATA_D3D12_OPTIONS11>(device, D3D12_FEATURE_D3D12_OPTIONS11) {
        println!("int64 atomics           descriptor heap resource {}", o.AtomicInt64OnDescriptorHeapResourceSupported.as_bool());
    }
    if let Some(o) = check::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>(device, D3D12_FEATURE_D3D12_OPTIONS7) {
        println!("mesh shader tier        {} (10 = tier 1)", o.MeshShaderTier.0);
    }
    if let Some(o) = check::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>(device, D3D12_FEATURE_D3D12_OPTIONS5) {
        println!("raytracing tier         {} (10 = 1.0, 11 = 1.1)", o.RaytracingTier.0);
    }
    if let Some(o) = check::<D3D12_FEATURE_DATA_D3D12_OPTIONS>(device, D3D12_FEATURE_D3D12_OPTIONS) {
        println!("resource binding tier   {}", o.ResourceBindingTier.0);
    }
    wave
}

/// Whether the host caps a view heap at a million views, as a run asks
/// before its first translation (the translation depends on it).
fn view_heap_capped(device: &ID3D12Device) -> bool {
    // SAFETY: plain heap creation; the heap is released at once.
    let made: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1_000_001,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
    };
    made.is_err()
}

fn stage_name(stage: HwStage) -> &'static str {
    match stage {
        HwStage::Cs => "compute",
        HwStage::Ms => "mesh",
        HwStage::Ps => "pixel",
        HwStage::Vs => "vertex",
        _ => "other",
    }
}

/// From a container's `PSV0` part: the wave lane counts the program expects
/// and, where the part holds them (version 2 on), its thread group's size.
fn runtime_info(container: &[u8]) -> Option<(u32, u32, Option<[u32; 3]>)> {
    let word = |at: usize| container.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let parts = word(28)? as usize;
    for i in 0..parts {
        let at = word(32 + 4 * i)? as usize;
        if container.get(at..at + 4)? == b"PSV0" {
            let info = at + 12;
            let size = word(at + 8)? as usize;
            let threads = (size >= 52).then(|| Some([word(info + size - 12)?, word(info + size - 8)?, word(info + size - 4)?])).flatten();
            return Some((word(info + 16)?, word(info + 20)?, threads));
        }
    }
    None
}

/// Translated and validated; the text says what was found on the way.
fn prepare(hash: &str, container: &[u8], validator: &mut Option<Validator>, dump: Option<&std::path::Path>) -> Option<HostShader> {
    let mut shader = match emit::translate(container) {
        Ok(shader) => shader,
        Err(e) => {
            println!("{hash}  does not translate: {e}");
            return None;
        }
    };
    println!("{hash}  {} program, {} bytes of DXIL, {} user data, {} samplers", stage_name(shader.stage), shader.dxil.len(), shader.user_data.len(), shader.samplers.len());
    if let Some((min, max, threads)) = runtime_info(&shader.dxil) {
        println!("{hash}  PSV0: wave lanes {min}..{max}, numthreads {threads:?}");
    }
    let checked = validator.as_mut().map(|v| v.validate(&mut shader.dxil));
    // `--dump <dir>`: the translation as the host gets it (signed by the
    // validator where it passed), for `dxc -dumpbin` and the like.
    if let Some(dir) = dump {
        let file = dir.join(format!("{hash}.dxil"));
        match std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&file, &shader.dxil)) {
            Ok(()) => println!("{hash}  written to {}", file.display()),
            Err(e) => println!("{hash}  not written: {e}"),
        }
    }
    match checked {
        Some(Ok(())) => println!("{hash}  validator: accepted"),
        Some(Err(e)) => println!("{hash}  validator: REFUSED: {}", e.lines().take(12).collect::<Vec<_>>().join(" | ")),
        None => println!("{hash}  validator: not loaded (set VICTORY_DXIL to dxil.dll)"),
    }
    Some(shader)
}

pub fn run(paths: Vec<PathBuf>, hashes: Vec<String>, pairs: Vec<String>, dump: Option<PathBuf>) -> Result<()> {
    let dump = dump.as_deref();
    // The debug layer's messages are what this command is for.
    // SAFETY: set before any other thread of ours is started.
    unsafe { std::env::set_var("VICTORY_D3D12_DEBUG", "1") };
    let host = Host::new(false).map_err(anyhow::Error::msg)?;
    {
        use windows::core::Interface;
        let layer = host.device.cast::<ID3D12InfoQueue>().is_ok();
        println!("debug layer             {}", if layer { "on" } else { "NOT AVAILABLE (install the Windows \"Graphics Tools\" feature): the host's reasons will be missing" });
    }
    let wave = report(&host.device);
    if wave != 0 {
        emit::set_host_max_wave(wave);
    }
    let capped = view_heap_capped(&host.device);
    println!("view heap capped at 1e6 {capped}");
    if capped {
        emit::set_view_heap_limit(999_999);
    }
    let pairs: Vec<(String, String)> = pairs.iter().filter_map(|p| p.split_once('+')).map(|(a, b)| (a.to_lowercase(), b.to_lowercase())).collect();
    let mut wanted: Vec<String> = hashes.iter().map(|h| h.to_lowercase()).collect();
    for (a, b) in &pairs {
        wanted.push(a.clone());
        wanted.push(b.clone());
    }
    if wanted.is_empty() {
        return Ok(());
    }
    let dll = Validator::default_path();
    let mut validator = Validator::load(dll.as_deref()).ok();
    let mut found: HashMap<String, Vec<u8>> = HashMap::new();
    for root in &crate::default_paths(paths)? {
        for entry in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else { continue };
            for source in sources(&data) {
                let Some(bytes) = source.bytes() else { continue };
                let Some(len) = bytes.get(24..28).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize).filter(|&n| n >= 32 && n <= bytes.len()) else {
                    continue;
                };
                let hash = blake3::hash(&bytes[..len]).to_hex()[..16].to_owned();
                if let Some(w) = wanted.iter().find(|w| hash.starts_with(w.as_str())) {
                    found.entry(w.clone()).or_insert_with(|| bytes[..len].to_vec());
                }
            }
        }
    }
    println!("\n== pipelines ==");
    let mut refused = 0u32;
    // Whether the pipeline was made, said in the host's own words if not.
    fn outcome(what: String, made: Result<(), String>) -> bool {
        match &made {
            Ok(()) => println!("{what}  pipeline: ACCEPTED"),
            Err(e) => println!("{what}  pipeline: REFUSED: {e}"),
        }
        made.is_ok()
    }
    for hash in hashes.iter().map(|h| h.to_lowercase()) {
        let Some(container) = found.get(&hash) else {
            println!("{hash}  not found in the paths given");
            refused += 1;
            continue;
        };
        let Some(shader) = prepare(&hash, container, &mut validator, dump) else {
            refused += 1;
            continue;
        };
        let made = match shader.stage {
            HwStage::Cs => host.compute_pipeline(&shader).map(drop),
            HwStage::Ms => host.mesh_pipeline(&shader, &host::stub_ps()).map(drop),
            HwStage::Vs => host.pipeline(&shader, &host::stub_ps()).map(drop),
            _ => host::stub_vs(&shader).and_then(|vs| host.pipeline(&vs, &shader).map(drop)),
        };
        refused += u32::from(!outcome(hash, made));
    }
    for (first, second) in &pairs {
        let (Some(a), Some(b)) = (found.get(first), found.get(second)) else {
            println!("{first}+{second}  not found in the paths given");
            refused += 1;
            continue;
        };
        let (Some(a), Some(b)) = (prepare(first, a, &mut validator, dump), prepare(second, b, &mut validator, dump)) else {
            refused += 1;
            continue;
        };
        let made = match a.stage {
            HwStage::Ms => host.mesh_pipeline(&a, &b).map(drop),
            _ => host.pipeline(&a, &b).map(drop),
        };
        refused += u32::from(!outcome(format!("{first}+{second}"), made));
    }
    if refused > 0 {
        bail!("{refused} refused");
    }
    Ok(())
}
