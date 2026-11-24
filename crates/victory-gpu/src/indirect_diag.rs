//! Diagnostics of indirect execution (diagnostics builds only).
//!
//! With `VICTORY_INDIRECT_LOG` set, each ExecuteIndirect logs its kind,
//! stride, record count, title addresses and the first records of its
//! argument buffer as the host copy holds them on the CPU. On the frames
//! listed in `VICTORY_CAPTURE_FRAMES`, the first records are also copied
//! into readback buffers right after the execution and read at the frame's
//! present, so the log shows what the GPU had there (after the clamp, which
//! runs before the execution). Shader writes to shadowed ranges are logged
//! on those frames too.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::host::{HostList, host, transition};
use crate::shadow::Shadow;

/// Commands logged from the CPU copy over the whole run.
const LOGGED: u64 = 400;
/// Shader writes logged over the whole run.
const WRITES_LOGGED: u64 = 200;
/// Commands read back per listed frame.
const PER_FRAME: usize = 20;
/// Records read back (and logged) per command.
const RECORDS: u32 = 4;

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_LOG").is_some())
}

/// A readback buffer waiting for the frame's present.
struct Pending {
    label: String,
    buffer: ID3D12Resource,
    bytes: usize,
}

// SAFETY: D3D12 objects are free-threaded; read only after the GPU finished.
unsafe impl Send for Pending {}

static PENDING: Mutex<Vec<Pending>> = Mutex::new(Vec::new());
/// The listed frame being recorded and the commands read back in it.
static FRAME: Mutex<(u64, usize)> = Mutex::new((0, 0));
static LOGS: AtomicU64 = AtomicU64::new(0);
static WRITES: AtomicU64 = AtomicU64::new(0);

/// The frame the current commands belong to (the one presenting next).
fn frame() -> u64 {
    crate::queue::presented() + 1
}

/// Records, after an ExecuteIndirect of `args` at offset `at` (the
/// execution's `(kind, stride, max)`), what the arguments are.
pub(crate) fn note_exec(
    list: &HostList,
    args: &Shadow,
    at: u64,
    (kind, stride, max): (u32, u32, u32),
    arguments: u64,
    counter: Option<(&Shadow, u64)>,
) {
    if !on() {
        return;
    }
    note_read("indirect arguments", arguments, u64::from(stride) * u64::from(max));
    let records = max.min(RECORDS);
    let bytes = (records as usize) * (stride as usize);
    let frame = frame();
    let listed = crate::capture::listed(frame);
    if LOGS.fetch_add(1, Relaxed) < LOGGED || listed {
        let cpu: Vec<u32> = (0..bytes / 4)
            .map(|i| args.read_u32(at as usize + 4 * i))
            .collect();
        let count = counter.map(|(c, at)| c.read_u32(at as usize));
        tracing::warn!(
            frame,
            kind,
            stride,
            max,
            arguments = format_args!("{arguments:#x}"),
            count = ?count,
            cpu = ?cpu,
            "indirect exec (host copy)"
        );
    }
    if !listed || bytes == 0 {
        return;
    }
    {
        let mut f = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if f.0 != frame {
            *f = (frame, 0);
        }
        if f.1 >= PER_FRAME {
            return;
        }
        f.1 += 1;
    }
    let Some(buffer) = readback(bytes) else {
        return;
    };
    // SAFETY: an open list; the copy buffer is in unordered access after
    // the execution (`done`), and the readback is a live buffer of `bytes`.
    unsafe {
        let l = &list.list;
        l.ResourceBarrier(&[transition(
            &args.buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        l.CopyBufferRegion(&buffer, 0, &args.buffer, at, bytes as u64);
        l.ResourceBarrier(&[transition(
            &args.buffer,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
    }
    let label = format!("kind {kind} stride {stride} max {max} at {arguments:#x}");
    PENDING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Pending { label, buffer, bytes });
}

/// A readback buffer of `bytes` bytes.
fn readback(bytes: usize) -> Option<ID3D12Resource> {
    let device = &host()?.device;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a readback buffer.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
    }
    buffer
}

/// At a present, after the submissions ran: logs the readbacks of the
/// frame's commands (what the GPU held there).
pub(crate) fn flush() {
    if !on() {
        return;
    }
    let pending = std::mem::take(&mut *PENDING.lock().unwrap_or_else(|e| e.into_inner()));
    for p in pending {
        let mut data = std::ptr::null_mut();
        // SAFETY: the copy ran in a submission that was waited for; the
        // mapping spans the bytes copied.
        let words: Vec<u32> = unsafe {
            if p.buffer.Map(0, None, Some(&mut data)).is_err() {
                continue;
            }
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), p.bytes);
            let words = bytes
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            p.buffer.Unmap(0, None);
            words
        };
        tracing::warn!(label = %p.label, gpu = ?words, "indirect exec (GPU copy after clamp)");
    }
}

/// A shader write to `[address, address + len)` of a shadowed range, on the
/// listed frames: the root UAV ranges of the title's compute passes.
pub(crate) fn note_write(address: u64, len: u64) {
    if !on() || !crate::capture::listed(frame()) {
        return;
    }
    {
        let mut ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner());
        ranges.push((address, len, frame()));
    }
    if WRITES.fetch_add(1, Relaxed) < WRITES_LOGGED {
        tracing::warn!(frame = frame(), address = format_args!("{address:#x}"), len, "shader write range");
    }
}

/// Shader-written ranges of the listed frames (address, length, frame).
static RANGES: Mutex<Vec<(u64, u64, u64)>> = Mutex::new(Vec::new());
static HITS: AtomicU64 = AtomicU64::new(0);

/// A read of `[address, address + len)` by `what` (index buffer, indirect
/// arguments or count): on a listed frame, logged when it overlaps a range
/// the shaders wrote on that frame (the consumer of a compute pass's output).
pub(crate) fn note_read(what: &str, address: u64, len: u64) {
    if !on() {
        return;
    }
    let now = frame();
    if !crate::capture::listed(now) {
        return;
    }
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    for &(start, wlen, _) in ranges.iter().filter(|r| r.2 == now) {
        if address < start + wlen && start < address + len && HITS.fetch_add(1, Relaxed) < WRITES_LOGGED {
            tracing::warn!(
                frame = now,
                what,
                read = format_args!("{address:#x}+{len:#x}"),
                written = format_args!("{start:#x}+{wlen:#x}"),
                "read of a range shaders wrote"
            );
        }
    }
}
