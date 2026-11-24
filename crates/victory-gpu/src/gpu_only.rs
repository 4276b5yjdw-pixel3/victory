//! Title memory the title reserves but never commits to our process (GPU
//! only: the CPU never touches it, so there is nothing to copy). Every view
//! of one aligned range shares a persistent zeroed buffer, so a write by one
//! dispatch is read back by the next, instead of falling into the shared
//! sink (which drops writes and reads zeros).
//!
//! shortcut: a range the CPU touches later is not copied from or to the
//! title's memory here; the title's own copies of that range are not seen.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

/// The range one buffer covers (64 KiB, the title's page size for these).
pub(crate) const RANGE: u64 = 64 << 10;

/// Range start -> its buffer.
static RANGES: Mutex<BTreeMap<u64, Arc<ID3D12Resource>>> = Mutex::new(BTreeMap::new());

/// `(range start, offset of address in its range)`. Pure, for tests.
pub(crate) fn split(address: u64) -> (u64, u64) {
    let start = address & !(RANGE - 1);
    (start, address - start)
}

/// The GPU-only buffer of `address`'s range, made zeroed on first use, with
/// the offset of `address` in it and the bytes to the range's end.
pub(crate) fn view(address: u64) -> Option<(ID3D12Resource, u64, u64)> {
    let (start, offset) = split(address);
    let mut ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(buffer) = ranges.get(&start) {
        return Some(((**buffer).clone(), offset, RANGE - offset));
    }
    let (buffer, data) = crate::host::host()?.shared_buffer(RANGE)?;
    // SAFETY: the buffer's CPU mapping, RANGE bytes, persistent; nothing on
    // the GPU uses it yet.
    unsafe { data.write_bytes(0, RANGE as usize) };
    tracing::debug!(start = format_args!("{start:#x}"), "GPU-only title range made");
    ranges.insert(start, Arc::new(buffer.clone()));
    Some((buffer, offset, RANGE - offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_address_splits_into_its_range_and_offset() {
        assert_eq!(split(0x1_0000), (0x1_0000, 0));
        assert_eq!(split(0x1_ffff), (0x1_0000, 0xffff));
        assert_eq!(split(0x7001_df100), (0x7001_d0000, 0xf100));
        assert_eq!(split(0x7001_d0000 + RANGE), (0x7001_e0000, 0));
    }
}
