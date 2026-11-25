//! Shared frame slot between the capture and sender threads.
//!
//! [`LatestFrameSlot`] holds a single latest-frame buffer protected by a mutex
//! and a condvar. The capture thread overwrites the slot with each new DDA grab;
//! the sender thread wakes up, swaps the BGRA data out, and transmits it over
//! UDP. Only the most recent frame is ever retained -- old frames cannot queue
//! behind inference.

use std::sync::{Condvar, Mutex};

/// Raw BGRA frame data plus metadata written by the capture loop.
pub struct FrameData {
    /// BGRA pixel data (4 bytes per pixel, row-major).
    pub bgra: Vec<u8>,
    /// Monotonically increasing sequence number (bumped on each post).
    pub sequence: u64,
    /// Wall-clock capture timestamp in microseconds since the UNIX epoch,
    /// stamped the instant `AcquireNextFrame` returns a new desktop frame.
    pub capture_unix_micros: u64,
    /// Frame width in pixels.
    pub width: u16,
    /// Frame height in pixels.
    pub height: u16,
    /// `true` when a new frame is available that the sender has not consumed.
    pub has_frame: bool,
    /// `true` when the capture loop has exited; the sender should drain and stop.
    pub stopped: bool,
}

impl Default for FrameData {
    fn default() -> Self {
        Self {
            bgra: Vec::new(),
            sequence: 0,
            capture_unix_micros: 0,
            width: 0,
            height: 0,
            has_frame: false,
            stopped: false,
        }
    }
}

/// A single-slot latest-frame buffer shared between the capture thread (writer)
/// and the sender thread (reader).
///
/// # Usage from the sender side
///
/// ```ignore
/// let mut local_bgra: Vec<u8> = Vec::new();
/// loop {
///     let mut data = slot.data.lock().unwrap();
///     // ... wait on slot.cv ...
///     std::mem::swap(&mut local_bgra, &mut data.bgra);
///     data.has_frame = false;
///     drop(data);
///     // send local_bgra over UDP
/// }
/// ```
pub struct LatestFrameSlot {
    pub data: Mutex<FrameData>,
    pub cv: Condvar,
}

impl LatestFrameSlot {
    pub fn new() -> Self {
        Self {
            data: Mutex::new(FrameData::default()),
            cv: Condvar::new(),
        }
    }

    /// Post a new captured frame, replacing any unconsumed previous frame.
    ///
    /// Takes ownership of the `bgra` vec; the caller should pass the buffer in
    /// and re-allocate or reuse a swapped-out buffer from the sender thread.
    pub fn post_frame(
        &self,
        bgra: Vec<u8>,
        width: u16,
        height: u16,
        capture_unix_micros: u64,
    ) {
        let mut data = self.data.lock().unwrap();
        data.bgra = bgra;
        data.width = width;
        data.height = height;
        data.capture_unix_micros = capture_unix_micros;
        data.has_frame = true;
        data.sequence += 1;
        drop(data);
        self.cv.notify_one();
    }

    /// Signal that the capture loop has exited. Wakes all waiters so they can
    /// observe `stopped == true` and shut down.
    pub fn signal_stop(&self) {
        let mut data = self.data.lock().unwrap();
        data.stopped = true;
        drop(data);
        self.cv.notify_all();
    }
}
