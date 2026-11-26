//! Streaming statistics with atomic counters.
//!
//! All fields are atomic so the capture, sender, and credit threads can update
//! them lock-free. `print_stats_line` exchanges every counter to zero and
//! prints a single-line status (same format as the C++ streamer).

use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// StreamStats
// ---------------------------------------------------------------------------

pub struct StreamStats {
    pub capture_attempts: AtomicU64,
    pub captured_frames: AtomicU64,
    pub skipped_frames: AtomicU64,
    pub capture_timeouts: AtomicU64,
    pub capture_access_lost: AtomicU64,
    pub capture_failures: AtomicU64,
    pub output_frames: AtomicU64,
    pub sent_frames: AtomicU64,
    pub dropped_frames: AtomicU64,
    pub would_block_drops: AtomicU64,
    pub credit_packets: AtomicU64,
    pub recovery_sends: AtomicU64,
    pub total_bytes: AtomicU64,
    pub total_packets: AtomicU64,
    pub total_capture_us: AtomicI64,
    pub total_acquire_us: AtomicI64,
    pub total_readback_us: AtomicI64,
    pub total_convert_us: AtomicI64,
    pub total_send_us: AtomicI64,
    pub last_capture_hr: AtomicU32,
}

impl StreamStats {
    pub fn new() -> Self {
        Self {
            capture_attempts: AtomicU64::new(0),
            captured_frames: AtomicU64::new(0),
            skipped_frames: AtomicU64::new(0),
            capture_timeouts: AtomicU64::new(0),
            capture_access_lost: AtomicU64::new(0),
            capture_failures: AtomicU64::new(0),
            output_frames: AtomicU64::new(0),
            sent_frames: AtomicU64::new(0),
            dropped_frames: AtomicU64::new(0),
            would_block_drops: AtomicU64::new(0),
            credit_packets: AtomicU64::new(0),
            recovery_sends: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            total_packets: AtomicU64::new(0),
            total_capture_us: AtomicI64::new(0),
            total_acquire_us: AtomicI64::new(0),
            total_readback_us: AtomicI64::new(0),
            total_convert_us: AtomicI64::new(0),
            total_send_us: AtomicI64::new(0),
            last_capture_hr: AtomicU32::new(0),
        }
    }
}

impl Default for StreamStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Status line printing
// ---------------------------------------------------------------------------

/// Exchange all counters to zero, compute rates and averages, and print a
/// single-line status to stdout (same format as the C++ streamer).
///
/// Returns the new `stats_start` instant to use for the next interval.
pub fn print_stats_line(
    stats: &StreamStats,
    stats_start: Instant,
    max_payload: usize,
) -> Instant {
    let now = Instant::now();
    let seconds = now.duration_since(stats_start).as_secs_f64();
    if seconds < 1.0 {
        return stats_start; // Not time yet
    }

    // Exchange all counters atomically.
    let r = Ordering::Relaxed;
    let capture_attempts = stats.capture_attempts.swap(0, r);
    let captured_frames = stats.captured_frames.swap(0, r);
    let skipped_frames = stats.skipped_frames.swap(0, r);
    let capture_timeouts = stats.capture_timeouts.swap(0, r);
    let capture_access_lost = stats.capture_access_lost.swap(0, r);
    let capture_failures = stats.capture_failures.swap(0, r);
    let output_frames = stats.output_frames.swap(0, r);
    let sent_frames = stats.sent_frames.swap(0, r);
    let dropped_frames = stats.dropped_frames.swap(0, r);
    let would_block_drops = stats.would_block_drops.swap(0, r);
    let credit_packets = stats.credit_packets.swap(0, r);
    let recovery_sends = stats.recovery_sends.swap(0, r);
    let total_bytes = stats.total_bytes.swap(0, r);
    let total_packets = stats.total_packets.swap(0, r);
    let total_capture_us = stats.total_capture_us.swap(0, r);
    let total_acquire_us = stats.total_acquire_us.swap(0, r);
    let total_readback_us = stats.total_readback_us.swap(0, r);
    let total_convert_us = stats.total_convert_us.swap(0, r);
    let total_send_us = stats.total_send_us.swap(0, r);

    let poll_fps = capture_attempts as f64 / seconds;
    let cap_fps = captured_frames as f64 / seconds;
    let skip_fps = skipped_frames as f64 / seconds;
    let out_fps = output_frames as f64 / seconds;
    let send_fps = sent_frames as f64 / seconds;
    let mbps = (total_bytes as f64 * 8.0) / (seconds * 1_000_000.0);

    let avg_capture = if captured_frames > 0 {
        total_capture_us as f64 / captured_frames as f64 / 1000.0
    } else {
        0.0
    };
    let avg_acquire = if captured_frames > 0 {
        total_acquire_us as f64 / captured_frames as f64 / 1000.0
    } else {
        0.0
    };
    let avg_readback = if captured_frames > 0 {
        total_readback_us as f64 / captured_frames as f64 / 1000.0
    } else {
        0.0
    };
    let avg_convert = if output_frames > 0 {
        total_convert_us as f64 / output_frames as f64 / 1000.0
    } else {
        0.0
    };
    let avg_send = if output_frames > 0 {
        total_send_us as f64 / output_frames as f64 / 1000.0
    } else {
        0.0
    };
    let drop_pct = if output_frames > 0 {
        dropped_frames as f64 * 100.0 / output_frames as f64
    } else {
        0.0
    };
    let packets_per_frame = if sent_frames > 0 {
        total_packets as f64 / sent_frames as f64
    } else {
        0.0
    };

    let last_hr = stats.last_capture_hr.load(Ordering::Relaxed);

    let mut line = format!(
        "\rCap:{:.0} Out:{:.0} Snd:{:.0} | {:.1}ms({:.1}+{:.1}+{:.1}) | {:.0}Mbps | Drop:{:.0}%({}) | Cr:{} Rc:{}",
        cap_fps,
        out_fps,
        send_fps,
        avg_capture + avg_convert + avg_send,
        avg_acquire,
        avg_readback,
        avg_send,
        mbps,
        drop_pct,
        would_block_drops,
        credit_packets,
        recovery_sends,
    );

    if capture_timeouts > 0 || capture_access_lost > 0 || capture_failures > 0 {
        line.push_str(&format!(" CT:{} AL:{} CF:{}", capture_timeouts, capture_access_lost, capture_failures));
    }
    if last_hr != 0 {
        line.push_str(&format!(" HR:0x{:X}", last_hr));
    }
    let visible_len = line.len() - 1;
    if visible_len < 120 {
        line.extend(std::iter::repeat(' ').take(120 - visible_len));
    }

    print!("{}", line);
    use std::io::Write;
    let _ = std::io::stdout().flush();

    now
}
