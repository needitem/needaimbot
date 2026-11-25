//! GamePC Screen Capture UDP Streamer — Rust port.
//!
//! Captures the Windows desktop via DXGI Desktop Duplication (DDA) and streams
//! BGRA frames over UDP to the Jetson inference PC.  Wire format is
//! byte-identical to the C++ version.

mod capture; // Worker B creates this — DDA capture implementation
mod config;
mod frame; // Worker B creates this — LatestFrameSlot condvar slot
mod stats;
mod udp;

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use socket2::{Domain, Protocol, SockAddr, Socket, Type};

use capture::{enumerate_capture_outputs, CaptureStatus, SimpleCapture};
use config::{clamp_packet_payload_bytes, load_config, parse_args, save_config, Config};
use frame::LatestFrameSlot;
use stats::{print_stats_line, StreamStats};
use udp::{
    read_frame_credit, send_frame_udp, warn_if_payload_exceeds_mtu, UDP_PIXEL_FORMAT_BGRA,
};

// ---------------------------------------------------------------------------
// Global running flag
// ---------------------------------------------------------------------------

static RUNNING: AtomicBool = AtomicBool::new(true);

// ---------------------------------------------------------------------------
// Windows realtime hints
// ---------------------------------------------------------------------------

/// Apply Windows thread priority and ideal-processor hints.
///
/// On non-Windows platforms this is a no-op (the code is cross-compiled for
/// Windows but we keep it compilable everywhere for `cargo check` on the dev
/// host).
#[cfg(target_os = "windows")]
fn apply_windows_realtime_hint(thread_name: Option<&str>, ideal_processor_offset: u32) {
    use windows::Win32::System::Threading::{
        GetActiveProcessorCount, GetCurrentThread, SetThreadIdealProcessor, SetThreadPriority,
        ALL_PROCESSOR_GROUPS, THREAD_PRIORITY_HIGHEST,
    };

    unsafe {
        let thread = GetCurrentThread();
        SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);

        let cpu_count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS as u16);
        if cpu_count > 1 {
            let offset = ideal_processor_offset.min(cpu_count - 1);
            let processor = cpu_count - 1 - offset;
            SetThreadIdealProcessor(thread, processor);
        }
    }

    // SetThreadDescription is optional (Win10 1607+). We skip the dynamic
    // lookup to keep this simple; the name is cosmetic.
    let _ = thread_name;
}

#[cfg(not(target_os = "windows"))]
fn apply_windows_realtime_hint(_thread_name: Option<&str>, _ideal_processor_offset: u32) {
    // No-op on non-Windows (allows `cargo check` on the dev host).
}

/// Enable 1ms timer resolution (Windows only).
#[cfg(target_os = "windows")]
fn enable_high_res_timer() -> bool {
    unsafe {
        windows::Win32::Media::timeBeginPeriod(1) == 0 // TIMERR_NOERROR
    }
}

#[cfg(not(target_os = "windows"))]
fn enable_high_res_timer() -> bool {
    true // no-op
}

/// Disable 1ms timer resolution.
#[cfg(target_os = "windows")]
fn disable_high_res_timer() {
    unsafe {
        windows::Win32::Media::timeEndPeriod(1);
    }
}

#[cfg(not(target_os = "windows"))]
fn disable_high_res_timer() {}

/// Set process priority to HIGH (Windows only).
#[cfg(target_os = "windows")]
fn set_high_priority_class() {
    use windows::Win32::System::Threading::{
        GetCurrentProcess, SetPriorityClass, HIGH_PRIORITY_CLASS,
    };
    unsafe {
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }
}

#[cfg(not(target_os = "windows"))]
fn set_high_priority_class() {}

// ---------------------------------------------------------------------------
// Exe directory helper
// ---------------------------------------------------------------------------

fn get_exe_directory() -> String {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            return parent.to_string_lossy().into_owned() + std::path::MAIN_SEPARATOR_STR;
        }
    }
    String::new()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // ---- 1. Load config from exe directory ----
    let exe_dir = get_exe_directory();
    let config_path = format!("{}config.ini", exe_dir);
    let mut config = load_config(&config_path).unwrap_or_else(|e| {
        eprintln!("Warning: failed to read config: {}", e);
        Config::default()
    });

    // ---- 2. Parse CLI args (override config) ----
    let args: Vec<String> = std::env::args().collect();
    match parse_args(&args, &mut config) {
        Ok(false) => {
            // --help printed
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Argument error: {}", e);
            std::process::exit(1);
        }
        Ok(true) => {}
    }
    config.packet_payload_bytes = clamp_packet_payload_bytes(config.packet_payload_bytes);

    // ---- 3. Enumerate outputs ----
    let available_outputs = enumerate_capture_outputs();
    if available_outputs.is_empty() {
        eprintln!("No capture outputs found");
        std::process::exit(1);
    }

    // ---- 4. GUI dialog (stub — the Win32 GUI is complex and optional) ----
    if config.use_gui {
        // TODO: Implement Win32 config dialog.  For now, the GUI is skipped
        // when running in CLI/headless mode.  Use --no-gui or UseGUI=0 in
        // config.ini. Printing a note so the operator knows.
        println!("GUI dialog not yet implemented in Rust port; proceeding with config.ini / CLI args");
        config.use_gui = false;
    }

    // Save config after GUI (or after deciding to skip it).
    if let Err(e) = save_config(&config, &config_path) {
        eprintln!("Warning: failed to save config to '{}': {}", config_path, e);
    }

    // ---- 5. Select output ----
    let selected_output = available_outputs
        .iter()
        .find(|o| o.index == config.output_index)
        .unwrap_or_else(|| {
            println!(
                "Invalid output index. Falling back to output {}",
                available_outputs[0].index
            );
            &available_outputs[0]
        });
    let output_index = selected_output.index;

    if selected_output.refresh_hz > 0 && config.target_fps < selected_output.refresh_hz {
        println!(
            "Note: target FPS ({}) is below monitor refresh rate ({}Hz); \
             capture is rate-capped. Set FPS >= refresh for lowest latency",
            config.target_fps, selected_output.refresh_hz
        );
    }
    if config.target_fps <= 0 {
        eprintln!("TargetFPS must be positive");
        std::process::exit(1);
    }
    if config.capture_width <= 0
        || config.capture_height <= 0
        || config.capture_width > u16::MAX as i32
        || config.capture_height > u16::MAX as i32
    {
        eprintln!(
            "CaptureWidth/CaptureHeight must be in range 1..{}",
            u16::MAX
        );
        std::process::exit(1);
    }

    // ---- Signal handling ----
    // Set Ctrl+C handler.
    let _ = ctrlc_handler();

    // ---- 6. Initialize capture ----
    let mut capture_device = SimpleCapture::new();
    if !capture_device.initialize(output_index) {
        eprintln!("Failed to initialize screen capture");
        std::process::exit(1);
    }

    println!("Output: {}", selected_output.label);
    println!(
        "Screen: {}x{}",
        capture_device.screen_width(),
        capture_device.screen_height()
    );

    if config.capture_width > capture_device.screen_width()
        || config.capture_height > capture_device.screen_height()
    {
        eprintln!("Capture region is larger than selected output");
        std::process::exit(1);
    }

    // Compute capture position (center offset or absolute).
    let output_center_x = capture_device.screen_width() / 2;
    let output_center_y = capture_device.screen_height() / 2;
    if config.use_capture_offset {
        config.capture_x =
            output_center_x + config.capture_offset_x - config.capture_width / 2;
        config.capture_y =
            output_center_y + config.capture_offset_y - config.capture_height / 2;
    } else if config.capture_x == 0 && config.capture_y == 0 {
        config.capture_x = (capture_device.screen_width() - config.capture_width) / 2;
        config.capture_y = (capture_device.screen_height() - config.capture_height) / 2;
    }

    let max_capture_x = capture_device.screen_width() - config.capture_width;
    let max_capture_y = capture_device.screen_height() - config.capture_height;
    let requested_x = config.capture_x;
    let requested_y = config.capture_y;
    config.capture_x = config.capture_x.clamp(0, max_capture_x);
    config.capture_y = config.capture_y.clamp(0, max_capture_y);
    if config.capture_x != requested_x || config.capture_y != requested_y {
        println!(
            "Capture region clamped from {},{} to {},{}",
            requested_x, requested_y, config.capture_x, config.capture_y
        );
    }

    println!(
        "Capture region: {},{} {}x{}",
        config.capture_x, config.capture_y, config.capture_width, config.capture_height
    );
    println!(
        "Capture center: {},{} (output center: {},{})",
        config.capture_x + config.capture_width / 2,
        config.capture_y + config.capture_height / 2,
        output_center_x,
        output_center_y
    );
    if config.use_capture_offset {
        println!(
            "Capture offset: {},{} (from selected output center)",
            config.capture_offset_x, config.capture_offset_y
        );
    } else {
        println!("Capture mode: absolute top-left (CaptureX/Y)");
    }

    let wire_bpp: u8 = 4; // BGRA = 4 bytes/pixel
    let wire_pixel_format: u8 = UDP_PIXEL_FORMAT_BGRA;
    let effective_payload = clamp_packet_payload_bytes(config.packet_payload_bytes);
    warn_if_payload_exceeds_mtu(effective_payload);
    let max_payload = effective_payload as usize;

    // ---- 7. Create UDP socket ----
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP)).unwrap_or_else(|e| {
        eprintln!("Failed to create socket: {}", e);
        std::process::exit(1);
    });

    // 2 MB send buffer
    let _ = socket.set_send_buffer_size(2 * 1024 * 1024);
    socket.set_nonblocking(true).unwrap_or_else(|e| {
        eprintln!("Warning: failed to set UDP sender socket to non-blocking mode: {}", e);
    });

    // ---- 8. Bind if local_bind_ip set ----
    if !config.local_bind_ip.is_empty() {
        let bind_ip: Ipv4Addr = config.local_bind_ip.parse().unwrap_or_else(|_| {
            eprintln!("Invalid LocalBindIP: {}", config.local_bind_ip);
            std::process::exit(1);
        });
        let bind_addr = SocketAddrV4::new(bind_ip, 0);
        socket
            .bind(&SockAddr::from(SocketAddr::V4(bind_addr)))
            .unwrap_or_else(|e| {
                eprintln!(
                    "Failed to bind sender socket to {}: {}",
                    config.local_bind_ip, e
                );
                std::process::exit(1);
            });
    }

    // ---- 9. Set destination addr ----
    let dest_ip: Ipv4Addr = config.inference_ip.parse().unwrap_or_else(|_| {
        eprintln!("Invalid InferenceIP: {}", config.inference_ip);
        std::process::exit(1);
    });
    let dest_addr = SocketAddr::V4(SocketAddrV4::new(dest_ip, config.send_port));

    // Print bound address info.
    let bound_ip_text = socket
        .local_addr()
        .ok()
        .and_then(|sa| sa.as_socket_ipv4().map(|v4| v4.ip().to_string()))
        .unwrap_or_else(|| "auto".to_string());

    println!(
        "GamePC Streamer (BGRA, {}B payload, UDP V3) started",
        max_payload
    );
    println!("Sending to: {}:{}", config.inference_ip, config.send_port);
    println!("Local bind IP: {}", bound_ip_text);
    println!(
        "Capture: event-driven (synced to presents), FPS cap: {}",
        config.target_fps
    );
    println!("Capture timeout: {}ms", config.capture_timeout_ms);
    println!("Press Ctrl+C to exit\n");

    // ---- OS tuning ----
    let high_res_timer = enable_high_res_timer();
    if !high_res_timer {
        eprintln!("Warning: failed to enable 1ms timer resolution");
    }
    set_high_priority_class();

    // ---- Shared state ----
    let latest_frame = Arc::new(LatestFrameSlot::new());
    let stats = Arc::new(StreamStats::new());
    let frame_credits = Arc::new(AtomicI32::new(1)); // Bootstrap one credit
    let min_credit_frame_id = Arc::new(AtomicU32::new(0));

    let credit_recovery_timeout =
        Duration::from_millis(config.credit_timeout_ms.clamp(5, 1000) as u64);

    let socket = Arc::new(socket);

    // ---- 10. Credit thread ----
    let credit_socket = Arc::clone(&socket);
    let credit_frame_credits = Arc::clone(&frame_credits);
    let credit_min_id = Arc::clone(&min_credit_frame_id);
    let credit_stats = Arc::clone(&stats);
    let credit_latest = Arc::clone(&latest_frame);
    let credit_thread = std::thread::Builder::new()
        .name("udp-credit".into())
        .spawn(move || {
            apply_windows_realtime_hint(Some("udp-credit"), 1);
            while RUNNING.load(Ordering::Relaxed) {
                let notify = || {
                    credit_latest.cv.notify_one();
                };
                let mut got_any = false;
                while read_frame_credit(
                    &credit_socket,
                    &credit_frame_credits,
                    &credit_min_id,
                    &credit_stats.credit_packets,
                    &notify,
                ) {
                    got_any = true;
                }
                if !got_any {
                    // Brief sleep to avoid busy-spinning when no data arrives.
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        })
        .expect("failed to spawn credit thread");

    // ---- 11. Sender thread ----
    let send_socket = Arc::clone(&socket);
    let send_frame_credits = Arc::clone(&frame_credits);
    let send_min_id = Arc::clone(&min_credit_frame_id);
    let send_stats = Arc::clone(&stats);
    let send_latest = Arc::clone(&latest_frame);
    let sender_thread = std::thread::Builder::new()
        .name("udp-send".into())
        .spawn(move || {
            apply_windows_realtime_hint(Some("udp-send"), 0);
            let mut bgra_local: Vec<u8> = Vec::new();
            let mut frame_id: u32 = 0;
            let mut last_sent_sequence: u64 = 0;
            let mut last_send_attempt = Instant::now() - credit_recovery_timeout;

            while RUNNING.load(Ordering::Relaxed) {
                let mut send_width: u16 = 0;
                let mut send_height: u16 = 0;
                let mut send_capture_unix_micros: u64 = 0;

                // Wait for a new frame + credit (or recovery timeout).
                {
                    let mut lock = send_latest.data.lock().unwrap();
                    lock = send_latest
                        .cv
                        .wait_timeout(lock, Duration::from_millis(1))
                        .unwrap()
                        .0;

                    if lock.stopped && !lock.has_frame {
                        break;
                    }
                    if !lock.has_frame || lock.sequence == last_sent_sequence {
                        continue;
                    }

                    let has_credit =
                        send_frame_credits.load(Ordering::Acquire) > 0;
                    let is_recovery = !has_credit
                        && Instant::now().duration_since(last_send_attempt)
                            >= credit_recovery_timeout;
                    if !has_credit && !is_recovery {
                        continue;
                    }
                    if has_credit {
                        send_frame_credits.fetch_sub(1, Ordering::AcqRel);
                    } else {
                        send_stats.recovery_sends.fetch_add(1, Ordering::Relaxed);
                    }

                    send_width = lock.width;
                    send_height = lock.height;
                    last_sent_sequence = lock.sequence;
                    send_capture_unix_micros = lock.capture_unix_micros;
                    std::mem::swap(&mut bgra_local, &mut lock.bgra);
                    lock.has_frame = false;
                }

                if bgra_local.is_empty() || send_width == 0 || send_height == 0 {
                    continue;
                }

                let frame_size = bgra_local.len();
                if frame_size == 0 || frame_size > u32::MAX as usize {
                    continue;
                }
                send_stats.output_frames.fetch_add(1, Ordering::Relaxed);

                let requested_min =
                    send_min_id.load(Ordering::Acquire);
                if frame_id < requested_min {
                    frame_id = requested_min;
                }

                let send_start = Instant::now();
                let result = send_frame_udp(
                    &send_socket,
                    &dest_addr,
                    &bgra_local,
                    frame_size,
                    frame_id,
                    send_width,
                    send_height,
                    wire_pixel_format,
                    wire_bpp,
                    max_payload,
                    send_capture_unix_micros,
                );
                let send_elapsed = send_start.elapsed();

                send_stats.total_send_us.fetch_add(
                    send_elapsed.as_micros() as i64,
                    Ordering::Relaxed,
                );
                send_stats.total_packets.fetch_add(
                    result.total_packets as u64,
                    Ordering::Relaxed,
                );

                frame_id = frame_id.wrapping_add(1);
                last_send_attempt = Instant::now();

                if result.would_block {
                    send_stats.would_block_drops.fetch_add(1, Ordering::Relaxed);
                }
                if !result.sent {
                    send_stats.dropped_frames.fetch_add(1, Ordering::Relaxed);
                } else {
                    send_stats.sent_frames.fetch_add(1, Ordering::Relaxed);
                    send_stats
                        .total_bytes
                        .fetch_add(frame_size as u64, Ordering::Relaxed);
                }
            }
        })
        .expect("failed to spawn sender thread");

    // ---- 12. Main capture loop ----
    let min_capture_interval =
        Duration::from_micros((1_000_000.0 / config.target_fps as f64) as u64);
    let mut last_capture_time = Instant::now() - min_capture_interval;
    let mut stats_start = Instant::now();
    let mut frame_data: Vec<u8> = Vec::new();

    // 0 would busy-spin, so clamp to at least 1ms.
    let capture_timeout_ms = config.capture_timeout_ms.clamp(1, 100) as u32;

    apply_windows_realtime_hint(Some("capture-main"), 2);

    while RUNNING.load(Ordering::Relaxed) {
        // Rate cap: sleep only when capturing faster than TargetFPS.
        let next_allowed = last_capture_time + min_capture_interval;
        let now = Instant::now();
        if now < next_allowed {
            std::thread::sleep(next_allowed - now);
        }

        // Capture
        let t1 = Instant::now();
        let result = capture_device.capture_frame(
            &mut frame_data,
            config.capture_x,
            config.capture_y,
            config.capture_width,
            config.capture_height,
            capture_timeout_ms,
        );
        let t2 = Instant::now();
        stats.capture_attempts.fetch_add(1, Ordering::Relaxed);

        if result.success {
            last_capture_time = Instant::now();
            stats.captured_frames.fetch_add(1, Ordering::Relaxed);
            stats.total_capture_us.fetch_add(
                t2.duration_since(t1).as_micros() as i64,
                Ordering::Relaxed,
            );
            stats
                .total_acquire_us
                .fetch_add(result.acquire_us, Ordering::Relaxed);
            stats
                .total_readback_us
                .fetch_add(result.readback_us, Ordering::Relaxed);

            // Post to latest-frame slot
            {
                let mut lock = latest_frame.data.lock().unwrap();
                std::mem::swap(&mut lock.bgra, &mut frame_data);
                lock.width = config.capture_width as u16;
                lock.height = config.capture_height as u16;
                lock.capture_unix_micros = result.capture_unix_micros;
                lock.has_frame = true;
                lock.sequence += 1;
            }
            latest_frame.cv.notify_one();
        } else {
            stats.skipped_frames.fetch_add(1, Ordering::Relaxed);
            stats
                .last_capture_hr
                .store(result.hr as u32, Ordering::Relaxed);
            match result.status {
                CaptureStatus::Timeout => {
                    stats.capture_timeouts.fetch_add(1, Ordering::Relaxed);
                }
                CaptureStatus::AccessLost => {
                    stats.capture_access_lost.fetch_add(1, Ordering::Relaxed);
                }
                CaptureStatus::Failed => {
                    stats.capture_failures.fetch_add(1, Ordering::Relaxed);
                }
                CaptureStatus::Captured => {}
            }
        }

        // Print stats if >= 1 second elapsed.
        stats_start = print_stats_line(&stats, stats_start, max_payload);
    }

    // ---- Graceful shutdown ----
    println!("\nShutting down...");
    {
        let mut lock = latest_frame.data.lock().unwrap();
        lock.stopped = true;
    }
    latest_frame.cv.notify_all();
    let _ = sender_thread.join();
    let _ = credit_thread.join();

    if high_res_timer {
        disable_high_res_timer();
    }

    // Socket is dropped automatically.
}

// ---------------------------------------------------------------------------
// Ctrl+C handler
// ---------------------------------------------------------------------------

fn ctrlc_handler() -> Result<(), Box<dyn std::error::Error>> {
    // Use the C runtime's signal() which is available on both Windows (MSVCRT)
    // and POSIX without an extra crate dependency.
    unsafe {
        signal(2 /* SIGINT */, sigint_handler);
    }
    Ok(())
}

extern "C" fn sigint_handler(_sig: i32) {
    RUNNING.store(false, Ordering::SeqCst);
}

// Binding to the C runtime signal() — available on Windows (MSVCRT) and POSIX.
extern "C" {
    fn signal(sig: std::os::raw::c_int, handler: extern "C" fn(i32)) -> usize;
}
