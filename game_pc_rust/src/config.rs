//! Configuration loading, saving, and CLI argument parsing.
//!
//! INI format is compatible with the C++ GamePC streamer. Sections ([Network],
//! [Capture], [Performance]) are ignored during parsing; only key=value lines
//! matter. Comments start with `;`, `#`, or `[`.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use crate::udp::{MAX_PACKET_PAYLOAD_BYTES, MIN_PACKET_PAYLOAD_BYTES};

// ---------------------------------------------------------------------------
// Config struct
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Config {
    pub inference_ip: String,
    pub send_port: u16,
    pub local_bind_ip: String,
    pub capture_x: i32,
    pub capture_y: i32,
    pub capture_width: i32,
    pub capture_height: i32,
    pub use_capture_offset: bool,
    pub capture_offset_x: i32,
    pub capture_offset_y: i32,
    pub target_fps: i32,
    pub output_index: i32,
    pub use_gui: bool,
    /// MTU-safe by default. 1400 keeps one application chunk inside one
    /// Ethernet frame (1500 MTU - 20 IP - 8 UDP - 44 app header = 1428).
    pub packet_payload_bytes: i32,
    pub capture_timeout_ms: i32,
    /// Credit mode reduces wasted network traffic and stale frames.
    pub credit_timeout_ms: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            inference_ip: "192.168.1.100".to_string(),
            send_port: 5007,
            local_bind_ip: String::new(),
            capture_x: 0,
            capture_y: 0,
            capture_width: 256,
            capture_height: 256,
            use_capture_offset: false,
            capture_offset_x: 0,
            capture_offset_y: 0,
            target_fps: 90,
            output_index: 0,
            use_gui: true,
            packet_payload_bytes: 1400,
            capture_timeout_ms: 5,
            credit_timeout_ms: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a boolean value from text. Recognises `1`, `true`, `yes`, `on`
/// (case-insensitive) as `true`; everything else is `false`.
pub fn parse_bool(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    matches!(lower.as_str(), "1" | "true" | "yes" | "on")
}

/// Clamp a packet payload byte count to the valid range.
pub fn clamp_packet_payload_bytes(bytes: i32) -> i32 {
    bytes.clamp(MIN_PACKET_PAYLOAD_BYTES, MAX_PACKET_PAYLOAD_BYTES)
}

// ---------------------------------------------------------------------------
// INI loading
// ---------------------------------------------------------------------------

/// Load configuration from an INI file. Missing file is not an error; a
/// message is printed and defaults are returned.
pub fn load_config<P: AsRef<Path>>(filename: P) -> io::Result<Config> {
    let filename = filename.as_ref();
    let contents = match fs::read_to_string(filename) {
        Ok(c) => c,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            println!(
                "Config file '{}' not found, using defaults",
                filename.display()
            );
            return Ok(Config::default());
        }
        Err(e) => return Err(e),
    };

    let mut config = Config::default();
    let mut capture_offset_key_seen = false;
    let mut use_capture_offset_key_seen = false;

    for raw_line in contents.lines() {
        // Trim whitespace
        let line = raw_line.trim();

        // Skip empty lines, comments, and section headers
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') || line.starts_with('[')
        {
            continue;
        }

        // Parse key=value
        let Some(eq_pos) = line.find('=') else {
            continue;
        };
        let key = line[..eq_pos].trim();
        let value = line[eq_pos + 1..].trim();

        match key {
            "InferenceIP" => config.inference_ip = value.to_string(),
            "SendPort" => {
                if let Ok(v) = value.parse::<u16>() {
                    config.send_port = v;
                }
            }
            "LocalBindIP" => config.local_bind_ip = value.to_string(),
            "CaptureX" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.capture_x = v;
                }
            }
            "CaptureY" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.capture_y = v;
                }
            }
            "CaptureWidth" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.capture_width = v;
                }
            }
            "CaptureHeight" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.capture_height = v;
                }
            }
            "UseCaptureOffset" => {
                config.use_capture_offset = parse_bool(value);
                use_capture_offset_key_seen = true;
            }
            "CaptureOffsetX" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.capture_offset_x = v;
                    capture_offset_key_seen = true;
                }
            }
            "CaptureOffsetY" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.capture_offset_y = v;
                    capture_offset_key_seen = true;
                }
            }
            "OutputIndex" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.output_index = v.max(0);
                }
            }
            "TargetFPS" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.target_fps = v;
                }
            }
            "UseGUI" => {
                config.use_gui = parse_bool(value);
            }
            "PacketPayloadBytes" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.packet_payload_bytes = clamp_packet_payload_bytes(v);
                }
            }
            "CaptureTimeoutMs" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.capture_timeout_ms = v.clamp(0, 100);
                }
            }
            "CreditTimeoutMs" => {
                if let Ok(v) = value.parse::<i32>() {
                    config.credit_timeout_ms = v.clamp(5, 1000);
                }
            }
            _ => {}
        }
    }

    // If CaptureOffset keys were set but UseCaptureOffset was not explicitly
    // stated, enable offset mode (matches C++ behavior).
    if capture_offset_key_seen && !use_capture_offset_key_seen {
        config.use_capture_offset = true;
    }

    println!("Loaded config from '{}'", filename.display());
    Ok(config)
}

// ---------------------------------------------------------------------------
// INI saving
// ---------------------------------------------------------------------------

/// Save configuration to an INI file (overwrites).
pub fn save_config<P: AsRef<Path>>(config: &Config, filename: P) -> io::Result<()> {
    let mut f = fs::File::create(filename)?;

    writeln!(f, "[Network]")?;
    writeln!(f, "InferenceIP={}", config.inference_ip)?;
    writeln!(f, "SendPort={}", config.send_port)?;
    writeln!(f, "LocalBindIP={}", config.local_bind_ip)?;
    writeln!(f)?;

    writeln!(f, "[Capture]")?;
    writeln!(f, "CaptureX={}", config.capture_x)?;
    writeln!(f, "CaptureY={}", config.capture_y)?;
    writeln!(f, "CaptureWidth={}", config.capture_width)?;
    writeln!(f, "CaptureHeight={}", config.capture_height)?;
    writeln!(
        f,
        "UseCaptureOffset={}",
        if config.use_capture_offset { 1 } else { 0 }
    )?;
    writeln!(f, "CaptureOffsetX={}", config.capture_offset_x)?;
    writeln!(f, "CaptureOffsetY={}", config.capture_offset_y)?;
    writeln!(f, "OutputIndex={}", config.output_index)?;
    writeln!(f)?;

    writeln!(f, "[Performance]")?;
    writeln!(f, "TargetFPS={}", config.target_fps)?;
    writeln!(
        f,
        "UseGUI={}",
        if config.use_gui { 1 } else { 0 }
    )?;
    writeln!(f, "PacketPayloadBytes={}", config.packet_payload_bytes)?;
    writeln!(f, "CaptureTimeoutMs={}", config.capture_timeout_ms)?;
    writeln!(f, "CreditTimeoutMs={}", config.credit_timeout_ms)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

fn print_usage(prog: &str) {
    println!("Usage: {} [options]", prog);
    println!("Options:");
    println!("  --ip <addr>       Inference PC IP (default: from config.ini)");
    println!("  --port <port>     Send port (default: from config.ini)");
    println!("  --bind-ip <addr>  Local NIC IP for sender socket bind (optional)");
    println!("  --region <x,y,w,h> Capture region (default: from config.ini)");
    println!("  --offset <x,y>    Capture center offset from selected output center");
    println!("  --output <idx>    Capture monitor index (default: from config.ini)");
    println!("  --fps <num>       Target FPS (default: from config.ini)");
    println!("  --payload-bytes <n> UDP payload bytes per packet, 512..60000");
    println!("  --capture-timeout-ms <n> Desktop Duplication wait, 0..100");
    println!("  --credit-timeout-ms <n> Recovery send timeout if credit packet is lost");
    println!("  --gui             Show startup config GUI");
    println!("  --no-gui          Skip startup config GUI");
    println!();
    println!("Config file: config.ini");
}

/// Parse command-line arguments, overriding values in `config`.
///
/// Returns `Ok(true)` on success, `Ok(false)` when `--help` was printed
/// (caller should exit 0), and `Err` on parse errors.
pub fn parse_args(args: &[String], config: &mut Config) -> Result<bool, String> {
    let prog = args.first().map(|s| s.as_str()).unwrap_or("game_pc_streamer");
    let mut i = 1;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "--ip" => {
                i += 1;
                config.inference_ip = args
                    .get(i)
                    .ok_or_else(|| "--ip requires an argument".to_string())?
                    .clone();
            }
            "--port" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--port requires an argument".to_string())?;
                config.send_port = val
                    .parse::<u16>()
                    .map_err(|e| format!("--port: {}", e))?;
            }
            "--bind-ip" => {
                i += 1;
                config.local_bind_ip = args
                    .get(i)
                    .ok_or_else(|| "--bind-ip requires an argument".to_string())?
                    .clone();
            }
            "--region" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--region requires an argument".to_string())?;
                let parts: Vec<&str> = val.split(',').collect();
                if parts.len() == 4 {
                    config.capture_x = parts[0].parse().unwrap_or(config.capture_x);
                    config.capture_y = parts[1].parse().unwrap_or(config.capture_y);
                    config.capture_width = parts[2].parse().unwrap_or(config.capture_width);
                    config.capture_height = parts[3].parse().unwrap_or(config.capture_height);
                }
                config.use_capture_offset = false;
            }
            "--offset" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--offset requires an argument".to_string())?;
                let parts: Vec<&str> = val.split(',').collect();
                if parts.len() == 2 {
                    config.capture_offset_x =
                        parts[0].parse().unwrap_or(config.capture_offset_x);
                    config.capture_offset_y =
                        parts[1].parse().unwrap_or(config.capture_offset_y);
                }
                config.use_capture_offset = true;
            }
            "--output" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--output requires an argument".to_string())?;
                config.output_index = val
                    .parse::<i32>()
                    .map_err(|e| format!("--output: {}", e))?
                    .max(0);
            }
            "--fps" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--fps requires an argument".to_string())?;
                config.target_fps = val
                    .parse::<i32>()
                    .map_err(|e| format!("--fps: {}", e))?;
            }
            "--payload-bytes" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--payload-bytes requires an argument".to_string())?;
                config.packet_payload_bytes = clamp_packet_payload_bytes(
                    val.parse::<i32>()
                        .map_err(|e| format!("--payload-bytes: {}", e))?,
                );
            }
            "--capture-timeout-ms" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--capture-timeout-ms requires an argument".to_string())?;
                config.capture_timeout_ms = val
                    .parse::<i32>()
                    .map_err(|e| format!("--capture-timeout-ms: {}", e))?
                    .clamp(0, 100);
            }
            "--credit-timeout-ms" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--credit-timeout-ms requires an argument".to_string())?;
                config.credit_timeout_ms = val
                    .parse::<i32>()
                    .map_err(|e| format!("--credit-timeout-ms: {}", e))?
                    .clamp(5, 1000);
            }
            "--gui" => {
                config.use_gui = true;
            }
            "--no-gui" => {
                config.use_gui = false;
            }
            "--help" | "-h" => {
                print_usage(prog);
                return Ok(false);
            }
            _ => {
                // Unknown args are silently ignored (matches C++ behaviour).
            }
        }
        i += 1;
    }
    Ok(true)
}
