//! DDA (Desktop Duplication API) screen capture with a 2-texture staging ring
//! buffer for pipelined GPU->CPU readback.
//!
//! This is a Rust port of the `SimpleCapture` class and
//! `enumerateCaptureOutputs()` from the C++ `game_pc/src/main.cpp`.

use std::time::SystemTime;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::UI::WindowsAndMessaging::*;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Outcome of a single `capture_frame` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureStatus {
    /// A new desktop frame was successfully captured.
    Captured,
    /// `AcquireNextFrame` timed out (no new frame from the compositor).
    Timeout,
    /// Desktop duplication access was lost (e.g. mode change); duplication has
    /// been re-created and the caller should retry.
    AccessLost,
    /// An unrecoverable error occurred.
    Failed,
}

/// Detailed result returned by [`SimpleCapture::capture_frame`].
pub struct CaptureFrameResult {
    pub success: bool,
    pub status: CaptureStatus,
    /// The raw HRESULT from the last significant DirectX call.
    pub hr: i32,
    /// Wall-clock capture timestamp (microseconds since the UNIX epoch), stamped
    /// the instant `AcquireNextFrame` returns a new frame.
    pub capture_unix_micros: u64,
    /// Microseconds spent blocked inside `AcquireNextFrame` (present wait).
    pub acquire_us: i64,
    /// Microseconds spent on GPU copy + Map + CPU memcpy (the pipelineable part).
    pub readback_us: i64,
}

impl Default for CaptureFrameResult {
    fn default() -> Self {
        Self {
            success: false,
            status: CaptureStatus::Failed,
            hr: 0,
            capture_unix_micros: 0,
            acquire_us: 0,
            readback_us: 0,
        }
    }
}

/// Describes one enumerated DXGI output (monitor).
pub struct OutputInfo {
    pub index: i32,
    pub width: i32,
    pub height: i32,
    pub refresh_hz: i32,
    pub primary: bool,
    pub device_name: String,
    pub label: String,
}

// ---------------------------------------------------------------------------
// Output enumeration
// ---------------------------------------------------------------------------

/// Enumerate all DXGI outputs attached to the default hardware adapter.
///
/// Creates a temporary D3D11 device with BGRA support, walks each output, and
/// collects resolution, refresh rate, primary flag and a human-readable label.
pub fn enumerate_capture_outputs() -> Vec<OutputInfo> {
    let mut outputs = Vec::new();

    // Create a temporary D3D11 device.
    let feature_levels = [D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0];
    let flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    let mut device: Option<ID3D11Device> = None;
    let mut obtained = D3D_FEATURE_LEVEL_11_0;
    let mut context: Option<ID3D11DeviceContext> = None;

    let hr = unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            None,
            flags,
            Some(&feature_levels),
            D3D11_SDK_VERSION,
            Some(&mut device),
            Some(&mut obtained),
            Some(&mut context),
        )
    };
    if hr.is_err() {
        return outputs;
    }
    let device = match device {
        Some(d) => d,
        None => return outputs,
    };

    // Device -> DXGI device -> adapter.
    let dxgi_device: IDXGIDevice = match device.cast() {
        Ok(d) => d,
        Err(_) => return outputs,
    };
    let adapter: IDXGIAdapter = match unsafe { dxgi_device.GetAdapter() } {
        Ok(a) => a,
        Err(_) => return outputs,
    };

    // Walk outputs.
    for index in 0u32.. {
        let output: IDXGIOutput = match unsafe { adapter.EnumOutputs(index) } {
            Ok(o) => o,
            Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(_) => continue,
        };

        let desc = match unsafe { output.GetDesc() } {
            Ok(d) => d,
            Err(_) => continue,
        };

        let width = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left;
        let height = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;
        let device_name = wchar_to_string(&desc.DeviceName);

        // Primary flag via GetMonitorInfo.
        let mut primary = false;
        if !desc.Monitor.is_invalid() {
            let mut mi = MONITORINFO {
                cbSize: std::mem::size_of::<MONITORINFO>() as u32,
                ..Default::default()
            };
            if unsafe { GetMonitorInfoW(desc.Monitor, &mut mi) }.as_bool() {
                primary = (mi.dwFlags & MONITORINFOF_PRIMARY) != 0;
            }
        }

        // Refresh rate via EnumDisplaySettingsW.
        let mut refresh_hz: i32 = 0;
        let mut dm = DEVMODEW {
            dmSize: std::mem::size_of::<DEVMODEW>() as u16,
            ..Default::default()
        };
        let dev_name_pcwstr = windows::core::PCWSTR(desc.DeviceName.as_ptr());
        if unsafe { EnumDisplaySettingsW(dev_name_pcwstr, ENUM_CURRENT_SETTINGS, &mut dm) }.as_bool()
        {
            if dm.dmDisplayFrequency > 1 {
                refresh_hz = dm.dmDisplayFrequency as i32;
            }
        }

        // Build human-readable label.
        let primary_suffix = if primary { ", Primary" } else { "" };
        let label = if refresh_hz > 0 {
            format!(
                "[{}] {}  {}x{} @ {}Hz{}",
                index, device_name, width, height, refresh_hz, primary_suffix
            )
        } else {
            format!(
                "[{}] {}  {}x{}{}",
                index, device_name, width, height, primary_suffix
            )
        };

        outputs.push(OutputInfo {
            index: index as i32,
            width,
            height,
            refresh_hz,
            primary,
            device_name,
            label,
        });
    }

    outputs
}

// ---------------------------------------------------------------------------
// SimpleCapture
// ---------------------------------------------------------------------------

/// DDA-based desktop capture with a 2-texture staging ring buffer.
pub struct SimpleCapture {
    device: Option<ID3D11Device>,
    context: Option<ID3D11DeviceContext>,
    output: Option<IDXGIOutput>,
    duplication: Option<IDXGIOutputDuplication>,

    /// Two staging textures for pipelined GPU readback.
    staging: [Option<ID3D11Texture2D>; 2],
    /// Current write target index (0 or 1).
    staging_index: usize,
    /// `true` after the first frame has been copied into a staging texture, so
    /// the "previous" slot is now valid for mapping.
    staging_ready: bool,
    staging_width: i32,
    staging_height: i32,

    screen_width: i32,
    screen_height: i32,
}

impl SimpleCapture {
    pub fn new() -> Self {
        Self {
            device: None,
            context: None,
            output: None,
            duplication: None,
            staging: [None, None],
            staging_index: 0,
            staging_ready: false,
            staging_width: 0,
            staging_height: 0,
            screen_width: 0,
            screen_height: 0,
        }
    }

    /// Create the D3D11 device, obtain the DXGI output at `output_index`, and
    /// set up desktop duplication.
    pub fn initialize(&mut self, output_index: i32) -> bool {
        let feature_levels = [D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0];
        let flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

        let mut device: Option<ID3D11Device> = None;
        let mut obtained = D3D_FEATURE_LEVEL_11_0;
        let mut context: Option<ID3D11DeviceContext> = None;

        let hr = unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                None,
                flags,
                Some(&feature_levels),
                D3D11_SDK_VERSION,
                Some(&mut device),
                Some(&mut obtained),
                Some(&mut context),
            )
        };
        if hr.is_err() {
            return false;
        }
        let device = match device {
            Some(d) => d,
            None => return false,
        };
        let context = match context {
            Some(c) => c,
            None => return false,
        };

        // Device -> DXGI device -> adapter.
        let dxgi_device: IDXGIDevice = match device.cast() {
            Ok(d) => d,
            Err(_) => return false,
        };
        let adapter: IDXGIAdapter = match unsafe { dxgi_device.GetAdapter() } {
            Ok(a) => a,
            Err(_) => return false,
        };

        let output: IDXGIOutput = match unsafe { adapter.EnumOutputs(output_index as u32) } {
            Ok(o) => o,
            Err(_) => return false,
        };

        // Cast to IDXGIOutput1 for DuplicateOutput.
        let output1: IDXGIOutput1 = match output.cast() {
            Ok(o) => o,
            Err(_) => return false,
        };
        let duplication = match unsafe { output1.DuplicateOutput(&device) } {
            Ok(d) => d,
            Err(_) => return false,
        };

        // Store screen dimensions from the output descriptor.
        if let Ok(desc) = unsafe { output.GetDesc() } {
            self.screen_width = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left;
            self.screen_height = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;
        }

        self.device = Some(device);
        self.context = Some(context);
        self.output = Some(output);
        self.duplication = Some(duplication);
        true
    }

    /// Capture a single BGRA frame (or a cropped sub-region) from the desktop.
    ///
    /// The crop region `(x, y, w, h)` is in desktop pixels. `out_data` is
    /// resized to `w * h * 4` bytes of BGRA data on success.
    ///
    /// The 2-texture staging ring pipelines GPU->CPU readback: the GPU copies
    /// *this* frame into `staging[current]`, while the CPU maps `staging[prev]`
    /// (whose GPU copy finished last frame) for the actual memcpy. On the first
    /// frame the current slot is mapped directly (no pipelining yet).
    pub fn capture_frame(
        &mut self,
        out_data: &mut Vec<u8>,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        timeout_ms: u32,
    ) -> CaptureFrameResult {
        let mut result = CaptureFrameResult::default();

        let duplication = match self.duplication.as_ref() {
            Some(d) => d.clone(),
            None => {
                result.hr = windows::Win32::Foundation::E_POINTER.0;
                return result;
            }
        };
        let context = match self.context.as_ref() {
            Some(c) => c,
            None => {
                result.hr = windows::Win32::Foundation::E_POINTER.0;
                return result;
            }
        };
        let device = match self.device.as_ref() {
            Some(d) => d,
            None => {
                result.hr = windows::Win32::Foundation::E_POINTER.0;
                return result;
            }
        };

        // ---- AcquireNextFrame ----
        let mut frame_info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource: Option<IDXGIResource> = None;

        let acq_start = std::time::Instant::now();
        let hr = unsafe { duplication.AcquireNextFrame(timeout_ms, &mut frame_info, &mut resource) };

        if let Err(e) = hr {
            let code = e.code();
            if code == DXGI_ERROR_WAIT_TIMEOUT {
                result.status = CaptureStatus::Timeout;
                result.hr = code.0;
                return result;
            }
            if code == DXGI_ERROR_ACCESS_LOST {
                // Re-duplicate the output.
                self.duplication = None;
                if let Some(ref output) = self.output {
                    if let Ok(output1) = output.cast::<IDXGIOutput1>() {
                        self.duplication = unsafe { output1.DuplicateOutput(device) }.ok();
                    }
                }
                result.status = CaptureStatus::AccessLost;
                result.hr = code.0;
                return result;
            }
            result.hr = code.0;
            return result;
        }

        // Stamp capture time immediately after acquiring the frame.
        result.capture_unix_micros = now_unix_micros();

        let readback_start = std::time::Instant::now();
        result.acquire_us = readback_start.duration_since(acq_start).as_micros() as i64;

        // Get the desktop texture from the DXGI resource.
        let resource = match resource {
            Some(r) => r,
            None => {
                let _ = unsafe { duplication.ReleaseFrame() };
                return result;
            }
        };
        let texture: ID3D11Texture2D = match resource.cast() {
            Ok(t) => t,
            Err(e) => {
                let _ = unsafe { duplication.ReleaseFrame() };
                result.hr = e.code().0;
                return result;
            }
        };

        // ---- Create / recreate staging ring if dimensions changed ----
        if self.staging[0].is_none() || self.staging_width != w || self.staging_height != h {
            let staging_desc = D3D11_TEXTURE2D_DESC {
                Width: w as u32,
                Height: h as u32,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0u32,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0u32,
            };

            let mut tex0: Option<ID3D11Texture2D> = None;
            let mut tex1: Option<ID3D11Texture2D> = None;
            let _ = unsafe { device.CreateTexture2D(&staging_desc, None, Some(&mut tex0)) };
            let _ = unsafe { device.CreateTexture2D(&staging_desc, None, Some(&mut tex1)) };
            self.staging[0] = tex0;
            self.staging[1] = tex1;
            self.staging_width = w;
            self.staging_height = h;
            self.staging_index = 0;
            self.staging_ready = false;
        }

        let cur_staging = match self.staging[self.staging_index].as_ref() {
            Some(s) => s,
            None => {
                let _ = unsafe { duplication.ReleaseFrame() };
                return result;
            }
        };

        // ---- GPU copy: desktop texture -> staging[current] ----
        let src_box = D3D11_BOX {
            left: x as u32,
            top: y as u32,
            front: 0,
            right: (x + w) as u32,
            bottom: (y + h) as u32,
            back: 1,
        };
        unsafe {
            context.CopySubresourceRegion(cur_staging, 0, 0, 0, 0, &texture, 0, Some(&src_box));
        }

        // ---- Map the PREVIOUS staging texture (its copy finished last frame) ----
        let map_index = if self.staging_ready {
            1 - self.staging_index
        } else {
            self.staging_index
        };
        let map_staging = match self.staging[map_index].as_ref() {
            Some(s) => s,
            None => {
                let _ = unsafe { duplication.ReleaseFrame() };
                return result;
            }
        };

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        match unsafe { context.Map(map_staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) } {
            Ok(()) => {}
            Err(e) => {
                let _ = unsafe { duplication.ReleaseFrame() };
                result.hr = e.code().0;
                return result;
            }
        };

        // ---- CPU copy: mapped staging -> out_data ----
        let frame_bytes = (w as usize) * (h as usize) * 4;
        out_data.resize(frame_bytes, 0);

        unsafe {
            let src = mapped.pData as *const u8;
            let row_pitch = mapped.RowPitch as usize;
            let row_bytes = (w as usize) * 4;

            if row_pitch == row_bytes {
                // Contiguous -- single memcpy.
                std::ptr::copy_nonoverlapping(src, out_data.as_mut_ptr(), frame_bytes);
            } else {
                // Padded rows -- copy row by row.
                let dst = out_data.as_mut_ptr();
                for row in 0..h as usize {
                    std::ptr::copy_nonoverlapping(
                        src.add(row * row_pitch),
                        dst.add(row * row_bytes),
                        row_bytes,
                    );
                }
            }
        }

        unsafe { context.Unmap(map_staging, 0) };

        // Flip the staging ring index for the next frame.
        self.staging_index = 1 - self.staging_index;
        self.staging_ready = true;

        let _ = unsafe { duplication.ReleaseFrame() };

        result.readback_us = readback_start.elapsed().as_micros() as i64;
        result.status = CaptureStatus::Captured;
        result.success = true;
        result.hr = 0; // S_OK
        result
    }

    pub fn screen_width(&self) -> i32 {
        self.screen_width
    }

    pub fn screen_height(&self) -> i32 {
        self.screen_height
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a fixed-size `u16` wide-char array (null-terminated) to a Rust
/// `String`.
fn wchar_to_string(wide: &[u16]) -> String {
    let len = wide.iter().position(|&c| c == 0).unwrap_or(wide.len());
    String::from_utf16_lossy(&wide[..len])
}

/// Return the current wall-clock time as microseconds since the UNIX epoch.
fn now_unix_micros() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}
