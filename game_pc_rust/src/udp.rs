//! UDP packet structures and send/receive logic.
//!
//! All wire-format structs are `#[repr(C, packed)]` and byte-identical to the
//! C++ originals so the Jetson inference PC can decode them without changes.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use socket2::Socket;

// ---------------------------------------------------------------------------
// Wire-format constants
// ---------------------------------------------------------------------------

/// "GPA3" little-endian — v3 header with `capture_unix_micros`.
pub const UDP_PACKET_V3_MAGIC: u32 = 0x33415047;
pub const UDP_PIXEL_FORMAT_RGB: u8 = 2;
pub const UDP_PIXEL_FORMAT_BGRA: u8 = 3;

/// "GPAC" little-endian — credit grant from inference PC.
pub const UDP_CREDIT_MAGIC: u32 = 0x43504147;

/// "GPAP" little-endian — clock-sync ping from inference PC.
pub const UDP_SYNC_PING_MAGIC: u32 = 0x50415047;
/// "GPAO" little-endian — clock-sync pong reply to inference PC.
pub const UDP_SYNC_PONG_MAGIC: u32 = 0x4F415047;

pub const MIN_PACKET_PAYLOAD_BYTES: i32 = 512;
pub const MAX_PACKET_PAYLOAD_BYTES: i32 = 60000;

/// Largest application payload fitting a standard 1500-byte Ethernet MTU:
/// 1500 - 20 (IPv4) - 8 (UDP) - 44 (UDPPacketHeaderV2) = 1428.
pub const MTU_SAFE_PACKET_PAYLOAD_BYTES: i32 = 1500 - 20 - 8 - 44;

// ---------------------------------------------------------------------------
// Wire-format structs — must be byte-identical to C++ `#pragma pack(push, 1)`
// ---------------------------------------------------------------------------

/// Frame data packet header (v3: includes `capture_unix_micros`).
/// Size: 44 bytes.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct UDPPacketHeaderV2 {
    pub magic: u32,              // 0x33415047 "GPA3"
    pub header_size: u16,        // sizeof this struct (44)
    pub flags: u16,              // reserved, 0
    pub frame_id: u32,
    pub payload_offset: u32,     // byte offset of this chunk within the frame
    pub frame_bytes: u32,        // total frame size in bytes
    pub chunk_index: u16,
    pub total_chunks: u16,
    pub chunk_size: u32,         // payload bytes in this packet
    pub frame_width: u16,
    pub frame_height: u16,
    pub pixel_format: u8,
    pub bytes_per_pixel: u8,
    pub reserved: u16,
    pub capture_unix_micros: u64, // system_clock epoch us at capture
}

const _: () = assert!(
    std::mem::size_of::<UDPPacketHeaderV2>() == 44,
    "UDPPacketHeaderV2 must be exactly 44 bytes for wire compatibility"
);

/// Credit packet from the inference PC granting send permission.
/// Size: 24 bytes.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct UDPCreditPacket {
    pub magic: u32,
    pub size: u16,
    pub flags: u16,
    pub min_frame_id: u32,
    pub credits: u32,
    pub sequence: u64,
}

const _: () = assert!(
    std::mem::size_of::<UDPCreditPacket>() == 24,
    "UDPCreditPacket must be exactly 24 bytes for wire compatibility"
);

/// Clock-sync ping (inference PC -> game PC).
/// Size: 16 bytes.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct UDPSyncPing {
    pub magic: u32,
    pub seq: u32,
    pub t1: u64,
}

const _: () = assert!(
    std::mem::size_of::<UDPSyncPing>() == 16,
    "UDPSyncPing must be exactly 16 bytes for wire compatibility"
);

/// Clock-sync pong (game PC -> inference PC).
/// Size: 32 bytes.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct UDPSyncPong {
    pub magic: u32,
    pub seq: u32,
    pub t1: u64,
    pub t2: u64,
    pub t3: u64,
}

const _: () = assert!(
    std::mem::size_of::<UDPSyncPong>() == 32,
    "UDPSyncPong must be exactly 32 bytes for wire compatibility"
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Current system-clock time as microseconds since the Unix epoch.
/// Used for the `capture_unix_micros` field (requires NTP sync for E2E latency).
pub fn now_unix_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Safe helper to read a `#[repr(C, packed)]` struct from a byte buffer.
///
/// # Safety
/// The caller must ensure `buf.len() >= size_of::<T>()`.
unsafe fn read_packed<T: Copy>(buf: &[u8]) -> T {
    debug_assert!(buf.len() >= std::mem::size_of::<T>());
    let mut val = std::mem::MaybeUninit::<T>::uninit();
    std::ptr::copy_nonoverlapping(
        buf.as_ptr(),
        val.as_mut_ptr() as *mut u8,
        std::mem::size_of::<T>(),
    );
    val.assume_init()
}

/// Safe helper to write a `#[repr(C, packed)]` struct to a byte slice.
fn as_bytes<T: Copy>(val: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(val as *const T as *const u8, std::mem::size_of::<T>()) }
}

// ---------------------------------------------------------------------------
// Frame send
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct FrameSendResult {
    pub sent: bool,
    pub would_block: bool,
    pub total_packets: u16,
}

/// Fragment and send one frame over UDP.
///
/// Each chunk is prepended with a [`UDPPacketHeaderV2`] and sent as a single
/// datagram via `send_to`.  Returns immediately if the socket would block.
#[allow(clippy::too_many_arguments)]
pub fn send_frame_udp(
    socket: &Socket,
    dest: &SocketAddr,
    data: &[u8],
    frame_size: usize,
    frame_id: u32,
    width: u16,
    height: u16,
    pixel_format: u8,
    bpp: u8,
    max_payload: usize,
    capture_unix_micros: u64,
) -> FrameSendResult {
    let mut result = FrameSendResult::default();
    if data.is_empty() || frame_size == 0 || frame_size > u32::MAX as usize || max_payload == 0 {
        return result;
    }

    let total_packets = (frame_size + max_payload - 1) / max_payload;
    if total_packets == 0 || total_packets > u16::MAX as usize {
        return result;
    }
    result.total_packets = total_packets as u16;

    let dest_sa: socket2::SockAddr = (*dest).into();

    // Pre-allocate a send buffer large enough for the largest datagram.
    let header_size = std::mem::size_of::<UDPPacketHeaderV2>();
    let mut send_buf = vec![0u8; header_size + max_payload];

    for i in 0..total_packets {
        let offset = i * max_payload;
        let remaining = frame_size - offset;
        let payload_size = remaining.min(max_payload);

        let header = UDPPacketHeaderV2 {
            magic: UDP_PACKET_V3_MAGIC,
            header_size: header_size as u16,
            flags: 0,
            frame_id,
            payload_offset: offset as u32,
            frame_bytes: frame_size as u32,
            chunk_index: i as u16,
            total_chunks: result.total_packets,
            chunk_size: payload_size as u32,
            frame_width: width,
            frame_height: height,
            pixel_format,
            bytes_per_pixel: bpp,
            reserved: 0,
            capture_unix_micros,
        };

        // Build datagram: header bytes + payload bytes
        send_buf[..header_size].copy_from_slice(as_bytes(&header));
        send_buf[header_size..header_size + payload_size]
            .copy_from_slice(&data[offset..offset + payload_size]);

        let total_len = header_size + payload_size;
        match socket.send_to(&send_buf[..total_len], &dest_sa) {
            Ok(sent) => {
                if sent != total_len {
                    return result; // partial send
                }
            }
            Err(e) => {
                let kind = e.kind();
                result.would_block = kind == std::io::ErrorKind::WouldBlock;
                return result;
            }
        }
    }

    result.sent = true;
    result
}

// ---------------------------------------------------------------------------
// Credit / sync receive
// ---------------------------------------------------------------------------

/// Non-blocking read for credit packets and clock-sync pings.
///
/// - On a sync ping: immediately replies with a pong.
/// - On a credit packet: updates `frame_credits` and `min_credit_frame_id`.
/// - Returns `true` if a valid packet was processed (keep draining).
///
/// The condvar `notify` is called when credits arrive so the sender thread
/// can wake up.
pub fn read_frame_credit(
    socket: &Socket,
    frame_credits: &AtomicI32,
    min_credit_frame_id: &AtomicU32,
    credit_counter: &std::sync::atomic::AtomicU64,
    notify: &dyn Fn(),
) -> bool {
    let mut buf = [0u8; 64];

    // socket2 recv_from requires a MaybeUninit buffer.
    let (ret, from_addr) = match socket.recv_from(unsafe {
        // SAFETY: buf is a valid, fully-initialized [u8; 64] reinterpreted as
        // [MaybeUninit<u8>; 64]; the layout is identical.
        &mut *(buf.as_mut_slice() as *mut [u8] as *mut [std::mem::MaybeUninit<u8>])
    }) {
        Ok((n, addr)) => (n, addr),
        Err(_) => return false,
    };

    if ret < 4 {
        return false;
    }

    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);

    // Clock-sync ping -> reply with pong
    if magic == UDP_SYNC_PING_MAGIC && ret >= std::mem::size_of::<UDPSyncPing>() {
        let t2 = now_unix_micros();
        let ping: UDPSyncPing = unsafe { read_packed(&buf) };
        let pong = UDPSyncPong {
            magic: UDP_SYNC_PONG_MAGIC,
            seq: ping.seq,
            t1: ping.t1,
            t2,
            t3: now_unix_micros(),
        };
        let pong_bytes = as_bytes(&pong);
        // socket2::Socket::send_to takes &[u8] for connected/unconnected sends.
        let _ = socket.send_to(pong_bytes, &from_addr);
        return true; // keep draining
    }

    // Credit packet
    if ret != std::mem::size_of::<UDPCreditPacket>() {
        return false;
    }
    let packet: UDPCreditPacket = unsafe { read_packed(&buf) };
    if packet.magic != UDP_CREDIT_MAGIC
        || packet.size != std::mem::size_of::<UDPCreditPacket>() as u16
        || packet.credits == 0
    {
        return false;
    }

    min_credit_frame_id.store(packet.min_frame_id, Ordering::Release);
    let requested = (packet.credits as i32).clamp(1, 8);
    // CAS loop: only increase credits, never decrease.
    let mut current = frame_credits.load(Ordering::Acquire);
    while current < requested {
        match frame_credits.compare_exchange_weak(
            current,
            requested,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => break,
            Err(c) => current = c,
        }
    }

    credit_counter.fetch_add(1, Ordering::Relaxed);
    notify();
    true
}

/// Warn once if the configured payload exceeds the MTU-safe limit.
pub fn warn_if_payload_exceeds_mtu(bytes: i32) {
    if bytes <= MTU_SAFE_PACKET_PAYLOAD_BYTES {
        return;
    }
    eprintln!(
        "[Warning] PacketPayloadBytes={} exceeds the MTU-safe {}; each chunk will be \
         IP-fragmented and a single lost fragment will drop the whole chunk. Use jumbo \
         frames or lower it.",
        bytes, MTU_SAFE_PACKET_PAYLOAD_BYTES
    );
}
