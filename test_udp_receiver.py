import socket

# UDP 수신 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5006))

print("Listening on port 5006 for button state broadcasts...")
print("Press Ctrl+C to exit\n")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        message = data.decode('utf-8').strip()
        print(f"Received from {addr}: {message}")

        # Parse STATE message
        if message.startswith("STATE:"):
            parts = message[6:].split(',')
            if len(parts) == 2:
                left_mouse = parts[0] == '1'
                right_mouse = parts[1] == '1'
                print(f"  → Left: {left_mouse}, Right: {right_mouse}")

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    sock.close()
