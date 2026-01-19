import socket
import time

UDP_IP = "192.168.1.124"
UDP_PORT = 5005

print(f"Sending square pattern to {UDP_IP}:{UDP_PORT}...")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def move(x, y):
    msg = f"MOVE:{x},{y}"
    sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))
    time.sleep(0.01) # Small delay between packets

try:
    # Draw a square
    side_length = 100
    step_size = 5
    
    print("Drawing square...")
    
    # Right
    for _ in range(0, side_length, step_size):
        move(step_size, 0)
        
    # Down
    for _ in range(0, side_length, step_size):
        move(0, step_size)
        
    # Left
    for _ in range(0, side_length, step_size):
        move(-step_size, 0)
        
    # Up
    for _ in range(0, side_length, step_size):
        move(0, -step_size)
        
    print("Done!")

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    sock.close()
