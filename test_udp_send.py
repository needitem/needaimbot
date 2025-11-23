import socket
import time

# UDP 전송 테스트
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Sending test UDP packets to 127.0.0.1:5006...")
for i in range(5):
    left = i % 2
    right = (i + 1) % 2
    msg = f"STATE:{left},{right}\n"
    sock.sendto(msg.encode('utf-8'), ('127.0.0.1', 5006))
    print(f"Sent: {msg.strip()}")
    time.sleep(1)

sock.close()
print("Done")
