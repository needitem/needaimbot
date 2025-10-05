import os
import stat
import ctypes
import sys

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    print("This script requires administrator privileges.")
    response = input("Do you want to restart with admin rights? (Y/N): ").strip().lower()
    if response == 'y':
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        sys.exit(0)
    else:
        print("Exiting... Script will not work without admin privileges.")
        sys.exit(1)

print("Arduino Spoof Restore Script")
print("This will restore Arduino to default Leonardo settings")
print()

# Arduino boards.txt 경로
boards_paths = [
    r"C:\Users\th072\AppData\Local\Arduino15\packages\arduino\hardware\avr\1.8.6\boards.txt",
    r"C:\Program Files (x86)\Arduino\hardware\arduino\avr\boards.txt"
]

boards_path = None
for path in boards_paths:
    if os.path.exists(path):
        boards_path = path
        print(f"Found boards.txt at: {path}")
        break

if not boards_path:
    raise FileNotFoundError(f"boards.txt not found at any of: {boards_paths}")

# 기본 Leonardo 설정
default_vid = "0x2341"
default_pid = "0x8036"
default_name = "Arduino Leonardo"
default_manufacturer = "Arduino LLC"

print(f"\nRestoring to default Leonardo settings:")
print(f"  VID: {default_vid}")
print(f"  PID: {default_pid}")
print(f"  Name: {default_name}")
print(f"  Manufacturer: {default_manufacturer}")
print(f"  COM Port: ENABLED")
print()

# boards.txt 수정
os.chmod(boards_path, stat.S_IWRITE)

with open(boards_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

new_lines = []
for i, line in enumerate(lines):
    if line.startswith("leonardo.name="):
        new_lines.append(f"leonardo.name={default_name}\n")
    elif line.startswith("leonardo.vid.") or line.startswith("leonardo.build.vid="):
        new_lines.append(line.split("=")[0] + f"={default_vid}\n")
    elif line.startswith("leonardo.pid.") or line.startswith("leonardo.build.pid="):
        new_lines.append(line.split("=")[0] + f"={default_pid}\n")
    elif line.startswith("leonardo.build.usb_product="):
        new_lines.append(f'leonardo.build.usb_product="{default_name}"\n')
    elif line.startswith("leonardo.build.usb_manufacturer="):
        new_lines.append(f'leonardo.build.usb_manufacturer="{default_manufacturer}"\n')
    elif line.startswith("leonardo.build.extra_flags="):
        # COM 포트 활성화 (CDC_DISABLED 제거)
        import re
        new_line = re.sub(r"\s*-DCDC_DISABLED", "", line)
        new_lines.append(new_line)
    else:
        new_lines.append(line)

with open(boards_path, "w", encoding="utf-8") as file:
    file.writelines(new_lines)

os.chmod(boards_path, stat.S_IREAD)
print("✓ boards.txt restored to default")

# USBCore.cpp 복원
usbcore_paths = [
    r"C:\Users\th072\AppData\Local\Arduino15\packages\arduino\hardware\avr\1.8.6\cores\arduino\USBCore.cpp",
    r"C:\Program Files (x86)\Arduino\hardware\arduino\avr\cores\arduino\USBCore.cpp"
]

usbcore_path = None
for path in usbcore_paths:
    if os.path.exists(path):
        usbcore_path = path
        print(f"Found USBCore.cpp at: {path}")
        break

if not usbcore_path:
    raise FileNotFoundError(f"USBCore.cpp not found at any of: {usbcore_paths}")

with open(usbcore_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Default Leonardo descriptor 복원
restored_lines = []
in_descriptor = False
for line in lines:
    if "const DeviceDescriptor USB_DeviceDescriptorIAD =" in line:
        restored_lines.append(line)
        in_descriptor = True
    elif in_descriptor and "D_DEVICE(" in line:
        # 기본 Leonardo descriptor로 복원
        restored_lines.append(f'\tD_DEVICE(0xEF,0x02,0x01,64,USB_VID,USB_PID,0x100,IMANUFACTURER,IPRODUCT,ISERIAL,1);\n')
        in_descriptor = False
    else:
        restored_lines.append(line)

with open(usbcore_path, "w", encoding="utf-8") as file:
    file.writelines(restored_lines)

print("✓ USBCore.cpp device descriptor restored to default")

# USBCore.h 전원 설정 복원
usbcoreh_paths = [
    r"C:\Users\th072\AppData\Local\Arduino15\packages\arduino\hardware\avr\1.8.6\cores\arduino\USBCore.h",
    r"C:\Program Files (x86)\Arduino\hardware\arduino\avr\cores\arduino\USBCore.h"
]

usbcoreh_path = None
for path in usbcoreh_paths:
    if os.path.exists(path):
        usbcoreh_path = path
        print(f"Found USBCore.h at: {path}")
        break

if not usbcoreh_path:
    raise FileNotFoundError(f"USBCore.h not found at any of: {usbcoreh_paths}")

with open(usbcoreh_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

updated_lines = []
inside_ifndef = False
default_power = 500  # Arduino Leonardo 기본값

for i, line in enumerate(lines):
    stripped = line.strip()

    if stripped.startswith("#ifndef USB_CONFIG_POWER"):
        inside_ifndef = True
        updated_lines.append(line)
        continue

    if inside_ifndef and stripped.startswith("#define USB_CONFIG_POWER"):
        updated_lines.append(f" #define USB_CONFIG_POWER                      ({default_power})\n")
        inside_ifndef = False
        continue

    updated_lines.append(line)

with open(usbcoreh_path, "w", encoding="utf-8") as file:
    file.writelines(updated_lines)

print(f"✓ USBCore.h power limit restored to {default_power}mA")

print("\n" + "="*60)
print("✓ Arduino successfully restored to default Leonardo settings!")
print("="*60)
print("\nNext steps:")
print("1. Open Arduino IDE")
print("2. Upload your sketch again")
print("3. The Arduino will appear as 'Arduino Leonardo' with default VID/PID")
print("4. COM port will be enabled for Serial communication")
print()
