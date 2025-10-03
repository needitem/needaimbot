import hid
import os
import stat
import time
import os
import re
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

print("Seconb's Basic Arduino Spoofer")
print("This isn't perfect, and won't make your Arduino 1:1 to a real mouse, but will work on MOST games. Don't use on main PC for full safety.")
print("Report any issues to Seconb on UnknownCheats")
 
time.sleep(3)
 
n = 1
for device in hid.enumerate():
    if device['usage_page'] == 0x01 and device['usage'] == 0x02:
        print(f"{n}. Name: {device['product_string']}")
        print(f"{n}. Manufacturer: {device['manufacturer_string']}")
        print(f"{n}. VID: 0x{device['vendor_id']:04X}")
        print(f"{n}. PID: 0x{device['product_id']:04X}")
        n += 1
 
print("Enter the mouse info to spoof your Arduino to:")
new_vid = input("New VID (e.g. 0x1234): ").lower()
new_pid = input("New PID (e.g. 0x5678): ").lower()
new_name = input("New Device Name: ")
new_manufacturer = input("New Manufacturer: ")
disable_com = input("Disable COM port? (Y or N) (N reenables it if you turned it off before): ").strip().lower()
spoof_descriptor = input("Spoof device descriptor to HID Mouse? (Y or N): ").strip().lower()
 
spoof_power = input("Do you want to spoof the power usage? (Y or N): ").strip().lower()
if spoof_power == "y":
    custom_power_limit = input("Enter the custom power limit (100 recommended): ").strip()
    try:
        custom_power_limit = int(custom_power_limit)
    except ValueError:
        custom_power_limit = 100
else:
    custom_power_limit = None
 
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
 
os.chmod(boards_path, stat.S_IWRITE)
 
with open(boards_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
 
new_lines = []
usb_mfr_added = False
for i, line in enumerate(lines):
    if line.startswith("leonardo.name="):
        new_lines.append(f"leonardo.name={new_name}\n")
    elif line.startswith("leonardo.vid.") or line.startswith("leonardo.build.vid="):
        new_lines.append(line.split("=")[0] + f"={new_vid}\n")
    elif line.startswith("leonardo.pid.") or line.startswith("leonardo.build.pid="):
        new_lines.append(line.split("=")[0] + f"={new_pid}\n")
    elif line.startswith("leonardo.build.usb_product="):
        new_lines.append(f'leonardo.build.usb_product="{new_name}"\n')
    elif line.startswith("leonardo.build.extra_flags="):
        if disable_com == "y":
            if "-DCDC_DISABLED" not in line:
                new_line = line.strip() + " -DCDC_DISABLED\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_line = re.sub(r"\s*-DCDC_DISABLED", "", line)
            new_lines.append(new_line)
        if not any("leonardo.build.usb_manufacturer=" in l for l in lines):
            new_lines.append(f'leonardo.build.usb_manufacturer="{new_manufacturer}"\n')
            usb_mfr_added = True
    else:
        new_lines.append(line)
 
with open(boards_path, "w", encoding="utf-8") as file:
    file.writelines(new_lines)
 
os.chmod(boards_path, stat.S_IREAD)
print("\n boards.txt updated")
 
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
 
if spoof_descriptor == "y":
    spoofed_lines = []
    in_descriptor = False
    for line in lines:
        if "const DeviceDescriptor USB_DeviceDescriptorIAD =" in line:
            spoofed_lines.append(line)
            in_descriptor = True
        elif in_descriptor and "D_DEVICE(" in line:
            spoofed_lines.append(f'\tD_DEVICE(0x00, 0x00, 0x00, 64,{new_vid.upper()},{new_pid.upper()},0x100,IMANUFACTURER,IPRODUCT,ISERIAL,1);\n')
            in_descriptor = False
        else:
            spoofed_lines.append(line)
 
    with open(usbcore_path, "w", encoding="utf-8") as file:
        file.writelines(spoofed_lines)
 
    print(" Device descriptor spoofed to HID Mouse.")
else:
    print(" Descriptor spoofing skipped.")
 
if spoof_power == "y":
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
 
    for i, line in enumerate(lines):
        stripped = line.strip()
 
        if stripped.startswith("#ifndef USB_CONFIG_POWER"):
            inside_ifndef = True
            updated_lines.append(line)
            continue
 
        if inside_ifndef and stripped.startswith("#define USB_CONFIG_POWER"):
            updated_lines.append(f" #define USB_CONFIG_POWER                      ({custom_power_limit})\n")
            inside_ifndef = False
            continue
 
        updated_lines.append(line)
 
    with open(usbcoreh_path, "w", encoding="utf-8") as file:
        file.writelines(updated_lines)
 
    print(f" Power limit spoofed to {custom_power_limit}.")
 
print(" COM port status handled via boards.txt.")
print(" Now upload any script to your Arduino and the device will appear spoofed.")
