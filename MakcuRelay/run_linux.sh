#!/bin/bash

# MakcuRelay Linux Run Helper Script

EXECUTABLE="./MakcuRelay"
BUILD_SCRIPT="./build_linux.sh"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable not found. Building..."
    if [ -f "$BUILD_SCRIPT" ]; then
        bash "$BUILD_SCRIPT"
    else
        echo "Error: Build script $BUILD_SCRIPT not found!"
        exit 1
    fi
fi

# Function to list serial ports
list_ports() {
    echo "Available Serial Ports:"
    echo "-----------------------"
    ports=$(ls /dev/ttyUSB* /dev/ttyACM* /dev/ttyS* 2>/dev/null)
    if [ -z "$ports" ]; then
        echo "No serial ports found!"
        echo "Please check if your device is connected."
    else
        for port in $ports; do
            echo "  $port"
        done
    fi
    echo "-----------------------"
}

# Check arguments
# Check arguments
if [ $# -ge 2 ]; then
    SERIAL_PORT=$1
    UDP_PORT=$2
else
    echo "Scanning for serial ports..."
    # Use positional parameters to store ports (POSIX compliant)
    set -- $(ls /dev/ttyUSB* /dev/ttyACM* /dev/ttyS* 2>/dev/null)
    
    if [ $# -eq 0 ]; then
        echo "No serial ports found!"
        echo "Please check connection and try again."
        exit 1
    fi

    echo "Available Serial Ports:"
    i=0
    for port in "$@"; do
        echo "[$i] $port"
        i=$((i + 1))
    done
    echo ""

    printf "Select port number [0]: "
    read port_index
    port_index=${port_index:-0}

    # Check if input is a number
    case "$port_index" in
        ''|*[!0-9]*) 
            echo "Invalid selection."
            exit 1 
            ;;
    esac

    if [ "$port_index" -ge "$#" ]; then
        echo "Invalid selection."
        exit 1
    fi

    # Retrieve selected port
    i=0
    for port in "$@"; do
        if [ "$i" -eq "$port_index" ]; then
            SERIAL_PORT=$port
            break
        fi
        i=$((i + 1))
    done
    
    printf "Enter UDP Port [5005]: "
    read UDP_PORT
    UDP_PORT=${UDP_PORT:-5005}
fi

echo ""
echo "Selected Serial Port: $SERIAL_PORT"
echo "Selected UDP Port:    $UDP_PORT"
echo ""

# Check if port exists
if [ ! -e "$SERIAL_PORT" ]; then
    echo "Error: Serial port $SERIAL_PORT does not exist."
    echo ""
    list_ports
    exit 1
fi

# Check permissions
if [ ! -r "$SERIAL_PORT" ] || [ ! -w "$SERIAL_PORT" ]; then
    echo "Warning: You may not have permission to access $SERIAL_PORT."
    echo "Try running: sudo usermod -a -G dialout $USER"
    echo "Then log out and log back in."
    echo "Or run with sudo (not recommended for regular use)."
    echo ""
    printf "Attempt to run anyway? (y/n) "
    read REPLY
    echo ""
    case "$REPLY" in
        [Yy]*) ;;
        *) exit 1 ;;
    esac
fi

# Run the application
echo "Starting MakcuRelay..."
$EXECUTABLE "$SERIAL_PORT" "$UDP_PORT"
