// Makcu Mouse Test - Move mouse in a square pattern
// Usage: test_makcu_mouse [port] [size]

#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

#include "needaimbot/mouse/input_drivers/MakcuConnection.h"

static std::atomic<bool> g_running{true};

void signalHandler(int sig) {
    std::cout << "\nStopping..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    std::string port = "/dev/ttyACM0";
    int size = 100;  // Square size in pixels
    int speed = 5;   // Pixels per step

    if (argc > 1) port = argv[1];
    if (argc > 2) size = std::stoi(argv[2]);
    if (argc > 3) speed = std::stoi(argv[3]);

    std::cout << "=== Makcu Mouse Square Test ===" << std::endl;
    std::cout << "Port: " << port << std::endl;
    std::cout << "Square size: " << size << " pixels" << std::endl;
    std::cout << "Speed: " << speed << " pixels/step" << std::endl;
    std::cout << "Press Ctrl+C to stop\n" << std::endl;

    std::signal(SIGINT, signalHandler);

    // Connect to Makcu
    MakcuConnection makcu(port, 4000000);

    if (!makcu.isOpen()) {
        std::cerr << "Failed to connect to Makcu on " << port << std::endl;
        return 1;
    }

    std::cout << "Connected! Testing simple move..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Test 1: Single large move
    std::cout << "Test 1: Single move(50, 0)..." << std::endl;
    makcu.move(50, 0);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "Test 2: Single move(0, 50)..." << std::endl;
    makcu.move(0, 50);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "Test 3: Continuous moves RIGHT..." << std::endl;
    for (int i = 0; i < 20 && g_running; i++) {
        makcu.move(10, 0);
        std::cout << "  sent M10,0" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "Test 4: Continuous moves DOWN..." << std::endl;
    for (int i = 0; i < 20 && g_running; i++) {
        makcu.move(0, 10);
        std::cout << "  sent M0,10" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "Tests complete!" << std::endl;

    std::cout << "Done." << std::endl;
    return 0;
}
