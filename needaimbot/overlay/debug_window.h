#ifndef DEBUG_WINDOW_H
#define DEBUG_WINDOW_H

#include <atomic>
#include <thread>
#include <d3d11.h>

// Standalone debug overlay window
// Displays detection bounding boxes and confidence on a transparent fullscreen overlay
namespace DebugOverlay {

// Initialize and start the debug overlay thread
void Start();

// Stop and cleanup the debug overlay
void Stop();

// Toggle visibility
void Toggle();

// Check if visible
bool IsVisible();

// Set visibility directly
void SetVisible(bool visible);

// Check if running
bool IsRunning();

} // namespace DebugOverlay

#endif // DEBUG_WINDOW_H
