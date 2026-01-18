#pragma once

// Delayed config save utility for overlay
// Prevents excessive disk I/O when user is rapidly changing settings

void OverlayConfig_MarkDirty();
void OverlayConfig_TrySave(const char* filename = nullptr);
