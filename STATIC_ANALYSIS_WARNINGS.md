# Static Analysis Warnings (CppCoreCheckRules)

Generated: 2026-01-06
Updated: 2026-01-06 - ALL WARNINGS FIXED ✅

## Summary
- Total warnings fixed: ~30
- Rule set: CppCoreCheckRules.ruleset
- Build status: SUCCESS (no warnings)

---

## Fixed Issues

### needaimbot.cpp
- ✅ Line 60: Removed unused variable 'ctx'
- ✅ Line 545: Removed unused variable 'numCores'

### overlay/draw_ai.cpp
- ✅ Line 344: Removed unused variable 'ctx'

### overlay/draw_buttons.cpp
- ✅ Line 44: Removed unused function 'detect_pressed_key'
- ✅ Line 72: Removed unused variable 'cursor_x'

### overlay/draw_capture.cpp
- ✅ Line 211: Removed unused variable 'ctx'

### overlay/draw_debug.cpp
- ✅ Line 47-48: Removed unused variables (g_crosshairH/S/V, g_crosshairHsvValid)

### overlay/draw_overlay.cpp
- ✅ Line 21, 37: Removed unused 'console_state_changed'

### overlay/draw_profile.cpp
- ✅ Line 188: Removed unused variable 'is_overwriting'

### overlay/overlay.cpp
- ✅ Line 447: Removed unused variable 'config'
- ✅ Line 532-540: Removed unused 'input_method_index'
- ✅ Line 553: Removed unused variable 'lastTime'

### overlay/draw_color_filter.cpp
- ✅ Line 240: Changed stride type from int to size_t
- ✅ Line 362: Removed unused exception variable 'e'

### scr/other_tools.cpp
- ✅ Line 165-166: Added parentheses around '&&' expressions

### capture/dda_capture.cpp
- ✅ Line 11: Removed unused 'kBytesPerPixel'

### mouse/input_drivers/kmboxNet.cpp
- ✅ Line 28-30: Removed unused variables (monitor_port, monitor_run, mask_keyboard_mouse_flag)
- ✅ Multiple lines: Removed unused 'err' variables, used (void) cast for ignored return values
- ✅ Line 144, 188, 346: Added double braces for struct initialization

### mouse/input_drivers/SerialConnection.cpp
- ✅ Line 43: Fixed field initialization order
- ✅ Line 354: Removed unused lambda capture 'this'

### mouse/input_drivers/MakcuConnection.cpp
- ✅ Line 26: Fixed field initialization order

### mouse/input_drivers/my_enc.cpp
- ✅ Line 5: Removed unused initial value for 'a4'
- ✅ Line 14, 16: Added parentheses around '&' in '^' expressions

### cuda/preprocessing.cu
- ✅ Line 137: Added static_cast<int> for size_t to int conversion

### cuda/cuda_error_check.h
- ✅ Line 9: Added #ifndef guard for CUDA_CHECK macro

### cuda/unified_graph_pipeline.cu
- ✅ Line 492: Added 20054 to nvcc --diag-suppress for __shared__ variable warning
