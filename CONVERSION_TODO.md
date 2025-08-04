# Blacklist to Whitelist Conversion TODO

## Core Structure Changes
- [x] Analyze current system architecture
- [x] Update ClassSetting struct - replace 'ignore' with 'allow'
- [x] Update config reading/writing logic
- [x] Update detector variables and logic
- [x] Update GPU kernel logic
- [x] Update UI text and logic
- [x] Update all INI files
- [x] Update mouse.cpp logic

## Files to Modify
1. needaimbot/config/config.h - ClassSetting struct
2. needaimbot/config/config.cpp - config reading/writing
3. needaimbot/detector/detector.h - variable names
4. needaimbot/detector/detector.cpp - logic updates
5. needaimbot/postprocess/postProcessGpu.cu - kernel logic
6. needaimbot/postprocess/filterGpu.h - function signatures
7. needaimbot/postprocess/filterGpu.cu - filter logic
8. needaimbot/overlay/draw_ai.cpp - UI updates
9. needaimbot/overlay/draw_debug.cpp - debug UI updates
10. needaimbot/mouse/mouse.cpp - filtering logic
11. All INI files in x64/Release/

## Logic Changes Required
- GPU kernels: Change from "skip if ignored" to "skip if not allowed"
- Host logic: Invert boolean checks
- Default values: Change from "ignore=false" (allow by default) to "allow=true" (allow by default)
- Variable names: m_host_ignore_flags -> m_host_allow_flags
- Config keys: Class_X_Ignore -> Class_X_Allow