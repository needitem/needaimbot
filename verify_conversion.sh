#!/bin/bash

echo "=== Verifying Blacklist to Whitelist Conversion ==="
echo ""

# Check for any remaining "ignore" references in critical files
echo "1. Checking for remaining 'ignore' references in critical files:"
echo ""

echo "Config files:"
if grep -r "ignore" needaimbot/config/ --exclude="*.md" --exclude="*.sh" | grep -i class; then
    echo "❌ Found remaining class ignore references in config files"
else
    echo "✅ No class ignore references found in config files"
fi

echo ""
echo "Detector files:"
if grep -r "ignore.*flag" needaimbot/detector/ --exclude="*.md" --exclude="*.sh"; then
    echo "❌ Found remaining ignore flag references in detector files"
else
    echo "✅ No ignore flag references found in detector files"
fi

echo ""
echo "GPU files:"
if grep -r "ignored_class_ids" needaimbot/postprocess/ --exclude="*.md" --exclude="*.sh"; then
    echo "❌ Found remaining ignored_class_ids references"
else
    echo "✅ No ignored_class_ids references found"
fi

echo ""
echo "UI files:"
if grep -r "Ignore.*##" needaimbot/overlay/ --exclude="*.md" --exclude="*.sh"; then
    echo "❌ Found remaining Ignore UI references"
else
    echo "✅ No Ignore UI references found"
fi

echo ""
echo "2. Checking INI files conversion:"
cd x64/Release
if ls *.ini 2>/dev/null | xargs grep -l "Class_.*_Ignore" 2>/dev/null; then
    echo "❌ Found INI files still using _Ignore"
else
    echo "✅ All INI files converted to _Allow"
fi

echo ""
echo "3. Checking logic consistency:"
# Check that mouse.cpp uses .allow instead of .ignore
if grep -q "class_setting\.allow" ../../needaimbot/mouse/mouse.cpp; then
    echo "✅ Mouse.cpp uses .allow field"
else
    echo "❌ Mouse.cpp still uses .ignore field"
fi

# Check detector uses allow flags
if grep -q "m_host_allow_flags" ../../needaimbot/detector/detector.cpp; then
    echo "✅ Detector uses allow flags"
else
    echo "❌ Detector still uses ignore flags"
fi

echo ""
echo "4. Checking GPU kernel logic:"
# Check that GPU kernels use inverted logic (skip when NOT allowed)
if grep -q "!d_allowed_class_ids\[" ../../needaimbot/postprocess/postProcessGpu.cu; then
    echo "✅ GPU kernels use whitelist logic (skip when not allowed)"
else
    echo "❌ GPU kernels may still use blacklist logic"
fi

echo ""
echo "=== Conversion Verification Complete ==="