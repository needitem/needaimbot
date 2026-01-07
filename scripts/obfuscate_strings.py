#!/usr/bin/env python3
"""
문자열 난독화 스크립트 (안전한 버전)
- 민감한 문자열 리터럴만 치환
- 심볼(변수명, 함수명)은 건드리지 않음
- 빌드 안정성 보장
"""

import os
import re
import shutil
import json
from pathlib import Path
from typing import Dict, Set

# 난독화 설정
CONFIG = {
    "source_dir": "needaimbot",
    "output_dir": "needaimbot_obf",
    "extensions": [".cpp", ".h", ".cu", ".hpp", ".cuh"],
}

# =============================================================================
# 민감한 문자열 치환 맵 (문자열 리터럴 내에서만 적용)
# =============================================================================
STRING_REPLACEMENTS = {
    # AI/ML 관련
    "YOLO": "GPNN",
    "yolo": "gpnn",
    "YOLOv": "GPv",
    "Yolo": "Gpnn",
    "detection": "analysis",
    "Detection": "Analysis",
    "DETECTION": "ANALYSIS",
    "detect": "analyze",
    "Detect": "Analyze",
    "TensorRT": "ComputeRT",
    "tensorrt": "computert",
    "inference": "processing",
    "Inference": "Processing",
    "neural": "compute",
    "Neural": "Compute",
    "model": "module",
    "Model": "Module",
    ".onnx": ".data",
    ".engine": ".cache",
    
    # 에이밍 관련 - Config 키 포함
    "aimbot": "assistant",
    "Aimbot": "Assistant",
    "AIMBOT": "ASSISTANT",
    "auto_aim": "auto_fcs",
    "ignore_up_aim": "ignore_up_fcs",
    "enable_aim_shoot_offset": "enable_fcs_fire_offset",
    "aim_shoot_offset_x": "fcs_fire_offset_x",
    "aim_shoot_offset_y": "fcs_fire_offset_y",
    "disable_upward_aim": "disable_upward_fcs",
    "aim_": "fcs_",
    "Aim_": "Fcs_",
    "_aim": "_fcs",
    "aiming": "focusing",
    "Aiming": "Focusing",
    " aim": " focus",
    "Aim ": "Focus ",
    "aim": "fcs",
    "target": "point",
    "Target": "Point",
    "TARGET": "POINT",
    "headshot": "priority",
    "Headshot": "Priority",
    "recoil": "stability",
    "Recoil": "Stability",
    "tracking": "following",
    "Tracking": "Following",
    "triggerbot": "autoclick",
    
    # 마우스/입력 관련 - Config 키 포함
    "mouse_move": "input_delta",
    "MouseMove": "InputDelta",
    "MouseMovement": "InputDelta",
    "mouseMovement": "inputDelta",
    "mouse move": "input delta",
    "mouse": "pointer",
    "Mouse": "Pointer",
    "MOUSE": "POINTER",
    "kmbox_ip": "dev_addr",
    "kmbox_port": "dev_port",
    "kmbox_mac": "dev_hwid",
    "kmbox": "extdev",
    "KMBox": "ExtDev",
    "KMBOX": "EXTDEV",
    "kmboxNet": "extdevNet",
    "KmboxInputMethod": "ExtDevInputMethod",
    "ghub_mouse": "drv_input",
    "GHubInputMethod": "DrvInputMethod",
    "ghub": "drv",
    "Ghub": "Drv",
    "GHUB": "DRV",
    "logitech": "peripheral",
    "Logitech": "Peripheral",
    "arduino_baudrate": "mcu_baud",
    "arduino_port": "mcu_port",
    "arduino_enable_keys": "mcu_enable_keys",
    "arduino": "mcu",
    "Arduino": "MCU",
    "ARDUINO": "MCU",
    "SerialConnection": "CommLink",
    "SerialInputMethod": "CommInputMethod",
    "Serial": "Comm",
    "serial": "comm",
    "SERIAL": "COMM",
    "SendInput": "DispatchInput",
    
    # 화면 캡처 관련 - Config 키 및 에러 메시지 포함
    "screen capture": "display acquire",
    "Screen Capture": "Display Acquire",
    "screenshot_button": "snapshot_btn",
    "screenshot_delay": "snapshot_delay",
    "screenshot": "snapshot",
    "Screenshot": "Snapshot",
    "Desktop Duplication": "Display Mirror",
    "desktop duplication": "display mirror",
    "DuplicateOutput": "MirrorOutput",
    "IDXGIDevice": "IDisplayDevice",
    "IDXGIOutput": "IDisplayOutput",
    "DDACapture": "DDAAcquire",
    "capture_borders": "acquire_borders",
    "capture_cursor": "acquire_cursor",
    "capture_method": "acquire_method",
    "capture_timeout_scale": "acquire_timeout_scale",
    "capture": "acquire",
    "Capture": "Acquire",
    "CAPTURE": "ACQUIRE",
    "screen": "display",
    "Screen": "Display",
    "SCREEN": "DISPLAY",
    
    # 오버레이 관련 - Config 키 포함
    "button_open_overlay": "button_open_layer",
    "overlay_opacity": "layer_opacity",
    "overlay_ui_scale": "layer_ui_scale",
    "overlay": "layer",
    "Overlay": "Layer",
    "OVERLAY": "LAYER",
    "esp": "visual",
    "ESP": "VISUAL",
    "wallhack": "enhancement",
    "cheat": "tool",
    "Cheat": "Tool",
    "hack": "mod",
    "Hack": "Mod",
    
    # 게임 관련
    "enemy": "object",
    "Enemy": "Object",
    "ENEMY": "OBJECT",
    "player": "entity",
    "Player": "Entity",
    "hitbox": "region",
    "Hitbox": "Region",
    "bounding box": "enclosing rect",
    "BoundingBox": "EnclosingRect",
    
    # Config 키 - detection 관련
    "detection_resolution": "analysis_resolution",
    "max_detections": "max_results",
    
    # 프로젝트 특정
    "needaimbot": "nvcontainer",
    "NeedAimbot": "NVContainer",
    "NEEDAIMBOT": "NVCONTAINER",
    "Need Aimbot": "NV Container",
}


class StringObfuscator:
    def __init__(self, config: dict):
        self.config = config
        self.stats = {"files": 0, "replacements": 0}
        
        # 긴 문자열부터 치환하도록 정렬
        self.sorted_replacements = sorted(
            STRING_REPLACEMENTS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
    
    # 코드 레벨에서 치환할 식별자 (CUDA 커널 함수명, 구조체명 등)
    CODE_LEVEL_REPLACEMENTS = {
        # CUDA 커널 함수명
        "decodeYolo11GpuKernel": "decodeGP11GpuKernel",
        "decodeYolo10GpuKernel": "decodeGP10GpuKernel",
        "decodeAndFilterYolo11Kernel": "decodeAndFilterGP11Kernel",
        "decodeAndFilterYolo10Kernel": "decodeAndFilterGP10Kernel",
        "decodeYolo11Gpu": "decodeGP11Gpu",
        "decodeYolo10Gpu": "decodeGP10Gpu",
        "decodeYoloOutput": "decodeGPOutput",
        "fusedTargetSelectionAndMovementKernel": "fusedPointSelectionAndDeltaKernel",
        "batchedDetectionClearKernel": "batchedAnalysisClearKernel",
        "executeMouseMovement": "executeInputDelta",
        "filterMouseMovement": "filterInputDelta",
        "configureMouseMovementBuffer": "configureInputDeltaBuffer",
        # 구조체/클래스명
        "MouseMovement": "InputDelta",
        "mouseMovement": "inputDelta",
        "m_mouseMovementUsesMappedMemory": "m_inputDeltaUsesMappedMemory",
        # 클래스명 (RTTI 제거용)
        "SerialConnection": "CommLink",
        "SerialInputMethod": "CommInputMethod",
        "KmboxInputMethod": "ExtDevInputMethod",
        "GHubInputMethod": "DrvInputMethod",
        # DDA 캡처 관련
        "DDACapture": "DDAAcquire",
        "DDACaptureAdapter": "DDAAcquireAdapter",
        "ICaptureProvider": "IAcquireProvider",
        "CaptureProvider": "AcquireProvider",
    }
    
    def obfuscate_string_content(self, string_inner: str) -> str:
        """문자열 내용만 치환"""
        result = string_inner
        for original, replacement in self.sorted_replacements:
            if original in result:
                result = result.replace(original, replacement)
                self.stats["replacements"] += 1
        return result
    
    def obfuscate_content(self, content: str) -> str:
        """소스 코드 내 문자열 리터럴만 난독화"""
        
        # Step 1: #include 라인 보존
        include_lines = []
        def save_include(match):
            include_lines.append(match.group(0))
            return f"\x00INC{len(include_lines)-1}\x00"
        
        content = re.sub(r'^[ \t]*#include[^\n]*$', save_include, content, flags=re.MULTILINE)
        
        # Step 1.5: 코드 레벨 식별자 치환 (CUDA 커널 함수명, 구조체명 등)
        sorted_code_replacements = sorted(
            self.CODE_LEVEL_REPLACEMENTS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        for original, replacement in sorted_code_replacements:
            # 단어 경계를 고려한 치환
            content = re.sub(r'\b' + re.escape(original) + r'\b', replacement, content)
            if original in content:
                self.stats["replacements"] += content.count(original)
        
        # Step 2: 문자열 리터럴 치환
        def replace_string(match):
            full_match = match.group(0)
            quote = full_match[0]
            inner = full_match[1:-1]
            
            # 문자열 내용 치환
            new_inner = self.obfuscate_string_content(inner)
            
            return quote + new_inner + quote
        
        # 문자열 리터럴만 찾아서 치환
        result = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', replace_string, content)
        
        # Step 3: #include 라인 복원
        for i, inc in enumerate(include_lines):
            result = result.replace(f"\x00INC{i}\x00", inc)
        
        return result
    
    def obfuscate_directory(self):
        """전체 디렉토리 난독화"""
        source_dir = Path(self.config["source_dir"])
        output_dir = Path(self.config["output_dir"])
        
        # 난독화 제외 폴더 (외부 라이브러리)
        exclude_dirs = {"modules", "imgui", "include"}
        
        # 출력 디렉토리 생성
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        # 전체 복사
        shutil.copytree(source_dir, output_dir)
        print(f"Copied {source_dir} -> {output_dir}")
        
        # 파일 난독화
        for ext in self.config["extensions"]:
            for file_path in output_dir.rglob(f"*{ext}"):
                # 제외 폴더 체크
                relative_path = file_path.relative_to(output_dir)
                if any(part in exclude_dirs for part in relative_path.parts):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    obfuscated = self.obfuscate_content(content)
                    
                    if content != obfuscated:
                        file_path.write_text(obfuscated, encoding='utf-8')
                        self.stats["files"] += 1
                        
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
        
        print(f"Obfuscated {self.stats['files']} files (excluded: {exclude_dirs})")
        print(f"Total string replacements: {self.stats['replacements']}")
        
        print(f"Obfuscated {self.stats['files']} files")
        print(f"Total string replacements: {self.stats['replacements']}")
        
        # 치환 맵 저장
        map_file = output_dir / "_string_map.json"
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump(STRING_REPLACEMENTS, f, indent=2, ensure_ascii=False)
        print(f"String map saved to {map_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="C++ String Obfuscator (Safe)")
    parser.add_argument("--source", default="needaimbot", help="Source directory")
    parser.add_argument("--output", default="needaimbot_obf", help="Output directory")
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config["source_dir"] = args.source
    config["output_dir"] = args.output
    
    obfuscator = StringObfuscator(config)
    obfuscator.obfuscate_directory()
    print("\nString obfuscation complete!")


if __name__ == "__main__":
    main()
