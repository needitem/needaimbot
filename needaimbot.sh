#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")/inference_pc"  # 스크립트 위치 기준 — 폴더 이동에 안전
exec ./build/bin/Release/simple_inference 
