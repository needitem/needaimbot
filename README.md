# NeedAimBot — 2PC Architecture

AI 기반 조준 보조 시스템. TensorRT 추론과 하드웨어 마우스 제어(Makcu/Kmbox)를
별도 Inference PC(Jetson/Linux)에서 실행하고, Game PC(Windows)는 화면 캡처와
UDP 전송만 담당하는 **듀얼 PC 아키텍처**.

```
┌─────────────────┐                    ┌──────────────────┐
│    Game PC      │    UDP (frames)    │  Inference PC    │
│  Windows        │ ─────────────────→ │  Linux / Jetson  │
│  화면 캡처       │                    │  TensorRT 추론   │
│  UDP 전송        │                    │  마우스 제어      │
└────────┬────────┘                    └────────┬─────────┘
         │          Makcu / Kmbox USB           │
         └──────────────←───────────────────────┘
                   Direct Mouse Input
```

## 디렉터리 구조

```
.
├── game_pc/              # Windows 화면 캡처 + UDP 전송
│   ├── CMakeLists.txt
│   ├── config.ini
│   └── src/main.cpp
├── inference_pc/         # TensorRT 추론 + 마우스 제어
│   ├── build.sh          # 빌드 스크립트 (Linux/Jetson)
│   ├── CMakeLists.txt
│   ├── simple_config.default.json
│   ├── CONFIG_REFERENCE.md
│   ├── needaimbot/       # 핵심 소스 (app, capture, cuda, mouse)
│   ├── presets/          # 프리셋 설정
│   └── third_party/
├── bench/                # 조준 튜닝 하네스 (Python)
├── tools/                # 운영 도구 (측정, 네트워크 튜닝, 프리셋)
├── needaimbot.sh         # Inference PC 실행 스크립트 (Jetson 클럭 고정 포함)
├── needaimbot_stable.sh  # 안정 버전 실행 스크립트
└── CMakeLists.txt        # 루트 CMake (inference_pc 래퍼)
```

## 빌드

### Inference PC (Linux / Jetson)

요구사항: CUDA Toolkit, TensorRT, CMake 3.20+

```bash
cd inference_pc
./build.sh
```

또는 수동:

```bash
cd inference_pc
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Game PC (Windows)

요구사항: Visual Studio 2022 (C++ Desktop Development), CMake 3.20+

```bash
cd game_pc
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## 실행

Inference PC에서:

```bash
./needaimbot.sh
```

이 스크립트는 Jetson GPU/CPU 클럭을 최대로 고정한 뒤 추론 바이너리를 실행하고,
종료 시(Ctrl-C) 클럭을 자동 복원합니다.

Game PC에서는 빌드된 `game_pc.exe`를 관리자 권한으로 실행합니다.

## 설정

- **기본 설정**: `inference_pc/simple_config.default.json`을 복사하여 수정
- **설정 항목 레퍼런스**: `inference_pc/CONFIG_REFERENCE.md`
- **프리셋**: `inference_pc/presets/` 디렉터리
- **Game PC 네트워크**: `game_pc/config.ini`

## 도구 (tools/)

| 파일 | 용도 |
|---|---|
| `measure.sh` | 캘리브레이션 데이터 수집 모드 전환 |
| `preset.sh` | 프리셋 적용 및 프리셋 README 자동 생성 |
| `gen_preset_readme.py` | 프리셋 문서 생성 |
| `net_tune.sh` | 네트워크 튜닝 (sysctl 등) |
| `net_measure.sh` | 네트워크 지연시간 측정 |
| `net_coalesce_probe.sh` | NIC coalesce 설정 탐색 |
| `needaimbot-nettune.service` | 네트워크 튜닝 systemd 서비스 |
| `install_hooks.sh` | Git pre-commit hook 설치 |
| `latency_mode_benchmark.py` | 지연시간 모드 벤치마크 |
| `makcu_square.py` | Makcu 사각파 테스트 |
| `udp_capture_benchmark.cpp` | UDP 캡처 벤치마크 (C++) |
| `udp_frame_load_test.py` | UDP 프레임 부하 테스트 |

## 벤치마크 (bench/)

조준 컨트롤러 튜닝, 실측 데이터 분석, 패리티 테스트를 위한 Python 하네스.
자세한 내용은 `bench/README.md` 참조.

## 하드웨어 마우스 설정

### Makcu (Linux/Jetson)

USB 리셋 권한을 위한 udev 규칙 설정:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1a86", MODE="0666"' | sudo tee /etc/udev/rules.d/99-makcu.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Kmbox

동일 네트워크에 연결 후 `simple_config.default.json`에서 IP/Port/UUID 설정.

## License

See [LICENSE](LICENSE).
