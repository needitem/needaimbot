
*   **FP16 & INT8 Support**: Optimize for speed with half-precision (FP16) or maximum throughput with INT8 quantization.
*   **Dynamic Shapes**: Configurable input dimensions to match your capture resolution.
*   **Layer Fusion**: Automatically fuses compatible layers for reduced memory bandwidth usage.
*   **NeedAimBot Compatible**: Generates engines strictly compatible with the NeedAimBot runtime.

## Prerequisites

*   **Python**: 3.10 or higher.
*   **NVIDIA Drivers**: Latest Game Ready Driver.
*   **CUDA Toolkit**: Must match the version used by NeedAimBot (v12.x).
*   **TensorRT**: Must match the version used by NeedAimBot (v10.x).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/needitem/EngineExport.git
    cd EngineExport
    ```
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare your model**: Place your trained `.onnx` file (e.g., `yolov8n.onnx`) in the `models/` directory.
2.  **Run the exporter**:
    ```bash
    python export.py --model models/yolov8n.onnx --precision fp16 --output models/yolov8n.engine
    ```
3.  **Deploy**:
    *   Copy the generated `.engine` file.
    *   Paste it into the `needaimbot/models/` directory of your main application.
    *   Update `config.ini` in NeedAimBot to load this new model.

## Advanced Options

*   `--precision int8`: Requires a calibration dataset (images) in the `calibration/` folder. Provides the fastest inference but requires careful calibration.
*   `--workspace`: Set the maximum workspace size (in MB) for the builder (default: 4096).

## Disclaimer

This tool is intended for educational and research purposes. Ensure you have the rights to use and modify the AI models you are exporting.
