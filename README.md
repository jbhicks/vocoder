# vocoder

Pure Go desktop app with Fyne GUI for converting WAV files to Donald Trump's voice using ONNX models.

## Features

- Simple GUI for selecting input/output WAV files
- Full RVC (Retrieval-based Voice Conversion) pipeline
- F0 extraction using RMVPE model
- Feature extraction using HuBERT model
- Voice generation using Trump RVC model

## Requirements

- Go 1.22+
- ONNX Runtime libraries
- X11/OpenGL libraries (for GUI on Linux)
- ONNX models in `models/` directory:
  - `hubert_base.onnx`
  - `rmvpe.onnx`
  - `Trump_RVC_v2.onnx`

## Installation

```bash
go mod download
go build
```

## Usage

1. Place the required ONNX models in the `models/` directory
2. Run the application: `./vocoder`
3. Select an input WAV file
4. Select an output path
5. Click "Convert to Trump Voice"

## Implementation

- `main.go`: Fyne GUI application (108 LOC)
- `converter.go`: RVC pipeline implementation (331 LOC)
- Total: 439 LOC (under 500 LOC requirement)
