package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// Converter handles the RVC voice conversion pipeline
type Converter struct {
	rmvpeSession  *ort.DynamicAdvancedSession
	hubertSession *ort.DynamicAdvancedSession
	rvcSession    *ort.DynamicAdvancedSession
}

// NewConverter initializes ONNX sessions for all models
func NewConverter() (*Converter, error) {
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	rmvpe, err := ort.NewDynamicAdvancedSession("models/rmvpe.onnx", []string{"input"}, []string{"output"}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to load rmvpe model: %w", err)
	}

	hubert, err := ort.NewDynamicAdvancedSession("models/hubert_base.onnx", []string{"source"}, []string{"embed"}, nil)
	if err != nil {
		rmvpe.Destroy()
		return nil, fmt.Errorf("failed to load hubert model: %w", err)
	}

	rvc, err := ort.NewDynamicAdvancedSession("models/Trump_RVC_v2.onnx",
		[]string{"phone", "pitch", "ds"}, []string{"audio"}, nil)
	if err != nil {
		rmvpe.Destroy()
		hubert.Destroy()
		return nil, fmt.Errorf("failed to load Trump RVC model: %w", err)
	}

	return &Converter{
		rmvpeSession:  rmvpe,
		hubertSession: hubert,
		rvcSession:    rvc,
	}, nil
}

// Close releases all ONNX sessions
func (c *Converter) Close() {
	if c.rmvpeSession != nil {
		c.rmvpeSession.Destroy()
	}
	if c.hubertSession != nil {
		c.hubertSession.Destroy()
	}
	if c.rvcSession != nil {
		c.rvcSession.Destroy()
	}
	ort.DestroyEnvironment()
}

// Convert processes a WAV file through the RVC pipeline
func (c *Converter) Convert(inputPath, outputPath string) error {
	audio, sampleRate, err := loadWAV(inputPath)
	if err != nil {
		return fmt.Errorf("failed to load WAV: %w", err)
	}

	// Resample to 16kHz for models if needed
	if sampleRate != 16000 {
		audio = resample(audio, sampleRate, 16000)
	}

	// Extract F0 (pitch) using RMVPE
	f0, err := c.extractF0(audio)
	if err != nil {
		return fmt.Errorf("failed to extract F0: %w", err)
	}

	// Extract features using HuBERT
	features, err := c.extractFeatures(audio)
	if err != nil {
		return fmt.Errorf("failed to extract features: %w", err)
	}

	// Generate Trump voice
	output, err := c.generate(features, f0)
	if err != nil {
		return fmt.Errorf("failed to generate voice: %w", err)
	}

	// Save output WAV
	if err := saveWAV(outputPath, output, 16000); err != nil {
		return fmt.Errorf("failed to save WAV: %w", err)
	}

	return nil
}

// extractF0 extracts pitch contour using RMVPE model
func (c *Converter) extractF0(audio []float32) ([]float32, error) {
	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(audio))), audio)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputs, err := c.rmvpeSession.Run([]ort.ArbitraryTensor{inputTensor})
	if err != nil {
		return nil, err
	}
	defer outputs[0].Destroy()

	f0Data := outputs[0].GetData().([]float32)
	return f0Data, nil
}

// extractFeatures extracts acoustic features using HuBERT
func (c *Converter) extractFeatures(audio []float32) ([]float32, error) {
	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(audio))), audio)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputs, err := c.hubertSession.Run([]ort.ArbitraryTensor{inputTensor})
	if err != nil {
		return nil, err
	}
	defer outputs[0].Destroy()

	features := outputs[0].GetData().([]float32)
	return features, nil
}

// generate produces output audio using the RVC model
func (c *Converter) generate(features, f0 []float32) ([]float32, error) {
	// Prepare input tensors
	featureLen := int64(len(features) / 256) // Assuming 256 feature dim
	phoneTensor, err := ort.NewTensor(ort.NewShape(1, featureLen, 256), features)
	if err != nil {
		return nil, err
	}
	defer phoneTensor.Destroy()

	// Upsample F0 to match feature length
	f0Upsampled := upsampleF0(f0, int(featureLen))
	pitchTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(f0Upsampled))), f0Upsampled)
	if err != nil {
		return nil, err
	}
	defer pitchTensor.Destroy()

	// Create speaker embedding (ds)
	ds := make([]float32, 256)
	dsTensor, err := ort.NewTensor(ort.NewShape(1, 256), ds)
	if err != nil {
		return nil, err
	}
	defer dsTensor.Destroy()

	outputs, err := c.rvcSession.Run([]ort.ArbitraryTensor{phoneTensor, pitchTensor, dsTensor})
	if err != nil {
		return nil, err
	}
	defer outputs[0].Destroy()

	audio := outputs[0].GetData().([]float32)
	return audio, nil
}

// loadWAV reads a WAV file and returns audio samples and sample rate
func loadWAV(path string) ([]float32, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}

	if len(data) < 44 {
		return nil, 0, fmt.Errorf("invalid WAV file: too short")
	}

	// Parse WAV header
	sampleRate := int(binary.LittleEndian.Uint32(data[24:28]))
	bitsPerSample := binary.LittleEndian.Uint16(data[34:36])
	numChannels := binary.LittleEndian.Uint16(data[22:24])

	// Convert to float32 mono
	audioData := data[44:]
	samples := make([]float32, len(audioData)/(int(bitsPerSample)/8)/int(numChannels))

	for i := range samples {
		var sample int16
		if bitsPerSample == 16 {
			idx := i * int(numChannels) * 2
			if idx+1 < len(audioData) {
				sample = int16(binary.LittleEndian.Uint16(audioData[idx : idx+2]))
			}
		}
		samples[i] = float32(sample) / 32768.0
	}

	return samples, sampleRate, nil
}

// saveWAV writes audio samples to a WAV file
func saveWAV(path string, audio []float32, sampleRate int) error {
	numSamples := len(audio)
	bitsPerSample := 16
	numChannels := 1
	byteRate := sampleRate * numChannels * bitsPerSample / 8
	blockAlign := numChannels * bitsPerSample / 8
	dataSize := numSamples * blockAlign

	header := make([]byte, 44)
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], uint32(36+dataSize))
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)
	binary.LittleEndian.PutUint16(header[20:22], 1)
	binary.LittleEndian.PutUint16(header[22:24], uint16(numChannels))
	binary.LittleEndian.PutUint32(header[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(header[28:32], uint32(byteRate))
	binary.LittleEndian.PutUint16(header[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(header[34:36], uint16(bitsPerSample))
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], uint32(dataSize))

	audioBytes := make([]byte, dataSize)
	for i, sample := range audio {
		// Clamp and convert to int16
		val := sample * 32768.0
		if val > 32767 {
			val = 32767
		} else if val < -32768 {
			val = -32768
		}
		binary.LittleEndian.PutUint16(audioBytes[i*2:i*2+2], uint16(int16(val)))
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.Write(header); err != nil {
		return err
	}
	if _, err := f.Write(audioBytes); err != nil {
		return err
	}

	return nil
}

// resample performs simple linear interpolation resampling
func resample(audio []float32, fromRate, toRate int) []float32 {
	if fromRate == toRate {
		return audio
	}
	ratio := float64(fromRate) / float64(toRate)
	newLen := int(float64(len(audio)) / ratio)
	resampled := make([]float32, newLen)

	for i := range resampled {
		srcIdx := float64(i) * ratio
		idx := int(srcIdx)
		frac := srcIdx - float64(idx)

		if idx+1 < len(audio) {
			resampled[i] = float32((1-frac)*float64(audio[idx]) + frac*float64(audio[idx+1]))
		} else if idx < len(audio) {
			resampled[i] = audio[idx]
		}
	}

	return resampled
}

// upsampleF0 upsamples F0 contour to target length
func upsampleF0(f0 []float32, targetLen int) []float32 {
	if len(f0) == targetLen {
		return f0
	}
	upsampled := make([]float32, targetLen)
	ratio := float64(len(f0)) / float64(targetLen)

	for i := range upsampled {
		srcIdx := float64(i) * ratio
		idx := int(srcIdx)
		frac := srcIdx - float64(idx)

		if idx+1 < len(f0) {
			upsampled[i] = float32((1-frac)*float64(f0[idx]) + frac*float64(f0[idx+1]))
		} else if idx < len(f0) {
			upsampled[i] = f0[idx]
		} else {
			upsampled[i] = 0
		}
	}

	// Apply median filter to smooth pitch contour
	for i := 1; i < len(upsampled)-1; i++ {
		vals := []float32{upsampled[i-1], upsampled[i], upsampled[i+1]}
		upsampled[i] = median(vals)
	}

	return upsampled
}

// median returns the median of three values
func median(vals []float32) float32 {
	if vals[0] > vals[1] {
		vals[0], vals[1] = vals[1], vals[0]
	}
	if vals[1] > vals[2] {
		vals[1], vals[2] = vals[2], vals[1]
	}
	if vals[0] > vals[1] {
		vals[0], vals[1] = vals[1], vals[0]
	}
	return vals[1]
}

// Suppress unused warning
var _ = math.Abs
