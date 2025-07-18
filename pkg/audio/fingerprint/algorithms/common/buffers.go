package common

import (
	"fmt"
)

// CircularBuffer implements a circular buffer for streaming audio processing
type CircularBuffer struct {
	buffer   []float64
	size     int
	writePos int
	readPos  int
	count    int
}

// NewCircularBuffer creates a new circular buffer
func NewCircularBuffer(size int) *CircularBuffer {
	return &CircularBuffer{
		buffer: make([]float64, size),
		size:   size,
	}
}

// Write adds data to the buffer
func (cb *CircularBuffer) Write(data []float64) int {
	written := 0
	for _, sample := range data {
		if cb.count < cb.size {
			cb.buffer[cb.writePos] = sample
			cb.writePos = (cb.writePos + 1) % cb.size
			cb.count++
			written++
		} else {
			// Buffer full, overwrite oldest data
			cb.buffer[cb.writePos] = sample
			cb.writePos = (cb.writePos + 1) % cb.size
			cb.readPos = (cb.readPos + 1) % cb.size
			written++
		}
	}
	return written
}

// Read reads data from the buffer
func (cb *CircularBuffer) Read(data []float64) int {
	read := 0
	for i := range data {
		if cb.count > 0 {
			data[i] = cb.buffer[cb.readPos]
			cb.readPos = (cb.readPos + 1) % cb.size
			cb.count--
			read++
		} else {
			break
		}
	}
	return read
}

// Peek reads data without consuming it
func (cb *CircularBuffer) Peek(data []float64) int {
	read := 0
	pos := cb.readPos
	remaining := cb.count

	for i := range data {
		if remaining > 0 {
			data[i] = cb.buffer[pos]
			pos = (pos + 1) % cb.size
			remaining--
			read++
		} else {
			break
		}
	}
	return read
}

// Available returns number of samples available for reading
func (cb *CircularBuffer) Available() int {
	return cb.count
}

// Space returns available space for writing
func (cb *CircularBuffer) Space() int {
	return cb.size - cb.count
}

// Clear empties the buffer
func (cb *CircularBuffer) Clear() {
	cb.writePos = 0
	cb.readPos = 0
	cb.count = 0
}

// IsFull returns true if buffer is full
func (cb *CircularBuffer) IsFull() bool {
	return cb.count == cb.size
}

// IsEmpty returns true if buffer is empty
func (cb *CircularBuffer) IsEmpty() bool {
	return cb.count == 0
}

// SlidingWindow implements a sliding window for frame-based processing
type SlidingWindow struct {
	buffer     []float64
	windowSize int
	hopSize    int
	writePos   int
	frameReady bool
}

// NewSlidingWindow creates a new sliding window
func NewSlidingWindow(windowSize, hopSize int) *SlidingWindow {
	return &SlidingWindow{
		buffer:     make([]float64, windowSize),
		windowSize: windowSize,
		hopSize:    hopSize,
	}
}

// AddSamples adds samples and returns frames when ready
func (sw *SlidingWindow) AddSamples(samples []float64) [][]float64 {
	var frames [][]float64

	for _, sample := range samples {
		sw.buffer[sw.writePos] = sample
		sw.writePos++

		// Check if we have a complete frame
		if sw.writePos >= sw.windowSize {
			// Extract frame
			frame := make([]float64, sw.windowSize)
			copy(frame, sw.buffer)
			frames = append(frames, frame)

			// Slide the window
			if sw.hopSize < sw.windowSize {
				// Overlap: shift buffer left by hopSize
				copy(sw.buffer, sw.buffer[sw.hopSize:])
				sw.writePos = sw.windowSize - sw.hopSize
			} else {
				// No overlap: reset buffer
				sw.writePos = 0
			}
		}
	}

	return frames
}

// Reset clears the sliding window
func (sw *SlidingWindow) Reset() {
	sw.writePos = 0
	sw.frameReady = false
	for i := range sw.buffer {
		sw.buffer[i] = 0.0
	}
}

// GetWindowSize returns the window size
func (sw *SlidingWindow) GetWindowSize() int {
	return sw.windowSize
}

// GetHopSize returns the hop size
func (sw *SlidingWindow) GetHopSize() int {
	return sw.hopSize
}

// DelayLine implements a delay line for audio effects
type DelayLine struct {
	buffer   []float64
	size     int
	writePos int
}

// NewDelayLine creates a new delay line
func NewDelayLine(maxDelaySamples int) *DelayLine {
	return &DelayLine{
		buffer: make([]float64, maxDelaySamples),
		size:   maxDelaySamples,
	}
}

// Process processes a sample through the delay line
func (dl *DelayLine) Process(input float64, delaySamples int) float64 {
	if delaySamples >= dl.size {
		delaySamples = dl.size - 1
	}

	// Read delayed sample
	readPos := (dl.writePos - delaySamples + dl.size) % dl.size
	output := dl.buffer[readPos]

	// Write new sample
	dl.buffer[dl.writePos] = input
	dl.writePos = (dl.writePos + 1) % dl.size

	return output
}

// ProcessInterpolated processes with fractional delay using linear interpolation
func (dl *DelayLine) ProcessInterpolated(input float64, delaySamples float64) float64 {
	if delaySamples >= float64(dl.size) {
		delaySamples = float64(dl.size - 1)
	}

	// Integer and fractional parts
	intDelay := int(delaySamples)
	fracDelay := delaySamples - float64(intDelay)

	// Read positions
	readPos1 := (dl.writePos - intDelay + dl.size) % dl.size
	readPos2 := (dl.writePos - intDelay - 1 + dl.size) % dl.size

	// Linear interpolation
	sample1 := dl.buffer[readPos1]
	sample2 := dl.buffer[readPos2]
	output := sample1 + fracDelay*(sample2-sample1)

	// Write new sample
	dl.buffer[dl.writePos] = input
	dl.writePos = (dl.writePos + 1) % dl.size

	return output
}

// Clear empties the delay line
func (dl *DelayLine) Clear() {
	for i := range dl.buffer {
		dl.buffer[i] = 0.0
	}
}

// OverlapAddBuffer manages overlap-add processing for STFT
type OverlapAddBuffer struct {
	buffer     []float64
	windowSize int
	hopSize    int
	overlap    int
}

// NewOverlapAddBuffer creates a new overlap-add buffer
func NewOverlapAddBuffer(windowSize, hopSize int) *OverlapAddBuffer {
	overlap := windowSize - hopSize
	overlap = max(overlap, 0)

	return &OverlapAddBuffer{
		buffer:     make([]float64, windowSize),
		windowSize: windowSize,
		hopSize:    hopSize,
		overlap:    overlap,
	}
}

// AddFrame adds a windowed frame and returns output samples
func (oab *OverlapAddBuffer) AddFrame(frame []float64) ([]float64, error) {
	if len(frame) != oab.windowSize {
		return nil, fmt.Errorf("frame size (%d) doesn't match window size (%d)", len(frame), oab.windowSize)
	}

	// Add frame to buffer with overlap
	for i := range frame {
		oab.buffer[i] += frame[i]
	}

	// Extract output samples
	output := make([]float64, oab.hopSize)
	copy(output, oab.buffer[:oab.hopSize])

	// Shift buffer for next frame
	if oab.overlap > 0 {
		copy(oab.buffer, oab.buffer[oab.hopSize:])
		// Clear the end
		for i := oab.overlap; i < oab.windowSize; i++ {
			oab.buffer[i] = 0.0
		}
	} else {
		// No overlap, clear entire buffer
		for i := range oab.buffer {
			oab.buffer[i] = 0.0
		}
	}

	return output, nil
}

// Reset clears the buffer
func (oab *OverlapAddBuffer) Reset() {
	for i := range oab.buffer {
		oab.buffer[i] = 0.0
	}
}
