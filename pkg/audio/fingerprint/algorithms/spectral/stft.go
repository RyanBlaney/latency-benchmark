package spectral

import (
	"fmt"
	"math/cmplx"
	"runtime"
	"sync"

	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
)

// STFT provides Short-Time Fourier Transform functionality
// Extracted from your existing SpectralAnalyzer implementation
type STFT struct {
	fft    *FFT
	logger logging.Logger
}

// STFTResult holds the result of STFT analysis
// This is your existing SpectrogramResult structure
type STFTResult struct {
	Magnitude      [][]float64    `json:"magnitude"`       // Time x Frequency magnitude matrix
	Phase          [][]float64    `json:"phase"`           // Time x Frequency phase matrix
	Complex        [][]complex128 `json:"-"`               // Raw complex spectrogram (not serialized)
	TimeFrames     int            `json:"time_frames"`     // Number of time frames
	FreqBins       int            `json:"freq_bins"`       // Number of frequency bins
	SampleRate     int            `json:"sample_rate"`     // Sample rate
	WindowSize     int            `json:"window_size"`     // FFT window size
	HopSize        int            `json:"hop_size"`        // Hop size between frames
	FreqResolution float64        `json:"freq_resolution"` // Frequency resolution (Hz/bin)
	TimeResolution float64        `json:"time_resolution"` // Time resolution (seconds/frame)
}

// Window interface for windowing functions
type Window interface {
	ApplyInPlace(signal []float64) error
}

// NewSTFT creates a new STFT calculator
func NewSTFT() *STFT {
	return &STFT{
		fft: NewFFT(),
	}
}

// ComputeWithWindow computes STFT with parallel processing and custom window type
// This is your existing ComputeSTFTWithWindow implementation
func (s *STFT) ComputeWithWindow(signal []float64, windowSize int, hopSize int, sampleRate int, window Window) (*STFTResult, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("empty signal")
	}

	if windowSize <= 0 {
		return nil, fmt.Errorf("window size must be positive")
	}

	if hopSize <= 0 {
		return nil, fmt.Errorf("hop size must be positive")
	}

	// Calculate number of frames
	numFrames := (len(signal)-windowSize)/hopSize + 1
	if numFrames <= 0 {
		return nil, fmt.Errorf("signal too short for given window size and hop size")
	}

	// Calculate frequency bins (positive frequencies only)
	freqBins := windowSize/2 + 1

	// Initialize result matrices
	magnitude := make([][]float64, numFrames)
	phase := make([][]float64, numFrames)
	complexSpectrum := make([][]complex128, numFrames)

	// Pre-allocate all arrays
	for i := range numFrames {
		magnitude[i] = make([]float64, freqBins)
		phase[i] = make([]float64, freqBins)
		complexSpectrum[i] = make([]complex128, freqBins)
	}

	// Determine optimal number of workers based on system and workload
	numWorkers := s.getOptimalWorkerCount(numFrames)

	// Channel for frame processing jobs
	type frameJob struct {
		frameIdx int
		startIdx int
		endIdx   int
	}

	jobs := make(chan frameJob, numFrames)

	// Launch worker goroutines
	var wg sync.WaitGroup

	for range numWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Reuse frame buffer for this worker
			frameBuffer := make([]float64, windowSize)

			for job := range jobs {
				// Safety check
				if job.endIdx > len(signal) {
					continue
				}

				// Extract and window the frame
				copy(frameBuffer, signal[job.startIdx:job.endIdx])

				// Apply window function in-place
				if window != nil {
					err := window.ApplyInPlace(frameBuffer)
					if err != nil {
						// Log error if logger available, otherwise continue
						continue
					}
				}

				// Compute FFT
				fftResult := s.fft.Compute(frameBuffer)

				// Extract positive frequencies and compute magnitude/phase
				for i := range freqBins {
					complexSpectrum[job.frameIdx][i] = fftResult[i]
					magnitude[job.frameIdx][i] = cmplx.Abs(fftResult[i])
					phase[job.frameIdx][i] = cmplx.Phase(fftResult[i])
				}
			}
		}()
	}

	// Send jobs to workers
	go func() {
		defer close(jobs)
		for frameIdx := range numFrames {
			startIdx := frameIdx * hopSize
			endIdx := startIdx + windowSize

			if endIdx <= len(signal) {
				jobs <- frameJob{
					frameIdx: frameIdx,
					startIdx: startIdx,
					endIdx:   endIdx,
				}
			}
		}
	}()

	// Wait for all workers to complete
	wg.Wait()

	result := &STFTResult{
		Magnitude:      magnitude,
		Phase:          phase,
		Complex:        complexSpectrum,
		TimeFrames:     numFrames,
		FreqBins:       freqBins,
		SampleRate:     sampleRate,
		WindowSize:     windowSize,
		HopSize:        hopSize,
		FreqResolution: float64(sampleRate) / float64(windowSize),
		TimeResolution: float64(hopSize) / float64(sampleRate),
	}

	return result, nil
}

// ComputeSingleFrame computes FFT for a single frame
// This is your existing ComputeFFT implementation adapted
func (s *STFT) ComputeSingleFrame(signal []float64, sampleRate int) (*STFTResult, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("empty signal")
	}

	// Compute FFT
	fftResult := s.fft.Compute(signal)

	// Only keep positive frequencies (including DC and Nyquist)
	freqBins := len(fftResult)/2 + 1
	freqBins = min(len(fftResult), freqBins)

	// Create single-frame result matrices
	magnitude := make([][]float64, 1)
	phase := make([][]float64, 1)
	complexSpectrum := make([][]complex128, 1)

	magnitude[0] = make([]float64, freqBins)
	phase[0] = make([]float64, freqBins)
	complexSpectrum[0] = make([]complex128, freqBins)

	// Extract magnitude and phase for positive frequencies
	for i := 0; i < freqBins; i++ {
		complexSpectrum[0][i] = fftResult[i]
		magnitude[0][i] = cmplx.Abs(fftResult[i])
		phase[0][i] = cmplx.Phase(fftResult[i])
	}

	result := &STFTResult{
		Magnitude:      magnitude,
		Phase:          phase,
		Complex:        complexSpectrum,
		TimeFrames:     1, // Single frame
		FreqBins:       freqBins,
		SampleRate:     sampleRate,
		WindowSize:     len(signal),                                // Original signal length
		HopSize:        len(signal),                                // No overlap for single frame
		FreqResolution: float64(sampleRate) / float64(len(signal)), // Frequency resolution
		TimeResolution: float64(len(signal)) / float64(sampleRate), // Duration of the signal
	}

	return result, nil
}

// getOptimalWorkerCount determines the optimal number of workers based on workload
// This is your existing implementation
func (s *STFT) getOptimalWorkerCount(numFrames int) int {
	// Base number on available CPUs
	numCPU := runtime.NumCPU()

	// For small workloads, don't over-parallelize
	if numFrames < 100 {
		return min(numCPU/2, numFrames)
	}

	// For medium workloads, use most CPUs
	if numFrames < 1000 {
		return min(numCPU, 8) // Cap at 8 for medium loads
	}

	// For large workloads, use all available CPUs
	return numCPU
}
