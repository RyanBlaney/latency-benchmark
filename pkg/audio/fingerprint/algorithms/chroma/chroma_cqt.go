package chroma

import (
	"math"
	"math/cmplx"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/spectral"
)

// ChromaCQT computes chromagram using Constant-Q Transform
//
// DIFFERENCE FROM ChromaSTFT:
// - ChromaSTFT: Uses FFT with linear frequency spacing
//   - Fixed frequency resolution across all frequencies
//   - Good for broadband analysis
//   - Computationally efficient
//
// - ChromaCQT: Uses Constant-Q Transform with logarithmic frequency spacing
//   - Variable frequency resolution (higher resolution at low frequencies)
//   - Matches human auditory perception and musical scales
//   - Better separation of low-frequency harmonics
//   - More accurate for musical analysis but computationally intensive
//
// CQT frequency spacing: f_k = f_min * 2^(k/bins_per_octave)
// This matches musical note spacing where each octave doubles in frequency
type ChromaCQT struct {
	sampleRate    int
	fft           *spectral.FFT
	minFreq       float64 // Minimum frequency (typically C1 ≈ 32.7 Hz)
	maxFreq       float64 // Maximum frequency
	binsPerOctave int     // Number of bins per octave (typically 12, 24, or 36)
	chromaBins    int     // Number of chroma bins (always 12)
	qFactor       float64 // Quality factor (frequency/bandwidth)
	tuningFreq    float64 // A4 frequency (default 440 Hz)

	// Pre-computed CQT kernel
	cqtKernel      [][]complex128 // CQT transformation matrix
	freqBins       []float64      // CQT frequency bins
	kernelComputed bool
}

// NewChromaCQT creates a new CQT-based chromagram calculator
func NewChromaCQT(sampleRate int, minFreq, maxFreq float64, binsPerOctave int, qFactor, tuningFreq float64) *ChromaCQT {
	return &ChromaCQT{
		sampleRate:    sampleRate,
		fft:           spectral.NewFFT(),
		minFreq:       minFreq,
		maxFreq:       maxFreq,
		binsPerOctave: binsPerOctave,
		chromaBins:    12,
		qFactor:       qFactor,
		tuningFreq:    tuningFreq,
	}
}

// NewChromaCQTDefault creates CQT chromagram with standard musical settings
func NewChromaCQTDefault(sampleRate int) *ChromaCQT {
	return NewChromaCQT(
		sampleRate,
		65.4,   // C2 frequency
		2093.0, // C7 frequency (5 octaves)
		12,     // 12 bins per octave (semitone resolution)
		25.0,   // Quality factor
		440.0,  // A4 = 440 Hz
	)
}

// ComputeChroma computes CQT-based chromagram from audio signal
func (cqt *ChromaCQT) ComputeChroma(signal []float64, hopSize int) ([][]float64, error) {
	if len(signal) == 0 {
		return nil, nil
	}

	// Initialize CQT kernel if not computed
	if !cqt.kernelComputed {
		err := cqt.computeCQTKernel()
		if err != nil {
			return nil, err
		}
	}

	// Compute CQT spectrogram
	cqtSpectrogram, err := cqt.computeCQTSpectrogram(signal, hopSize)
	if err != nil {
		return nil, err
	}

	// Convert CQT to chromagram by summing across octaves
	chromagram := cqt.convertCQTToChroma(cqtSpectrogram)

	return chromagram, nil
}

// computeCQTKernel pre-computes the CQT transformation kernel
func (cqt *ChromaCQT) computeCQTKernel() error {
	// Calculate number of CQT bins
	numOctaves := math.Log2(cqt.maxFreq / cqt.minFreq)
	totalBins := int(numOctaves * float64(cqt.binsPerOctave))

	// Generate CQT frequency bins
	cqt.freqBins = make([]float64, totalBins)
	for k := range totalBins {
		cqt.freqBins[k] = cqt.minFreq * math.Pow(2.0, float64(k)/float64(cqt.binsPerOctave))
	}

	// Determine FFT size (next power of 2 that accommodates longest kernel)
	maxKernelLength := cqt.calculateKernelLength(cqt.freqBins[0]) // Lowest frequency has longest kernel
	fftSize := nextPowerOfTwo(maxKernelLength * 2)                // Zero-pad for circular convolution

	// Initialize kernel matrix
	cqt.cqtKernel = make([][]complex128, totalBins)

	for k, freq := range cqt.freqBins {
		kernelLength := cqt.calculateKernelLength(freq)

		// Generate time-domain kernel (complex exponential windowed by Gaussian)
		kernel := make([]complex128, fftSize)

		// Calculate window parameters
		bandwidth := freq / cqt.qFactor
		sigma := float64(cqt.sampleRate) / (2.0 * math.Pi * bandwidth)

		center := kernelLength / 2
		for n := range kernelLength {
			t := float64(n - center)

			// Gaussian window
			window := math.Exp(-(t * t) / (2.0 * sigma * sigma))

			// Complex exponential
			phase := 2.0 * math.Pi * freq * t / float64(cqt.sampleRate)
			exponential := cmplx.Exp(complex(0, phase))

			// Windowed complex exponential
			kernel[n] = complex(window, 0) * exponential
		}

		// FFT of kernel for efficient convolution
		cqt.cqtKernel[k] = cqt.fft.Compute(complexToReal(kernel))
	}

	cqt.kernelComputed = true
	return nil
}

// calculateKernelLength calculates the length of CQT kernel for given frequency
func (cqt *ChromaCQT) calculateKernelLength(frequency float64) int {
	// Kernel length is inversely proportional to frequency (Q = f/bandwidth)
	kernelLength := int(cqt.qFactor * float64(cqt.sampleRate) / frequency)

	// Ensure odd length for symmetry
	if kernelLength%2 == 0 {
		kernelLength++
	}

	// Minimum and maximum bounds
	if kernelLength < 3 {
		kernelLength = 3
	}
	if kernelLength > cqt.sampleRate/2 {
		kernelLength = cqt.sampleRate / 2
	}

	return kernelLength
}

// computeCQTSpectrogram computes the CQT spectrogram
func (cqt *ChromaCQT) computeCQTSpectrogram(signal []float64, hopSize int) ([][]float64, error) {
	numFrames := (len(signal) - hopSize) / hopSize
	if numFrames <= 0 {
		numFrames = 1
	}

	spectrogram := make([][]float64, numFrames)

	// Determine FFT size from kernel
	fftSize := len(cqt.cqtKernel[0])

	for frameIdx := 0; frameIdx < numFrames; frameIdx++ {
		startIdx := frameIdx * hopSize

		// Extract frame with zero-padding if necessary
		frame := make([]float64, fftSize)
		for i := range fftSize {
			if startIdx+i < len(signal) {
				frame[i] = signal[startIdx+i]
			}
		}

		// FFT of signal frame
		frameFFT := cqt.fft.Compute(frame)

		// Apply CQT kernels
		cqtFrame := make([]float64, len(cqt.freqBins))
		for k := range cqt.freqBins {
			// Pointwise multiplication in frequency domain (convolution in time)
			cqtBin := complex(0, 0)
			for n := 0; n < len(frameFFT) && n < len(cqt.cqtKernel[k]); n++ {
				cqtBin += frameFFT[n] * cmplx.Conj(cqt.cqtKernel[k][n])
			}

			// Magnitude
			cqtFrame[k] = cmplx.Abs(cqtBin)
		}

		spectrogram[frameIdx] = cqtFrame
	}

	return spectrogram, nil
}

// convertCQTToChroma converts CQT spectrogram to chromagram
func (cqt *ChromaCQT) convertCQTToChroma(cqtSpectrogram [][]float64) [][]float64 {
	if len(cqtSpectrogram) == 0 {
		return [][]float64{}
	}

	chromagram := make([][]float64, len(cqtSpectrogram))

	for t := range cqtSpectrogram {
		chromagram[t] = make([]float64, cqt.chromaBins)

		// Sum across octaves for each chroma class
		for k, freq := range cqt.freqBins {
			// Convert frequency to chroma bin
			midiNote := cqt.frequencyToMIDI(freq)
			chromaBin := int(math.Round(midiNote)) % cqt.chromaBins
			if chromaBin < 0 {
				chromaBin += cqt.chromaBins
			}

			// Add energy to chroma bin (use magnitude squared for energy)
			energy := cqtSpectrogram[t][k] * cqtSpectrogram[t][k]
			chromagram[t][chromaBin] += energy
		}

		// Normalize chroma frame
		cqt.normalizeChromaFrame(chromagram[t])
	}

	return chromagram
}

// frequencyToMIDI converts frequency to MIDI note number
func (cqt *ChromaCQT) frequencyToMIDI(frequency float64) float64 {
	if frequency <= 0 {
		return 0
	}

	// MIDI note number: 69 + 12 * log2(f/440)
	return 69.0 + 12.0*math.Log2(frequency/cqt.tuningFreq)
}

// normalizeChromaFrame normalizes a single chroma frame
func (cqt *ChromaCQT) normalizeChromaFrame(chromaFrame []float64) {
	// Calculate total energy
	totalEnergy := 0.0
	for _, energy := range chromaFrame {
		totalEnergy += energy
	}

	// Normalize to unit sum
	if totalEnergy > 1e-10 {
		for i := range chromaFrame {
			chromaFrame[i] /= totalEnergy
		}
	}
}

// GetChromaLabels returns the chroma bin labels
func (cqt *ChromaCQT) GetChromaLabels() []string {
	return []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
}

// GetCQTFrequencies returns the CQT frequency bins
func (cqt *ChromaCQT) GetCQTFrequencies() []float64 {
	if !cqt.kernelComputed {
		return []float64{}
	}

	freqs := make([]float64, len(cqt.freqBins))
	copy(freqs, cqt.freqBins)
	return freqs
}

// SetTuning updates the tuning frequency and recomputes kernel
func (cqt *ChromaCQT) SetTuning(tuningFreq float64) {
	cqt.tuningFreq = tuningFreq
	cqt.kernelComputed = false // Force recomputation
}

// GetQFactor returns the quality factor
func (cqt *ChromaCQT) GetQFactor() float64 {
	return cqt.qFactor
}

// SetQFactor updates the quality factor and recomputes kernel
func (cqt *ChromaCQT) SetQFactor(qFactor float64) {
	cqt.qFactor = qFactor
	cqt.kernelComputed = false // Force recomputation
}

// Helper functions

// complexToReal extracts real part of complex array for FFT input
func complexToReal(data []complex128) []float64 {
	real := make([]float64, len(data))
	for i, val := range data {
		real[i] = cmplx.Abs(val) // Use magnitude for real representation
	}
	return real
}

// nextPowerOfTwo finds the next power of 2 >= n
func nextPowerOfTwo(n int) int {
	if n <= 0 {
		return 1
	}

	power := 1
	for power < n {
		power <<= 1
	}
	return power
}
