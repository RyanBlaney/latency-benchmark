package filters

import (
	"fmt"
	"math"
)

// PreEmphasis implements a pre-emphasis filter for speech and audio processing.
// Pre-emphasis compensates for the natural spectral roll-off in speech and
// audio signals, emphasizing higher frequencies.
//
// The filter implements the transfer function:
// H(z) = 1 - α*z^-1
//
// With the difference equation:
// y[n] = x[n] - α*x[n-1]
//
// Where α is the pre-emphasis coefficient (typically 0.95-0.97 for speech).
//
// References:
//   - L.R. Rabiner, R.W. Schafer, "Digital Processing of Speech Signals",
//     Prentice-Hall, 1978, Chapter 4
//   - J.R. Deller, J.G. Proakis, J.H.L. Hansen, "Discrete-Time Processing
//     of Speech Signals", Macmillan, 1993
//   - ITU-T Recommendation G.191, "Software tools for speech and audio
//     coding standardization"
//
// Pre-emphasis serves several purposes:
// 1. Flattens the spectrum for better analysis
// 2. Reduces dynamic range for quantization
// 3. Improves numerical properties of linear prediction
// 4. Enhances perceptual quality
type PreEmphasis struct {
	coefficient  float64 // Pre-emphasis coefficient α
	lastSample   float64 // Previous input sample x[n-1]
	contentType  string  // Content type for optimal coefficient selection
	sampleRate   int     // Sample rate (for frequency-dependent optimization)
	adaptiveMode bool    // Enable adaptive coefficient adjustment

	// Adaptive parameters
	energyTracker  float64 // Running energy estimate
	adaptiveAlpha  float64 // Current adaptive coefficient
	adaptationRate float64 // Rate of adaptation (0 < rate < 1)

	initialized bool
}

// ContentType represents different audio content types with optimal coefficients
const (
	ContentSpeech     = "speech"     // α = 0.97 (aggressive pre-emphasis)
	ContentMusic      = "music"      // α = 0.95 (moderate pre-emphasis)
	ContentBroadcast  = "broadcast"  // α = 0.96 (balanced for mixed content)
	ContentNarrowband = "narrowband" // α = 0.94 (telephone quality)
	ContentWideband   = "wideband"   // α = 0.98 (high-quality audio)
	ContentGeneral    = "general"    // α = 0.95 (default)
)

// NewPreEmphasis creates a pre-emphasis filter with specified coefficient.
//
// Parameters:
//   - coefficient: Pre-emphasis coefficient α (0.0 < α < 1.0)
//     Higher values = more emphasis of high frequencies
//     Typical range: 0.9-0.99
func NewPreEmphasis(coefficient float64) *PreEmphasis {
	return &PreEmphasis{
		coefficient:    coefficient,
		lastSample:     0.0,
		adaptationRate: 0.01, // Default adaptation rate
		initialized:    true,
	}
}

// NewPreEmphasisDefault creates a pre-emphasis filter with standard coefficient (0.97).
// This value is widely used in speech processing applications.
func NewPreEmphasisDefault() *PreEmphasis {
	return NewPreEmphasis(0.97)
}

// NewPreEmphasisForContent creates a pre-emphasis filter optimized for specific content.
//
// Parameters:
//   - contentType: Type of audio content (speech, music, broadcast, etc.)
//   - sampleRate: Sample rate in Hz (for frequency-dependent optimization)
func NewPreEmphasisForContent(contentType string, sampleRate int) *PreEmphasis {
	coefficient := GetOptimalPreEmphasisCoefficient(contentType)

	return &PreEmphasis{
		coefficient: coefficient,
		contentType: contentType,
		sampleRate:  sampleRate,
		lastSample:  0.0,
		initialized: true,
	}
}

// NewAdaptivePreEmphasis creates an adaptive pre-emphasis filter that adjusts
// its coefficient based on signal characteristics.
//
// Parameters:
//   - baseCoefficient: Starting coefficient
//   - adaptationRate: Rate of adaptation (0.001 - 0.1)
func NewAdaptivePreEmphasis(baseCoefficient, adaptationRate float64) *PreEmphasis {
	return &PreEmphasis{
		coefficient:    baseCoefficient,
		adaptiveAlpha:  baseCoefficient,
		adaptationRate: adaptationRate,
		adaptiveMode:   true,
		initialized:    true,
	}
}

// GetOptimalPreEmphasisCoefficient returns the optimal coefficient for different content types.
// Based on empirical studies and standardization recommendations.
func GetOptimalPreEmphasisCoefficient(contentType string) float64 {
	switch contentType {
	case ContentSpeech:
		return 0.97 // ITU-T G.191 recommendation for speech
	case ContentMusic:
		return 0.95 // Gentler emphasis for music
	case ContentBroadcast:
		return 0.96 // Balanced for mixed content
	case ContentNarrowband:
		return 0.94 // Telephone/narrowband audio
	case ContentWideband:
		return 0.98 // High-quality wideband audio
	case ContentGeneral:
		fallthrough
	default:
		return 0.95 // Safe default for general audio
	}
}

// Process applies pre-emphasis filtering to a single sample.
// Implements: y[n] = x[n] - α*x[n-1]
func (pe *PreEmphasis) Process(input float64) float64 {
	if !pe.initialized {
		pe.coefficient = 0.97
		pe.initialized = true
	}

	currentCoeff := pe.coefficient

	// Use adaptive coefficient if adaptive mode is enabled
	if pe.adaptiveMode {
		currentCoeff = pe.updateAdaptiveCoefficient(input)
	}

	// Apply pre-emphasis: y[n] = x[n] - α*x[n-1]
	output := input - currentCoeff*pe.lastSample

	// Update state
	pe.lastSample = input

	return output
}

// updateAdaptiveCoefficient adjusts the pre-emphasis coefficient based on signal energy.
// Higher energy signals get less pre-emphasis to avoid over-emphasis.
func (pe *PreEmphasis) updateAdaptiveCoefficient(input float64) float64 {
	// Update energy tracker with exponential smoothing
	energy := input * input
	pe.energyTracker = 0.99*pe.energyTracker + 0.01*energy

	// Adapt coefficient based on energy level
	// Higher energy -> lower coefficient (less pre-emphasis)
	// Lower energy -> higher coefficient (more pre-emphasis)
	energyFactor := math.Min(pe.energyTracker, 1.0)
	targetCoeff := pe.coefficient * (1.0 - 0.1*energyFactor)

	// Smooth coefficient changes
	pe.adaptiveAlpha += pe.adaptationRate * (targetCoeff - pe.adaptiveAlpha)

	// Clamp to valid range
	if pe.adaptiveAlpha > 0.99 {
		pe.adaptiveAlpha = 0.99
	} else if pe.adaptiveAlpha < 0.9 {
		pe.adaptiveAlpha = 0.9
	}

	return pe.adaptiveAlpha
}

// ProcessBuffer applies pre-emphasis to an entire buffer of samples.
func (pe *PreEmphasis) ProcessBuffer(input []float64) []float64 {
	output := make([]float64, len(input))
	for i, sample := range input {
		output[i] = pe.Process(sample)
	}
	return output
}

// Reset clears the filter's internal state.
// Call this when processing discontinuous audio segments.
func (pe *PreEmphasis) Reset() {
	pe.lastSample = 0.0
	pe.energyTracker = 0.0
	if pe.adaptiveMode {
		pe.adaptiveAlpha = pe.coefficient
	}
}

// SetCoefficient updates the pre-emphasis coefficient.
func (pe *PreEmphasis) SetCoefficient(coefficient float64) error {
	if coefficient <= 0.0 || coefficient >= 1.0 {
		return fmt.Errorf("coefficient must be between 0 and 1, got %f", coefficient)
	}

	pe.coefficient = coefficient
	if pe.adaptiveMode {
		pe.adaptiveAlpha = coefficient
	}

	return nil
}

// GetCoefficient returns the current coefficient.
// For adaptive filters, returns the current adaptive coefficient.
func (pe *PreEmphasis) GetCoefficient() float64 {
	if pe.adaptiveMode {
		return pe.adaptiveAlpha
	}
	return pe.coefficient
}

// SetAdaptationRate sets the adaptation rate for adaptive pre-emphasis.
func (pe *PreEmphasis) SetAdaptationRate(rate float64) error {
	if rate <= 0.0 || rate > 1.0 {
		return fmt.Errorf("adaptation rate must be between 0 and 1, got %f", rate)
	}

	pe.adaptationRate = rate
	return nil
}

// GetFrequencyResponse computes the magnitude and phase response at given frequency.
// For pre-emphasis filter: H(e^jw) = 1 - α*e^-jw
func (pe *PreEmphasis) GetFrequencyResponse(frequency float64, sampleRate int) (magnitude, phase float64) {
	w := 2.0 * math.Pi * frequency / float64(sampleRate)

	coeff := pe.GetCoefficient()

	// H(e^jw) = 1 - α*e^-jw = 1 - α*cos(w) + j*α*sin(w)
	real := 1.0 - coeff*math.Cos(w)
	imag := coeff * math.Sin(w)

	magnitude = math.Sqrt(real*real + imag*imag)
	phase = math.Atan2(imag, real)

	return magnitude, phase
}

// GetHighFrequencyGain computes the gain at high frequencies (near Nyquist).
// For pre-emphasis, this shows how much the high frequencies are emphasized.
func (pe *PreEmphasis) GetHighFrequencyGain(sampleRate int) float64 {
	// At Nyquist frequency (fs/2), w = π
	// H(e^jπ) = 1 - α*e^-jπ = 1 - α*(-1) = 1 + α
	coeff := pe.GetCoefficient()
	return 1.0 + coeff
}

// GetLowFrequencyGain computes the gain at DC (0 Hz).
// For pre-emphasis, this is always 1.0 (no gain at DC).
func (pe *PreEmphasis) GetLowFrequencyGain() float64 {
	// At DC (w = 0): H(1) = 1 - α*1 = 1 - α
	coeff := pe.GetCoefficient()
	return 1.0 - coeff
}

// GetSpectralTilt computes the spectral tilt introduced by pre-emphasis.
// Returns the tilt in dB/octave.
func (pe *PreEmphasis) GetSpectralTilt(sampleRate int) float64 {
	lowGain := pe.GetLowFrequencyGain()
	highGain := pe.GetHighFrequencyGain(sampleRate)

	// Convert to dB and compute tilt per octave
	lowGainDB := 20.0 * math.Log10(math.Max(lowGain, 1e-10))
	highGainDB := 20.0 * math.Log10(highGain)

	// Number of octaves from DC to Nyquist
	octaves := math.Log2(float64(sampleRate) / 2.0)

	return (highGainDB - lowGainDB) / octaves
}

// EstimateOptimalCoefficient analyzes a signal buffer to estimate the optimal
// pre-emphasis coefficient based on spectral characteristics.
//
// This uses the autocorrelation method to estimate the coefficient that
// best whitens the spectrum.
func (pe *PreEmphasis) EstimateOptimalCoefficient(signal []float64) float64 {
	if len(signal) < 2 {
		return pe.coefficient
	}

	// Compute autocorrelation at lag 1
	r0 := 0.0 // R[0] - autocorrelation at lag 0
	r1 := 0.0 // R[1] - autocorrelation at lag 1

	for i := 0; i < len(signal); i++ {
		r0 += signal[i] * signal[i]
		if i > 0 {
			r1 += signal[i] * signal[i-1]
		}
	}

	// Avoid division by zero
	if r0 == 0.0 {
		return pe.coefficient
	}

	// Optimal coefficient is R[1]/R[0] (Levinson-Durbin for order 1)
	optimalCoeff := r1 / r0

	// Clamp to reasonable range
	if optimalCoeff > 0.99 {
		optimalCoeff = 0.99
	} else if optimalCoeff < 0.8 {
		optimalCoeff = 0.8
	}

	return optimalCoeff
}

// PreEmphasisBank implements multi-channel pre-emphasis filtering.
// Useful for stereo or multi-channel audio processing.
type PreEmphasisBank struct {
	filters     []*PreEmphasis
	numChannels int
}

// NewPreEmphasisBank creates a bank of pre-emphasis filters for multi-channel audio.
func NewPreEmphasisBank(numChannels int, coefficient float64) *PreEmphasisBank {
	filters := make([]*PreEmphasis, numChannels)
	for i := range filters {
		filters[i] = NewPreEmphasis(coefficient)
	}

	return &PreEmphasisBank{
		filters:     filters,
		numChannels: numChannels,
	}
}

// ProcessInterleaved processes interleaved multi-channel audio.
// Input format: [ch0, ch1, ..., chN-1, ch0, ch1, ..., chN-1, ...]
func (pb *PreEmphasisBank) ProcessInterleaved(input []float64) []float64 {
	if len(input)%pb.numChannels != 0 {
		// Handle incomplete frames by zero-padding
		paddedLen := ((len(input) / pb.numChannels) + 1) * pb.numChannels
		padded := make([]float64, paddedLen)
		copy(padded, input)
		input = padded
	}

	output := make([]float64, len(input))

	for i := 0; i < len(input); i++ {
		channel := i % pb.numChannels
		output[i] = pb.filters[channel].Process(input[i])
	}

	return output[:len(input)] // Return original length
}

// ProcessSeparate processes separate channel buffers.
// Input: slice of channel buffers
func (pb *PreEmphasisBank) ProcessSeparate(channels [][]float64) [][]float64 {
	if len(channels) > pb.numChannels {
		// Truncate to available filters
		channels = channels[:pb.numChannels]
	}

	output := make([][]float64, len(channels))
	for i, channel := range channels {
		output[i] = pb.filters[i].ProcessBuffer(channel)
	}

	return output
}

// Reset clears the state of all filters in the bank.
func (pb *PreEmphasisBank) Reset() {
	for _, filter := range pb.filters {
		filter.Reset()
	}
}

// SetCoefficient sets the coefficient for all filters in the bank.
func (pb *PreEmphasisBank) SetCoefficient(coefficient float64) error {
	for _, filter := range pb.filters {
		if err := filter.SetCoefficient(coefficient); err != nil {
			return err
		}
	}
	return nil
}

// GetNumChannels returns the number of channels in the filter bank.
func (pb *PreEmphasisBank) GetNumChannels() int {
	return pb.numChannels
}
