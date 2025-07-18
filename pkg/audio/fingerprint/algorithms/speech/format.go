package speech

import (
	"fmt"
	"math"
	"sort"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/windowing"
)

// FormantAnalyzer extracts vocal tract resonances (formants) from speech
// WHY: Formants are critical for vowel identification, speaker characteristics,
// and linguistic content analysis. F1 and F2 primarily determine vowel identity.
type FormantAnalyzer struct {
	sampleRate  int
	windowSize  int
	maxFormants int
	maxFreq     float64
	minFreq     float64

	// LPC parameters
	lpcOrder    int
	preEmphasis float64

	// Internal components
	lpcAnalyzer *LPCAnalyzer
	window      *windowing.Hamming
}

// FormantResult contains formant analysis results
type FormantResult struct {
	Formants         []FormantData `json:"formants"`     // Detected formants (F1, F2, F3, etc.)
	VocalTractLength float64       `json:"vtl"`          // Estimated vocal tract length (cm)
	Quality          float64       `json:"quality"`      // Analysis quality score
	LPCOrder         int           `json:"lpc_order"`    // LPC order used
	NumFormants      int           `json:"num_formants"` // Number of formants found
}

// FormantData represents a single formant measurement
type FormantData struct {
	Frequency  float64 `json:"frequency"`  // Formant frequency (Hz)
	Bandwidth  float64 `json:"bandwidth"`  // Formant bandwidth (Hz)
	Amplitude  float64 `json:"amplitude"`  // Formant amplitude (relative)
	Confidence float64 `json:"confidence"` // Detection confidence (0-1)
}

// NewFormantAnalyzer creates a new formant analyzer
func NewFormantAnalyzer(sampleRate int) *FormantAnalyzer {
	windowSize := 1024
	if sampleRate >= 16000 {
		windowSize = 2048 // Larger window for higher sample rates
	}

	lpcOrder := 12 + sampleRate/1000 // Rule of thumb for speech

	return &FormantAnalyzer{
		sampleRate:  sampleRate,
		windowSize:  windowSize,
		maxFormants: 4,
		maxFreq:     float64(sampleRate) / 2.0,
		minFreq:     50.0, // Minimum formant frequency
		lpcOrder:    lpcOrder,
		preEmphasis: 0.97,
		lpcAnalyzer: NewLPCAnalyzer(sampleRate, lpcOrder),
		window:      windowing.NewHamming(windowSize, true),
	}
}

// NewFormantAnalyzerWithParams creates formant analyzer with custom parameters
func NewFormantAnalyzerWithParams(sampleRate, windowSize, lpcOrder int, maxFormants int) *FormantAnalyzer {
	return &FormantAnalyzer{
		sampleRate:  sampleRate,
		windowSize:  windowSize,
		maxFormants: maxFormants,
		maxFreq:     float64(sampleRate) / 2.0,
		minFreq:     50.0,
		lpcOrder:    lpcOrder,
		preEmphasis: 0.97,
		lpcAnalyzer: NewLPCAnalyzer(sampleRate, lpcOrder),
		window:      windowing.NewHamming(windowSize, true),
	}
}

// AnalyzeFormants extracts formants from speech signal
func (f *FormantAnalyzer) AnalyzeFormants(signal []float64) (*FormantResult, error) {
	if len(signal) < f.windowSize {
		return nil, fmt.Errorf("signal too short for formant analysis (need at least %d samples)", f.windowSize)
	}

	// Step 1: Preprocess signal
	processed := f.preprocessSignal(signal)

	// Step 2: Perform LPC analysis
	lpcResult, err := f.lpcAnalyzer.Analyze(processed)
	if err != nil {
		return nil, fmt.Errorf("LPC analysis failed: %w", err)
	}

	// Step 3: Find formants from LPC spectral envelope
	formants, err := f.findFormantsFromLPC(lpcResult)
	if err != nil {
		return nil, fmt.Errorf("formant extraction failed: %w", err)
	}

	// Step 4: Post-process and validate formants
	validFormants := f.validateFormants(formants)

	// Step 5: Estimate vocal tract length
	vtl := f.estimateVocalTractLength(validFormants)

	// Step 6: Calculate analysis quality
	quality := f.calculateAnalysisQuality(validFormants, lpcResult)

	return &FormantResult{
		Formants:         validFormants,
		VocalTractLength: vtl,
		Quality:          quality,
		LPCOrder:         f.lpcOrder,
		NumFormants:      len(validFormants),
	}, nil
}

// preprocessSignal applies pre-emphasis and windowing
func (f *FormantAnalyzer) preprocessSignal(signal []float64) []float64 {
	// Ensure we have the right length
	length := minInt(len(signal), f.windowSize)
	windowed := make([]float64, length)

	// Apply pre-emphasis filter: y[n] = x[n] - α*x[n-1]
	windowed[0] = signal[0]
	for i := 1; i < length; i++ {
		windowed[i] = signal[i] - f.preEmphasis*signal[i-1]
	}

	// Apply Hamming window
	windowCoeffs := f.window.GetCoefficients()
	for i := range length {
		windowIdx := i * len(windowCoeffs) / length
		if windowIdx < len(windowCoeffs) {
			windowed[i] *= windowCoeffs[windowIdx]
		}
	}

	return windowed
}

// findFormantsFromLPC extracts formants from LPC analysis
func (f *FormantAnalyzer) findFormantsFromLPC(lpcResult *LPCResult) ([]FormantData, error) {
	// Get LPC spectral envelope
	nfft := 1024
	envelope, err := f.lpcAnalyzer.GetSpectralEnvelope(lpcResult.Coefficients, nfft)
	if err != nil {
		return nil, err
	}

	// Find peaks in the spectral envelope
	peaks := f.findSpectralPeaks(envelope)

	// Convert peak indices to frequencies and extract formant data
	var formants []FormantData
	freqResolution := float64(f.sampleRate) / float64(nfft)

	for _, peakIdx := range peaks {
		frequency := float64(peakIdx) * freqResolution

		// Skip if outside valid formant range
		if frequency < f.minFreq || frequency > f.maxFreq {
			continue
		}

		// Estimate formant properties
		amplitude := envelope[peakIdx]
		bandwidth := f.estimateFormantBandwidth(envelope, peakIdx, freqResolution)
		confidence := f.calculateFormantConfidence(frequency, amplitude, bandwidth)

		formants = append(formants, FormantData{
			Frequency:  frequency,
			Bandwidth:  bandwidth,
			Amplitude:  amplitude,
			Confidence: confidence,
		})
	}

	// Sort by frequency and limit to maxFormants
	sort.Slice(formants, func(i, j int) bool {
		return formants[i].Frequency < formants[j].Frequency
	})

	if len(formants) > f.maxFormants {
		formants = formants[:f.maxFormants]
	}

	return formants, nil
}

// findSpectralPeaks finds peaks in the LPC spectral envelope
func (f *FormantAnalyzer) findSpectralPeaks(envelope []float64) []int {
	if len(envelope) < 3 {
		return []int{}
	}

	var peaks []int
	minPeakHeight := 0.1 // Minimum relative peak height

	// Find maximum value for normalization
	maxVal := 0.0
	for _, val := range envelope {
		if val > maxVal {
			maxVal = val
		}
	}

	if maxVal == 0 {
		return peaks
	}

	// Find local maxima
	for i := 1; i < len(envelope)-1; i++ {
		// Check if this is a local maximum
		if envelope[i] > envelope[i-1] && envelope[i] > envelope[i+1] {
			// Check if peak is significant enough
			if envelope[i]/maxVal > minPeakHeight {
				peaks = append(peaks, i)
			}
		}
	}

	return peaks
}

// estimateFormantBandwidth estimates bandwidth of a formant peak
func (f *FormantAnalyzer) estimateFormantBandwidth(envelope []float64, peakIdx int, freqResolution float64) float64 {
	if peakIdx < 0 || peakIdx >= len(envelope) {
		return 100.0 // Default bandwidth
	}

	peakHeight := envelope[peakIdx]
	halfHeight := peakHeight / 2.0

	// Find left and right boundaries at half-height
	leftIdx := peakIdx
	rightIdx := peakIdx

	// Search left
	for i := peakIdx - 1; i >= 0; i-- {
		if envelope[i] <= halfHeight {
			leftIdx = i
			break
		}
	}

	// Search right
	for i := peakIdx + 1; i < len(envelope); i++ {
		if envelope[i] <= halfHeight {
			rightIdx = i
			break
		}
	}

	// Calculate bandwidth in Hz
	bandwidth := float64(rightIdx-leftIdx) * freqResolution

	// Ensure reasonable bandwidth limits
	if bandwidth < 50.0 {
		bandwidth = 50.0
	} else if bandwidth > 500.0 {
		bandwidth = 500.0
	}

	return bandwidth
}

// calculateFormantConfidence estimates confidence in formant detection
func (f *FormantAnalyzer) calculateFormantConfidence(frequency, amplitude, bandwidth float64) float64 {
	confidence := 1.0

	// Frequency-based confidence (higher for typical formant ranges)
	if frequency >= 300 && frequency <= 3500 {
		confidence *= 1.0
	} else if frequency >= 100 && frequency <= 5000 {
		confidence *= 0.7
	} else {
		confidence *= 0.3
	}

	// Amplitude-based confidence (higher amplitude = higher confidence)
	amplitudeConfidence := math.Min(amplitude, 1.0)
	confidence *= amplitudeConfidence

	// Bandwidth-based confidence (reasonable bandwidth = higher confidence)
	if bandwidth >= 50 && bandwidth <= 300 {
		confidence *= 1.0
	} else if bandwidth >= 30 && bandwidth <= 500 {
		confidence *= 0.8
	} else {
		confidence *= 0.5
	}

	return math.Max(0.0, math.Min(1.0, confidence))
}

// validateFormants removes unrealistic formants and ensures proper ordering
func (f *FormantAnalyzer) validateFormants(formants []FormantData) []FormantData {
	var valid []FormantData

	for _, formant := range formants {
		// Basic validation criteria
		if formant.Frequency < f.minFreq || formant.Frequency > f.maxFreq {
			continue
		}

		if formant.Confidence < 0.2 { // Minimum confidence threshold
			continue
		}

		if formant.Bandwidth <= 0 || formant.Bandwidth > 1000 {
			continue
		}

		valid = append(valid, formant)
	}

	// Ensure formants are properly ordered and spaced
	if len(valid) > 1 {
		valid = f.ensureProperSpacing(valid)
	}

	return valid
}

// ensureProperSpacing ensures formants are properly spaced
func (f *FormantAnalyzer) ensureProperSpacing(formants []FormantData) []FormantData {
	if len(formants) <= 1 {
		return formants
	}

	var spaced []FormantData
	spaced = append(spaced, formants[0])

	minSpacing := 200.0 // Minimum spacing between formants (Hz)

	for i := 1; i < len(formants); i++ {
		// Check spacing from last added formant
		lastFormant := spaced[len(spaced)-1]
		if formants[i].Frequency-lastFormant.Frequency >= minSpacing {
			spaced = append(spaced, formants[i])
		} else {
			// Keep the one with higher confidence
			if formants[i].Confidence > lastFormant.Confidence {
				spaced[len(spaced)-1] = formants[i]
			}
		}
	}

	return spaced
}

// estimateVocalTractLength estimates vocal tract length from formant frequencies
func (f *FormantAnalyzer) estimateVocalTractLength(formants []FormantData) float64 {
	if len(formants) == 0 {
		return 17.5 // Average adult vocal tract length (cm)
	}

	// Use the relationship: VTL ≈ (2n-1) * c / (4 * Fn)
	// where c = speed of sound (35000 cm/s), n = formant number
	const speedOfSound = 35000.0 // cm/s

	totalVTL := 0.0
	count := 0

	for i, formant := range formants {
		if formant.Frequency > 0 && formant.Confidence > 0.3 {
			n := float64(i + 1) // Formant number (1-based)
			vtl := (2*n - 1) * speedOfSound / (4 * formant.Frequency)

			// Only use reasonable values
			if vtl >= 10.0 && vtl <= 25.0 {
				totalVTL += vtl
				count++
			}
		}
	}

	if count > 0 {
		return totalVTL / float64(count)
	}

	// Gender-based estimates if no good formants found
	// This is a fallback - in practice you might use other cues
	return 17.5 // Average
}

// calculateAnalysisQuality estimates overall quality of formant analysis
func (f *FormantAnalyzer) calculateAnalysisQuality(formants []FormantData, lpcResult *LPCResult) float64 {
	if len(formants) == 0 {
		return 0.0
	}

	// Quality based on number of formants found
	formantCountQuality := math.Min(float64(len(formants))/3.0, 1.0)

	// Quality based on average confidence
	avgConfidence := 0.0
	for _, formant := range formants {
		avgConfidence += formant.Confidence
	}
	avgConfidence /= float64(len(formants))

	// Quality based on LPC analysis (lower residual energy = better)
	lpcQuality := 1.0
	if lpcResult.ResidualEnergy > 0 {
		// Normalize residual energy (this is a heuristic)
		lpcQuality = math.Max(0.0, 1.0-math.Min(1.0, lpcResult.ResidualEnergy))
	}

	// Quality based on filter stability
	stabilityQuality := 0.0
	if lpcResult.StabilityCheck {
		stabilityQuality = 1.0
	}

	// Combine all quality measures
	return (formantCountQuality + avgConfidence + lpcQuality + stabilityQuality) / 4.0
}

// AnalyzeMultipleFrames analyzes formants across multiple frames
func (f *FormantAnalyzer) AnalyzeMultipleFrames(signal []float64, frameSize, hopSize int) ([]FormantResult, error) {
	if frameSize <= 0 {
		frameSize = f.windowSize
	}
	if hopSize <= 0 {
		hopSize = frameSize / 2
	}

	var results []FormantResult

	for i := 0; i < len(signal)-frameSize; i += hopSize {
		frame := signal[i : i+frameSize]

		result, err := f.AnalyzeFormants(frame)
		if err != nil {
			// Skip this frame but continue with others
			continue
		}

		results = append(results, *result)
	}

	return results, nil
}

// GetFormantFrequencies extracts just the formant frequencies for simple use
func (f *FormantAnalyzer) GetFormantFrequencies(signal []float64) ([]float64, error) {
	result, err := f.AnalyzeFormants(signal)
	if err != nil {
		return nil, err
	}

	frequencies := make([]float64, len(result.Formants))
	for i, formant := range result.Formants {
		frequencies[i] = formant.Frequency
	}

	return frequencies, nil
}
