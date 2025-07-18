package speech

import (
	"fmt"
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/tonal"
)

// VoiceQualityAnalyzer analyzes voice quality characteristics
// WHY: Jitter, shimmer, and other voice quality measures are important for
// speaker identification, health assessment, emotion recognition, and audio quality
type VoiceQualityAnalyzer struct {
	sampleRate    int
	minF0         float64 // Minimum F0 for analysis
	maxF0         float64 // Maximum F0 for analysis
	pitchDetector *tonal.PitchDetector
}

// VoiceQualityResult contains voice quality measurements
type VoiceQualityResult struct {
	// Perturbation measures
	Jitter  float64 `json:"jitter"`  // Pitch period irregularity (%)
	Shimmer float64 `json:"shimmer"` // Amplitude irregularity (%)

	// Noise measures
	HNR          float64 `json:"hnr"`           // Harmonic-to-noise ratio (dB)
	NoiseMeasure float64 `json:"noise_measure"` // Overall noise level

	// Stability measures
	F0Stability        float64 `json:"f0_stability"`        // Fundamental frequency stability
	AmplitudeStability float64 `json:"amplitude_stability"` // Amplitude stability

	// Overall measures
	VoicingStrength float64 `json:"voicing_strength"` // Overall voicing strength
	OverallQuality  float64 `json:"overall_quality"`  // Composite quality score

	// Analysis metadata
	NumPeriods      int     `json:"num_periods"`      // Number of pitch periods analyzed
	MeanF0          float64 `json:"mean_f0"`          // Mean fundamental frequency
	F0Range         float64 `json:"f0_range"`         // F0 range (max - min)
	AnalysisQuality float64 `json:"analysis_quality"` // Quality of the analysis
}

// NewVoiceQualityAnalyzer creates a new voice quality analyzer
func NewVoiceQualityAnalyzer(sampleRate int) *VoiceQualityAnalyzer {
	return &VoiceQualityAnalyzer{
		sampleRate:    sampleRate,
		minF0:         50.0,  // Minimum F0 (Hz)
		maxF0:         500.0, // Maximum F0 (Hz)
		pitchDetector: tonal.NewPitchDetector(sampleRate),
	}
}

// AnalyzeVoiceQuality performs comprehensive voice quality analysis
func (vqa *VoiceQualityAnalyzer) AnalyzeVoiceQuality(signal []float64) (*VoiceQualityResult, error) {
	if len(signal) < vqa.sampleRate { // Need at least 1 second
		return nil, fmt.Errorf("signal too short for voice quality analysis (need at least 1 second)")
	}

	// Extract pitch periods
	periods, f0Values, err := vqa.extractPitchPeriodsAndF0(signal)
	if err != nil {
		return nil, fmt.Errorf("pitch period extraction failed: %w", err)
	}

	if len(periods) < 3 {
		return nil, fmt.Errorf("insufficient pitch periods for analysis (found %d, need at least 3)", len(periods))
	}

	// Calculate jitter (pitch period irregularity)
	jitter := vqa.calculateJitter(periods)

	// Calculate shimmer (amplitude irregularity)
	shimmer := vqa.calculateShimmer(periods)

	// Calculate harmonic-to-noise ratio
	hnr := vqa.calculateHNR(signal, f0Values)

	// Calculate stability measures
	f0Stability := vqa.calculateF0Stability(f0Values)
	ampStability := vqa.calculateAmplitudeStability(periods)

	// Calculate overall measures
	voicingStrength := vqa.calculateVoicingStrength(signal)
	noiseMeasure := vqa.calculateNoiseMeasure(signal)

	// Calculate F0 statistics
	meanF0, f0Range := vqa.calculateF0Statistics(f0Values)

	// Calculate overall quality
	overallQuality := vqa.calculateOverallQuality(jitter, shimmer, hnr, f0Stability)

	// Calculate analysis quality
	analysisQuality := vqa.calculateAnalysisQuality(len(periods), f0Stability, hnr)

	return &VoiceQualityResult{
		Jitter:             jitter,
		Shimmer:            shimmer,
		HNR:                hnr,
		NoiseMeasure:       noiseMeasure,
		F0Stability:        f0Stability,
		AmplitudeStability: ampStability,
		VoicingStrength:    voicingStrength,
		OverallQuality:     overallQuality,
		NumPeriods:         len(periods),
		MeanF0:             meanF0,
		F0Range:            f0Range,
		AnalysisQuality:    analysisQuality,
	}, nil
}

// extractPitchPeriodsAndF0 extracts pitch periods and F0 values
func (vqa *VoiceQualityAnalyzer) extractPitchPeriodsAndF0(signal []float64) ([][]float64, []float64, error) {
	frameSize := 1024
	hopSize := 256

	var periods [][]float64
	var f0Values []float64
	var lastPeriodEnd int

	// Analyze signal frame by frame to detect periods
	for i := 0; i < len(signal)-frameSize; i += hopSize {
		frame := signal[i : i+frameSize]

		// Detect pitch for this frame
		pitchResult, err := vqa.pitchDetector.DetectPitch(frame)
		if err != nil {
			continue
		}

		// Only process voiced frames with reliable pitch
		if pitchResult.Voicing > 0.5 && pitchResult.Confidence > 0.5 {
			f0 := pitchResult.Pitch
			if f0 >= vqa.minF0 && f0 <= vqa.maxF0 {
				// Calculate expected period length in samples
				periodLength := int(float64(vqa.sampleRate) / f0)

				// Extract period starting from current position
				periodStart := i
				if periodStart < lastPeriodEnd {
					periodStart = lastPeriodEnd
				}

				periodEnd := periodStart + periodLength
				if periodEnd < len(signal) {
					period := make([]float64, periodLength)
					copy(period, signal[periodStart:periodEnd])

					periods = append(periods, period)
					f0Values = append(f0Values, f0)
					lastPeriodEnd = periodEnd
				}
			}
		}
	}

	return periods, f0Values, nil
}

// calculateJitter computes pitch period irregularity
func (vqa *VoiceQualityAnalyzer) calculateJitter(periods [][]float64) float64 {
	if len(periods) < 2 {
		return 0.0
	}

	// Calculate period lengths
	lengths := make([]float64, len(periods))
	for i, period := range periods {
		lengths[i] = float64(len(period))
	}

	// Calculate average period length
	avgLength := 0.0
	for _, length := range lengths {
		avgLength += length
	}
	avgLength /= float64(len(lengths))

	// Calculate absolute jitter (average absolute difference between consecutive periods)
	jitterSum := 0.0
	for i := 1; i < len(lengths); i++ {
		diff := math.Abs(lengths[i] - lengths[i-1])
		jitterSum += diff
	}

	if avgLength == 0 {
		return 0.0
	}

	// Return relative jitter as percentage
	return (jitterSum / float64(len(lengths)-1)) / avgLength * 100.0
}

// calculateShimmer computes amplitude irregularity
func (vqa *VoiceQualityAnalyzer) calculateShimmer(periods [][]float64) float64 {
	if len(periods) < 2 {
		return 0.0
	}

	// Calculate RMS amplitude for each period
	amplitudes := make([]float64, len(periods))
	for i, period := range periods {
		rms := 0.0
		for _, sample := range period {
			rms += sample * sample
		}
		amplitudes[i] = math.Sqrt(rms / float64(len(period)))
	}

	// Calculate average amplitude
	avgAmplitude := 0.0
	for _, amp := range amplitudes {
		avgAmplitude += amp
	}
	avgAmplitude /= float64(len(amplitudes))

	// Calculate absolute shimmer
	shimmerSum := 0.0
	for i := 1; i < len(amplitudes); i++ {
		diff := math.Abs(amplitudes[i] - amplitudes[i-1])
		shimmerSum += diff
	}

	if avgAmplitude == 0 {
		return 0.0
	}

	// Return relative shimmer as percentage
	return (shimmerSum / float64(len(amplitudes)-1)) / avgAmplitude * 100.0
}

// calculateHNR computes harmonic-to-noise ratio
func (vqa *VoiceQualityAnalyzer) calculateHNR(signal []float64, f0Values []float64) float64 {
	if len(f0Values) == 0 {
		return 0.0
	}

	// Use mean F0 for HNR calculation
	meanF0 := 0.0
	for _, f0 := range f0Values {
		meanF0 += f0
	}
	meanF0 /= float64(len(f0Values))

	// Calculate autocorrelation-based HNR
	frameSize := 2048
	if len(signal) < frameSize {
		return 0.0
	}

	// Take a representative frame
	startIdx := len(signal)/2 - frameSize/2
	if startIdx < 0 {
		startIdx = 0
	}
	frame := signal[startIdx : startIdx+frameSize]

	// Calculate autocorrelation
	autocorr := make([]float64, frameSize)
	for lag := 0; lag < frameSize; lag++ {
		sum := 0.0
		count := 0
		for i := 0; i < frameSize-lag; i++ {
			sum += frame[i] * frame[i+lag]
			count++
		}
		if count > 0 {
			autocorr[lag] = sum / float64(count)
		}
	}

	// Find peak corresponding to fundamental period
	expectedLag := int(float64(vqa.sampleRate) / meanF0)
	if expectedLag >= len(autocorr) {
		return 0.0
	}

	// Find maximum in expected range
	maxCorr := 0.0
	searchRange := expectedLag / 4 // Search ±25% around expected lag
	startSearch := maxInt(1, expectedLag-searchRange)
	endSearch := minInt(len(autocorr)-1, expectedLag+searchRange)

	for i := startSearch; i <= endSearch; i++ {
		if autocorr[i] > maxCorr {
			maxCorr = autocorr[i]
		}
	}

	// Calculate HNR in dB
	if maxCorr > 0 && maxCorr < autocorr[0] {
		ratio := maxCorr / (autocorr[0] - maxCorr)
		return 10 * math.Log10(ratio)
	}

	return 0.0
}

// calculateF0Stability measures fundamental frequency stability
func (vqa *VoiceQualityAnalyzer) calculateF0Stability(f0Values []float64) float64 {
	if len(f0Values) < 2 {
		return 0.0
	}

	// Calculate coefficient of variation
	mean := 0.0
	for _, f0 := range f0Values {
		mean += f0
	}
	mean /= float64(len(f0Values))

	variance := 0.0
	for _, f0 := range f0Values {
		diff := f0 - mean
		variance += diff * diff
	}
	variance /= float64(len(f0Values))

	if mean == 0 {
		return 0.0
	}

	cv := math.Sqrt(variance) / mean
	return math.Max(0.0, 1.0-cv) // Higher values = more stable
}

// calculateAmplitudeStability measures amplitude stability across periods
func (vqa *VoiceQualityAnalyzer) calculateAmplitudeStability(periods [][]float64) float64 {
	if len(periods) < 2 {
		return 0.0
	}

	// Calculate RMS for each period
	amplitudes := make([]float64, len(periods))
	for i, period := range periods {
		rms := 0.0
		for _, sample := range period {
			rms += sample * sample
		}
		amplitudes[i] = math.Sqrt(rms / float64(len(period)))
	}

	// Calculate coefficient of variation
	mean := 0.0
	for _, amp := range amplitudes {
		mean += amp
	}
	mean /= float64(len(amplitudes))

	variance := 0.0
	for _, amp := range amplitudes {
		diff := amp - mean
		variance += diff * diff
	}
	variance /= float64(len(amplitudes))

	if mean == 0 {
		return 0.0
	}

	cv := math.Sqrt(variance) / mean
	return math.Max(0.0, 1.0-cv) // Higher values = more stable
}

// calculateVoicingStrength estimates overall voicing strength
func (vqa *VoiceQualityAnalyzer) calculateVoicingStrength(signal []float64) float64 {
	// Use pitch detection to estimate voicing strength
	pitchResult, err := vqa.pitchDetector.DetectPitch(signal)
	if err != nil {
		return 0.0
	}

	return pitchResult.Voicing
}

// calculateNoiseMeasure estimates overall noise level
func (vqa *VoiceQualityAnalyzer) calculateNoiseMeasure(signal []float64) float64 {
	// Simple noise measure based on high-frequency content
	if len(signal) < 1024 {
		return 0.0
	}

	// Calculate energy in high vs low frequencies (simplified)
	frame := signal[:1024]

	// Apply simple high-pass filtering (difference filter)
	highFreqEnergy := 0.0
	totalEnergy := 0.0

	for i := 1; i < len(frame); i++ {
		diff := frame[i] - frame[i-1]
		highFreqEnergy += diff * diff
		totalEnergy += frame[i] * frame[i]
	}

	if totalEnergy == 0 {
		return 0.0
	}

	return highFreqEnergy / totalEnergy
}

// calculateF0Statistics computes F0 statistics
func (vqa *VoiceQualityAnalyzer) calculateF0Statistics(f0Values []float64) (float64, float64) {
	if len(f0Values) == 0 {
		return 0.0, 0.0
	}

	// Calculate mean
	mean := 0.0
	for _, f0 := range f0Values {
		mean += f0
	}
	mean /= float64(len(f0Values))

	// Calculate range
	minF0 := f0Values[0]
	maxF0 := f0Values[0]
	for _, f0 := range f0Values {
		if f0 < minF0 {
			minF0 = f0
		}
		if f0 > maxF0 {
			maxF0 = f0
		}
	}

	return mean, maxF0 - minF0
}

// calculateOverallQuality computes overall voice quality score
func (vqa *VoiceQualityAnalyzer) calculateOverallQuality(jitter, shimmer, hnr, f0Stability float64) float64 {
	// Normalize measures (lower jitter/shimmer = better, higher HNR/stability = better)
	jitterScore := math.Max(0, 1.0-jitter/5.0)       // 5% jitter = poor quality
	shimmerScore := math.Max(0, 1.0-shimmer/10.0)    // 10% shimmer = poor quality
	hnrScore := math.Min(1.0, math.Max(0, hnr/20.0)) // 20dB HNR = excellent
	stabilityScore := f0Stability                    // Already normalized 0-1

	return (jitterScore + shimmerScore + hnrScore + stabilityScore) / 4.0
}

// calculateAnalysisQuality estimates quality of the analysis itself
func (vqa *VoiceQualityAnalyzer) calculateAnalysisQuality(numPeriods int, f0Stability, hnr float64) float64 {
	// Quality based on number of periods
	periodQuality := math.Min(1.0, float64(numPeriods)/10.0) // 10+ periods = good

	// Quality based on F0 stability (stable F0 = reliable analysis)
	stabilityQuality := f0Stability

	// Quality based on HNR (higher HNR = cleaner signal = more reliable)
	hnrQuality := math.Min(1.0, math.Max(0, hnr/15.0)) // 15dB HNR = good quality

	return (periodQuality + stabilityQuality + hnrQuality) / 3.0
}

// Helper functions
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
