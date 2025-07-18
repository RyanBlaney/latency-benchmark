package speech

import (
	"fmt"
	"math"
)

// SpeechAnalyzer provides comprehensive speech analysis capabilities
// WHY: Speech analysis requires multiple specialized algorithms working together
// to extract linguistic, prosodic, and speaker characteristics from audio
type SpeechAnalyzer struct {
	sampleRate           int
	formantAnalyzer      *FormantAnalyzer
	voiceQualityAnalyzer *VoiceQualityAnalyzer
	lpcAnalyzer          *LPCAnalyzer
}

// SpeechAnalysisResult contains comprehensive speech analysis results
type SpeechAnalysisResult struct {
	// Formant analysis
	FormantResult *FormantResult `json:"formant_result,omitempty"`

	// Voice quality analysis
	VoiceQualityResult *VoiceQualityResult `json:"voice_quality_result,omitempty"`

	// LPC analysis (if requested separately)
	LPCResult *LPCResult `json:"lpc_result,omitempty"`

	// Overall speech characteristics
	IsSpeech        bool    `json:"is_speech"`       // Whether input appears to be speech
	SpeechQuality   float64 `json:"speech_quality"`  // Overall speech quality score
	Intelligibility float64 `json:"intelligibility"` // Estimated intelligibility

	// Analysis metadata
	AnalysisDuration float64 `json:"analysis_duration"` // Processing time (ms)
	SignalLength     float64 `json:"signal_length"`     // Input signal length (seconds)
}

// NewSpeechAnalyzer creates a new comprehensive speech analyzer
func NewSpeechAnalyzer(sampleRate int) *SpeechAnalyzer {
	return &SpeechAnalyzer{
		sampleRate:           sampleRate,
		formantAnalyzer:      NewFormantAnalyzer(sampleRate),
		voiceQualityAnalyzer: NewVoiceQualityAnalyzer(sampleRate),
		lpcAnalyzer:          NewLPCAnalyzer(sampleRate, 0), // 0 = auto-determine order
	}
}

// AnalyzeSpeech performs comprehensive speech analysis
func (sa *SpeechAnalyzer) AnalyzeSpeech(signal []float64) (*SpeechAnalysisResult, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("empty signal provided")
	}

	result := &SpeechAnalysisResult{
		SignalLength: float64(len(signal)) / float64(sa.sampleRate),
	}

	// Quick speech detection check
	isSpeech := sa.detectSpeech(signal)
	result.IsSpeech = isSpeech

	if !isSpeech {
		// If not speech, return early with minimal analysis
		result.SpeechQuality = 0.0
		result.Intelligibility = 0.0
		return result, nil
	}

	// Perform formant analysis
	formantResult, err := sa.formantAnalyzer.AnalyzeFormants(signal)
	if err == nil {
		result.FormantResult = formantResult
	}

	// Perform voice quality analysis
	voiceQualityResult, err := sa.voiceQualityAnalyzer.AnalyzeVoiceQuality(signal)
	if err == nil {
		result.VoiceQualityResult = voiceQualityResult
	}

	// Calculate overall speech characteristics
	result.SpeechQuality = sa.calculateOverallSpeechQuality(result)
	result.Intelligibility = sa.estimateIntelligibility(result)

	return result, nil
}

// AnalyzeFormantsOnly performs only formant analysis (faster for basic needs)
func (sa *SpeechAnalyzer) AnalyzeFormantsOnly(signal []float64) (*FormantResult, error) {
	return sa.formantAnalyzer.AnalyzeFormants(signal)
}

// AnalyzeVoiceQualityOnly performs only voice quality analysis
func (sa *SpeechAnalyzer) AnalyzeVoiceQualityOnly(signal []float64) (*VoiceQualityResult, error) {
	return sa.voiceQualityAnalyzer.AnalyzeVoiceQuality(signal)
}

// GetFormantFrequencies extracts just formant frequencies (F1, F2, F3, F4)
func (sa *SpeechAnalyzer) GetFormantFrequencies(signal []float64) ([]float64, error) {
	return sa.formantAnalyzer.GetFormantFrequencies(signal)
}

// detectSpeech performs basic speech detection
func (sa *SpeechAnalyzer) detectSpeech(signal []float64) bool {
	if len(signal) < sa.sampleRate/4 { // Less than 250ms
		return false
	}

	// Simple heuristics for speech detection:
	// 1. Check zero crossing rate (speech has moderate ZCR)
	// 2. Check spectral characteristics
	// 3. Check energy distribution

	zcr := sa.calculateZCR(signal)

	// Speech typically has ZCR between 0.02 and 0.2
	if zcr < 0.01 || zcr > 0.3 {
		return false
	}

	// Check energy distribution
	energy := sa.calculateRMSEnergy(signal)
	if energy < 0.001 { // Too quiet
		return false
	}

	// Check for periodicity (simple autocorrelation check)
	hasPeriodicity := sa.checkPeriodicity(signal)

	return hasPeriodicity
}

// calculateZCR computes zero crossing rate
func (sa *SpeechAnalyzer) calculateZCR(signal []float64) float64 {
	if len(signal) <= 1 {
		return 0
	}

	crossings := 0
	for i := 1; i < len(signal); i++ {
		if (signal[i-1] >= 0 && signal[i] < 0) || (signal[i-1] < 0 && signal[i] >= 0) {
			crossings++
		}
	}

	return float64(crossings) / float64(len(signal)-1)
}

// calculateRMSEnergy computes RMS energy
func (sa *SpeechAnalyzer) calculateRMSEnergy(signal []float64) float64 {
	if len(signal) == 0 {
		return 0
	}

	sum := 0.0
	for _, sample := range signal {
		sum += sample * sample
	}

	return math.Sqrt(sum / float64(len(signal)))
}

// checkPeriodicity checks for periodic structure (simple autocorrelation)
func (sa *SpeechAnalyzer) checkPeriodicity(signal []float64) bool {
	if len(signal) < 1024 {
		return false
	}

	// Use a subset for efficiency
	frame := signal[:1024]

	// Simple autocorrelation check
	maxLag := 400 // Corresponds to ~50Hz at 22kHz
	maxCorr := 0.0

	for lag := 20; lag < maxLag && lag < len(frame)/2; lag++ {
		corr := 0.0
		count := 0

		for i := 0; i < len(frame)-lag; i++ {
			corr += frame[i] * frame[i+lag]
			count++
		}

		if count > 0 {
			corr /= float64(count)
			if corr > maxCorr {
				maxCorr = corr
			}
		}
	}

	// Normalize by energy
	energy := 0.0
	for _, sample := range frame {
		energy += sample * sample
	}
	energy /= float64(len(frame))

	if energy > 0 {
		maxCorr /= energy
	}

	// If we find significant periodicity, likely speech
	return maxCorr > 0.1
}

// calculateOverallSpeechQuality computes overall speech quality
func (sa *SpeechAnalyzer) calculateOverallSpeechQuality(result *SpeechAnalysisResult) float64 {
	if !result.IsSpeech {
		return 0.0
	}

	quality := 0.5 // Base quality for detected speech

	// Incorporate formant analysis quality
	if result.FormantResult != nil {
		quality = (quality + result.FormantResult.Quality) / 2.0
	}

	// Incorporate voice quality
	if result.VoiceQualityResult != nil {
		quality = (quality + result.VoiceQualityResult.OverallQuality) / 2.0
	}

	return quality
}

// estimateIntelligibility estimates speech intelligibility
func (sa *SpeechAnalyzer) estimateIntelligibility(result *SpeechAnalysisResult) float64 {
	if !result.IsSpeech {
		return 0.0
	}

	intelligibility := 0.5 // Base intelligibility

	// Formant clarity contributes to intelligibility
	if result.FormantResult != nil && len(result.FormantResult.Formants) >= 2 {
		// Check F1-F2 separation (important for vowel distinction)
		f1 := result.FormantResult.Formants[0].Frequency
		f2 := result.FormantResult.Formants[1].Frequency

		if f2 > f1 {
			separation := f2 - f1
			if separation > 500 { // Good F1-F2 separation
				intelligibility += 0.2
			}
		}

		// Formant quality contributes
		intelligibility = (intelligibility + result.FormantResult.Quality) / 2.0
	}

	// Voice quality contributes (less jitter/shimmer = more intelligible)
	if result.VoiceQualityResult != nil {
		// Good HNR improves intelligibility
		if result.VoiceQualityResult.HNR > 10 {
			intelligibility += 0.1
		}

		// Low jitter/shimmer improves intelligibility
		if result.VoiceQualityResult.Jitter < 2.0 && result.VoiceQualityResult.Shimmer < 5.0 {
			intelligibility += 0.1
		}
	}

	return math.Min(1.0, intelligibility)
}

// EstimateGender provides basic gender estimation from formants
func (sa *SpeechAnalyzer) EstimateGender(signal []float64) (string, float64, error) {
	formantResult, err := sa.formantAnalyzer.AnalyzeFormants(signal)
	if err != nil {
		return "unknown", 0.0, err
	}

	if len(formantResult.Formants) < 2 {
		return "unknown", 0.0, fmt.Errorf("insufficient formants for gender estimation")
	}

	f1 := formantResult.Formants[0].Frequency
	f2 := formantResult.Formants[1].Frequency

	// Simple gender classification based on F1/F2
	// TODO: maybe classify by vocal range
	// These are rough thresholds and would need refinement for production use
	if f1 < 450 && f2 < 2200 {
		return "male", 0.7, nil
	} else if f1 > 500 && f2 > 2400 {
		return "female", 0.7, nil
	} else {
		return "unknown", 0.3, nil
	}
}

// EstimateAge provides basic age estimation from voice characteristics
func (sa *SpeechAnalyzer) EstimateAge(signal []float64) (string, float64, error) {
	voiceResult, err := sa.voiceQualityAnalyzer.AnalyzeVoiceQuality(signal)
	if err != nil {
		return "unknown", 0.0, err
	}

	// Very simple age estimation based on voice quality
	// This is highly simplified and would need much more sophisticated modeling
	if voiceResult.Jitter > 3.0 || voiceResult.Shimmer > 8.0 {
		return "elderly", 0.4, nil // Higher perturbation often indicates older voice
	} else if voiceResult.MeanF0 > 200 && voiceResult.F0Range > 100 {
		return "young", 0.4, nil // Higher, more variable F0 might indicate younger speaker
	} else {
		return "adult", 0.3, nil
	}
}
