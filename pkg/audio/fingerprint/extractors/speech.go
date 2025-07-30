package extractors

import (
	"fmt"
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/filters"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/spectral"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/speech"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/temporal"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/algorithms/tonal"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
)

// SpeechFeatureExtractor extracts features optimized for talk/news content
// WHY: Speech content requires specialized analysis focusing on vocal characteristics,
// intelligibility, speaker properties, and linguistic content rather than musical features
type SpeechFeatureExtractor struct {
	config *config.FeatureConfig
	logger logging.Logger
	isNews bool

	// Speech-specific algorithms
	speechAnalyzer       *speech.SpeechAnalyzer
	formantAnalyzer      *speech.FormantAnalyzer
	voiceQualityAnalyzer *speech.VoiceQualityAnalyzer

	// Core spectral analysis algorithms
	spectralCentroid  *spectral.SpectralCentroid
	spectralRolloff   *spectral.SpectralRolloff
	spectralBandwidth *spectral.SpectralBandwidth
	spectralFlatness  *spectral.SpectralFlatness
	spectralCrest     *spectral.SpectralCrest
	spectralSlope     *spectral.SpectralSlope
	spectralFlux      *spectral.SpectralFlux
	zeroCrossing      *spectral.ZeroCrossingRate

	// Temporal analysis algorithms
	energy           *temporal.Energy
	envelope         *temporal.Envelope
	onsetDetection   *temporal.OnsetDetection
	silenceDetection *temporal.SilenceDetection
	dynamicRange     *temporal.DynamicRange

	// Tonal analysis for speech
	pitchDetector *tonal.PitchDetector

	// Audio preprocessing
	preEmphasis *filters.PreEmphasis
	mfcc        *spectral.MFCC
}

// NewSpeechFeatureExtractor creates a speech-specific feature extractor
// WHY: Speech analysis needs different parameter tuning and feature emphasis
// compared to music - formants, voice quality, and intelligibility are priorities
func NewSpeechFeatureExtractor(config *config.FeatureConfig, isNews bool) *SpeechFeatureExtractor {
	logger := logging.WithFields(logging.Fields{
		"component": "speech_feature_extractor",
		"is_news":   isNews,
	})

	return &SpeechFeatureExtractor{
		config: config,
		logger: logger,
		isNews: isNews,

		// Initialize speech-specific algorithms
		speechAnalyzer:       speech.NewSpeechAnalyzer(config.SampleRate),
		formantAnalyzer:      speech.NewFormantAnalyzer(config.SampleRate),
		voiceQualityAnalyzer: speech.NewVoiceQualityAnalyzer(config.SampleRate),

		// Initialize core spectral algorithms
		spectralCentroid:  spectral.NewSpectralCentroid(config.SampleRate),
		spectralRolloff:   spectral.NewSpectralRolloff(config.SampleRate),
		spectralBandwidth: spectral.NewSpectralBandwidth(config.SampleRate),
		spectralFlatness:  spectral.NewSpectralFlatness(),
		spectralCrest:     spectral.NewSpectralCrest(),
		spectralSlope:     spectral.NewSpectralSlope(config.SampleRate),
		spectralFlux:      spectral.NewSpectralFlux(),
		zeroCrossing:      spectral.NewZeroCrossingRate(config.SampleRate),

		// Initialize temporal algorithms
		energy:           temporal.NewEnergy(config.WindowSize, config.HopSize, config.SampleRate),
		envelope:         temporal.NewEnvelope(),
		onsetDetection:   temporal.NewOnsetDetection(),
		silenceDetection: temporal.NewSilenceDetection(),
		dynamicRange:     temporal.NewDynamicRange(),

		// Initialize tonal algorithms
		pitchDetector: tonal.NewPitchDetector(config.SampleRate),

		// Initialize preprocessing
		preEmphasis: filters.NewPreEmphasisForContent("speech", config.SampleRate),
		mfcc:        spectral.NewMFCC(config.SampleRate, config.MFCCCoefficients),
	}
}

func (s *SpeechFeatureExtractor) GetName() string {
	return "SpeechFeatureExtractor"
}

func (s *SpeechFeatureExtractor) GetContentType() config.ContentType {
	if s.isNews {
		return config.ContentNews
	}
	return config.ContentTalk
}

func (s *SpeechFeatureExtractor) GetFeatureWeights() map[string]float64 {
	if s.config.SimilarityWeights != nil {
		return s.config.SimilarityWeights
	}

	// Default weights optimized for speech content
	// WHY: Speech analysis prioritizes MFCC and speech-specific features
	// over musical characteristics like harmony and rhythm
	weights := map[string]float64{
		"mfcc":     0.40, // High weight - critical for speech recognition
		"speech":   0.35, // High weight - formants, voice quality
		"spectral": 0.15, // Medium weight - general spectral shape
		"temporal": 0.10, // Lower weight - less critical than in music
	}

	// News content may benefit from slightly different weighting
	if s.isNews {
		weights["speech"] = 0.40 // Even higher emphasis on speech characteristics
		weights["mfcc"] = 0.35   // Slightly lower MFCC weight
	}

	return weights
}

func (s *SpeechFeatureExtractor) ExtractFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*ExtractedFeatures, error) {
	if spectrogram == nil {
		return nil, fmt.Errorf("spectrogram cannot be nil")
	}
	if len(pcm) == 0 {
		return nil, fmt.Errorf("PCM data cannot be empty")
	}
	if sampleRate <= 0 {
		return nil, fmt.Errorf("sample rate must be positive")
	}

	logger := s.logger.WithFields(logging.Fields{
		"function":              "ExtractFeatures",
		"spectrogram_frames":    spectrogram.TimeFrames,
		"spectrogram_freq_bins": spectrogram.FreqBins,
		"pcm_length":            len(pcm),
		"sample_rate":           sampleRate,
	})

	logger.Debug("Extracting speech features...")

	features := &ExtractedFeatures{
		ExtractionMetadata: make(map[string]any),
	}

	// Step 1: Apply speech-optimized pre-processing
	preprocessedPCM, err := s.preprocessForSpeech(pcm)
	if err != nil {
		logger.Warn("Pre-processing failed, using original PCM", logging.Fields{"error": err})
		preprocessedPCM = pcm
	}

	// Step 2: Extract MFCC features (critical for speech)
	if s.config.EnableMFCC {
		logger.Debug("Extracting MFCC features...")
		mfccFeatures, err := s.extractMFCCFeatures(spectrogram)
		if err != nil {
			logger.Error(err, "Failed to extract MFCC features")
			return nil, fmt.Errorf("MFCC extraction failed: %w", err)
		}
		features.MFCC = mfccFeatures
	}

	// Step 3: Extract speech-specific features (high priority)
	if s.config.EnableSpeechFeatures {
		logger.Debug("Extracting speech-specific features...")
		speechFeatures, err := s.extractSpeechFeatures(preprocessedPCM)
		if err != nil {
			logger.Error(err, "Failed to extract speech features")
			// Don't fail completely - speech features are important but not critical
			logger.Warn("Continuing without speech features")
		} else {
			features.SpeechFeatures = speechFeatures
		}
	}

	// Step 4: Extract spectral features
	logger.Debug("Extracting spectral features...")
	spectralFeatures, err := s.extractSpectralFeatures(spectrogram, preprocessedPCM)
	if err != nil {
		logger.Error(err, "Failed to extract spectral features")
		return nil, fmt.Errorf("spectral feature extraction failed: %w", err)
	}
	features.SpectralFeatures = spectralFeatures

	// Step 5: Extract temporal features
	if s.config.EnableTemporalFeatures {
		logger.Debug("Extracting temporal features...")
		temporalFeatures, err := s.extractTemporalFeatures(preprocessedPCM, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract temporal features")
			// Temporal features are less critical for speech
			logger.Warn("Continuing without temporal features")
		} else {
			features.TemporalFeatures = temporalFeatures
		}
	}

	// Step 6: Extract energy features
	logger.Debug("Extracting energy features...")
	energyFeatures, err := s.extractEnergyFeatures(preprocessedPCM, spectrogram)
	if err != nil {
		logger.Error(err, "Failed to extract energy features")
		return nil, fmt.Errorf("energy feature extraction failed: %w", err)
	}
	features.EnergyFeatures = energyFeatures

	// Step 7: Extract basic harmonic features (for voicing analysis)
	logger.Debug("Extracting harmonic features...")
	harmonicFeatures, err := s.extractHarmonicFeatures(preprocessedPCM)
	if err != nil {
		logger.Warn("Failed to extract harmonic features", logging.Fields{"error": err})
		// Harmonic features are helpful but not essential for speech
	} else {
		features.HarmonicFeatures = harmonicFeatures
	}

	// Add extraction metadata
	features.ExtractionMetadata["extractor_type"] = "speech"
	features.ExtractionMetadata["content_subtype"] = s.getContentSubtype()
	features.ExtractionMetadata["algorithms_used"] = "speech,spectral,temporal,filters,tonal"
	features.ExtractionMetadata["pre_emphasis_applied"] = true
	features.ExtractionMetadata["sample_rate"] = sampleRate
	features.ExtractionMetadata["spectrogram_frames"] = spectrogram.TimeFrames
	features.ExtractionMetadata["optimization"] = "speech_optimized"

	logger.Debug("Speech feature extraction completed successfully")
	return features, nil
}

// preprocessForSpeech applies speech-optimized preprocessing
func (s *SpeechFeatureExtractor) preprocessForSpeech(pcm []float64) ([]float64, error) {
	// Apply pre-emphasis filter optimized for speech
	preprocessed := s.preEmphasis.ProcessBuffer(pcm)
	if len(preprocessed) == 0 {
		return pcm, fmt.Errorf("pre-emphasis processing failed")
	}
	return preprocessed, nil
}

// extractMFCCFeatures extracts MFCC coefficients optimized for speech
func (s *SpeechFeatureExtractor) extractMFCCFeatures(spectrogram *analyzers.SpectrogramResult) ([][]float64, error) {
	mfccFeatures, err := s.mfcc.ComputeFrames(spectrogram.Magnitude)
	if err != nil {
		return nil, fmt.Errorf("MFCC computation failed: %w", err)
	}

	// Validate MFCC features
	if len(mfccFeatures) == 0 {
		return nil, fmt.Errorf("no MFCC features extracted")
	}

	return mfccFeatures, nil
}

// extractSpeechFeatures extracts comprehensive speech characteristics
func (s *SpeechFeatureExtractor) extractSpeechFeatures(pcm []float64) (*SpeechFeatures, error) {
	// Perform comprehensive speech analysis
	speechResult, err := s.speechAnalyzer.AnalyzeSpeech(pcm)
	if err != nil {
		return nil, fmt.Errorf("speech analysis failed: %w", err)
	}

	// Check if input is actually speech
	if !speechResult.IsSpeech {
		return &SpeechFeatures{
			FormantFrequencies: [][]float64{},
			VoicingProbability: []float64{},
			SpectralTilt:       []float64{},
			SpeechRate:         0.0,
			PauseDuration:      []float64{},
			VocalTractLength:   17.5, // Default
			Jitter:             0.0,
			Shimmer:            0.0,
		}, nil
	}

	features := &SpeechFeatures{
		SpeechRate: s.estimateSpeechRate(pcm, speechResult),
	}

	// Extract formant information
	if speechResult.FormantResult != nil {
		features.FormantFrequencies = s.convertFormantData(speechResult.FormantResult.Formants)
		features.VocalTractLength = speechResult.FormantResult.VocalTractLength
	} else {
		features.FormantFrequencies = [][]float64{}
		features.VocalTractLength = 17.5 // Average vocal tract length
	}

	// Extract voice quality information
	if speechResult.VoiceQualityResult != nil {
		features.Jitter = speechResult.VoiceQualityResult.Jitter
		features.Shimmer = speechResult.VoiceQualityResult.Shimmer
	}

	// Extract frame-by-frame features
	features.VoicingProbability = s.extractVoicingProbability(pcm)
	features.SpectralTilt = s.extractSpectralTilt(pcm)
	features.PauseDuration = s.extractPauseDurations(pcm)

	return features, nil
}

// extractSpectralFeatures uses algorithms package for spectral analysis
func (s *SpeechFeatureExtractor) extractSpectralFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64) (*SpectralFeatures, error) {
	features := &SpectralFeatures{
		SpectralCentroid:  make([]float64, spectrogram.TimeFrames),
		SpectralRolloff:   make([]float64, spectrogram.TimeFrames),
		SpectralBandwidth: make([]float64, spectrogram.TimeFrames),
		SpectralFlatness:  make([]float64, spectrogram.TimeFrames),
		SpectralCrest:     make([]float64, spectrogram.TimeFrames),
		SpectralSlope:     make([]float64, spectrogram.TimeFrames),
		ZeroCrossingRate:  make([]float64, spectrogram.TimeFrames),
	}

	frameSize := spectrogram.WindowSize
	hopSize := spectrogram.HopSize

	// Process each frame using algorithms package
	for t := 0; t < spectrogram.TimeFrames; t++ {
		if t >= len(spectrogram.Magnitude) {
			break
		}

		magnitude := spectrogram.Magnitude[t]

		// Compute spectral features using algorithms
		features.SpectralCentroid[t] = s.spectralCentroid.Compute(magnitude)
		features.SpectralRolloff[t] = s.spectralRolloff.Compute(magnitude, 0.85)
		features.SpectralBandwidth[t] = s.spectralBandwidth.Compute(magnitude, features.SpectralCentroid[t])
		features.SpectralFlatness[t] = s.spectralFlatness.Compute(magnitude)
		features.SpectralCrest[t] = s.spectralCrest.Compute(magnitude)
		features.SpectralSlope[t] = s.spectralSlope.Compute(magnitude)

		// Zero crossing rate from PCM
		start := t * hopSize
		end := start + frameSize
		end = min(end, len(pcm))
		if start < len(pcm) {
			frame := pcm[start:end]
			features.ZeroCrossingRate[t] = s.zeroCrossing.Compute(frame)
		}
	}

	// Calculate spectral flux
	if spectrogram.TimeFrames > 1 {
		flux := s.spectralFlux.Compute(spectrogram.Magnitude)
		features.SpectralFlux = flux
	}

	return features, nil
}

// extractTemporalFeatures uses algorithms package for temporal analysis
func (s *SpeechFeatureExtractor) extractTemporalFeatures(pcm []float64, sampleRate int) (*TemporalFeatures, error) {
	features := &TemporalFeatures{}

	// Extract RMS energy using the correct Energy method
	features.RMSEnergy = s.energy.ComputeShortTimeEnergy(pcm)

	// Calculate dynamic range using Energy's loudness range method
	features.DynamicRange = s.energy.ComputeLoudnessRange(pcm)

	// Simple silence detection
	silenceRatio := s.calculateSilenceRatio(pcm)
	features.SilenceRatio = silenceRatio

	// Calculate basic amplitude statistics
	features.PeakAmplitude = 0.0
	sum := 0.0
	for _, sample := range pcm {
		abs := math.Abs(sample)
		if abs > features.PeakAmplitude {
			features.PeakAmplitude = abs
		}
		sum += abs
	}
	if len(pcm) > 0 {
		features.AverageAmplitude = sum / float64(len(pcm))
	}

	// Simple onset detection based on energy changes
	onsets := s.detectOnsets(features.RMSEnergy, s.config.HopSize, sampleRate)
	features.OnsetDensity = float64(len(onsets)) / (float64(len(pcm)) / float64(sampleRate))

	// Extract attack times from energy onsets
	features.AttackTime = s.calculateAttackTimes(onsets, features.RMSEnergy)

	// Extract envelope shape (simplified)
	features.EnvelopeShape = s.extractSimpleEnvelope(pcm)

	return features, nil
}

// extractEnergyFeatures uses algorithms package for energy analysis
func (s *SpeechFeatureExtractor) extractEnergyFeatures(pcm []float64, spectrogram *analyzers.SpectrogramResult) (*EnergyFeatures, error) {
	features := &EnergyFeatures{}

	// Extract frame-based energy using Energy algorithm
	features.ShortTimeEnergy = s.energy.ComputeShortTimeEnergy(pcm)

	// Calculate energy variance using Energy algorithm
	features.EnergyVariance = s.energy.ComputeEnergyVariance(features.ShortTimeEnergy)

	// Calculate loudness range using Energy algorithm
	features.LoudnessRange = s.energy.ComputeLoudnessRange(pcm)

	// Calculate frame-based features
	numFrames := len(features.ShortTimeEnergy)
	features.EnergyEntropy = make([]float64, numFrames)
	features.LowEnergyRatio = make([]float64, numFrames)
	features.HighEnergyRatio = make([]float64, numFrames)

	for i := range numFrames {
		// Energy entropy for this frame (using the energy value directly)
		if features.ShortTimeEnergy[i] > 0 {
			features.EnergyEntropy[i] = -features.ShortTimeEnergy[i] * math.Log(features.ShortTimeEnergy[i]+1e-10)
		}

		// Calculate frequency band energy ratios if spectrogram is available
		if i < len(spectrogram.Magnitude) {
			magnitude := spectrogram.Magnitude[i]
			lowEnergy := 0.0
			highEnergy := 0.0
			totalEnergy := 0.0

			splitPoint := len(magnitude) / 4 // Rough low/high frequency split
			for j, mag := range magnitude {
				energy := mag * mag
				totalEnergy += energy
				if j < splitPoint {
					lowEnergy += energy
				} else {
					highEnergy += energy
				}
			}

			if totalEnergy > 0 {
				features.LowEnergyRatio[i] = lowEnergy / totalEnergy
				features.HighEnergyRatio[i] = highEnergy / totalEnergy
			}
		}
	}

	return features, nil
}

// extractHarmonicFeatures extracts basic harmonic features for speech
func (s *SpeechFeatureExtractor) extractHarmonicFeatures(pcm []float64) (*HarmonicFeatures, error) {
	features := &HarmonicFeatures{}

	// Frame-based pitch detection
	frameSize := 1024
	hopSize := 512
	numFrames := (len(pcm)-frameSize)/hopSize + 1

	features.PitchEstimate = make([]float64, numFrames)
	features.PitchConfidence = make([]float64, numFrames)
	features.VoicingStrength = make([]float64, numFrames)

	for i := range numFrames {
		start := i * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		frame := pcm[start:end]
		pitchResult, err := s.pitchDetector.DetectPitch(frame)
		if err == nil {
			features.PitchEstimate[i] = pitchResult.Pitch
			features.PitchConfidence[i] = pitchResult.Confidence
			features.VoicingStrength[i] = pitchResult.Voicing
		}
	}

	// Simple harmonic ratio estimation (placeholder)
	features.HarmonicRatio = make([]float64, numFrames)
	features.InharmonicityRatio = make([]float64, numFrames)
	features.TonalCentroid = make([]float64, numFrames)

	// These would typically use more sophisticated algorithms
	// For now, provide reasonable estimates based on voicing
	for i := range numFrames {
		// Higher voicing strength suggests more harmonic content
		features.HarmonicRatio[i] = features.VoicingStrength[i] * 10.0 // Convert to dB-like scale
		features.InharmonicityRatio[i] = 1.0 - features.VoicingStrength[i]

		// Tonal centroid approximated from pitch
		if features.PitchEstimate[i] > 0 {
			features.TonalCentroid[i] = features.PitchEstimate[i]
		}
	}

	return features, nil
}

// Helper methods for speech-specific feature extraction

func (s *SpeechFeatureExtractor) convertFormantData(formants []speech.FormantData) [][]float64 {
	if len(formants) == 0 {
		return [][]float64{}
	}

	// Convert to format expected by features: [frame][formant_number]
	// For single-frame analysis, create one frame with all formants
	result := make([][]float64, 1)
	result[0] = make([]float64, len(formants))

	for i, formant := range formants {
		result[0][i] = formant.Frequency
	}

	return result
}

func (s *SpeechFeatureExtractor) extractVoicingProbability(pcm []float64) []float64 {
	frameSize := 1024
	hopSize := 512
	numFrames := (len(pcm)-frameSize)/hopSize + 1

	voicing := make([]float64, numFrames)

	for i := range numFrames {
		start := i * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		frame := pcm[start:end]
		pitchResult, err := s.pitchDetector.DetectPitch(frame)
		if err == nil {
			voicing[i] = pitchResult.Voicing
		}
	}

	return voicing
}

func (s *SpeechFeatureExtractor) extractSpectralTilt(pcm []float64) []float64 {
	frameSize := 1024
	hopSize := 512
	numFrames := (len(pcm)-frameSize)/hopSize + 1

	tilt := make([]float64, numFrames)

	// This would typically require spectral analysis
	// For now, provide a simplified implementation
	for i := range numFrames {
		start := i * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		frame := pcm[start:end]

		// Simple spectral tilt approximation using high-frequency energy
		highFreqEnergy := 0.0
		lowFreqEnergy := 0.0

		// Use difference between samples as proxy for high-frequency content
		for j := 1; j < len(frame); j++ {
			diff := frame[j] - frame[j-1]
			highFreqEnergy += diff * diff
			lowFreqEnergy += frame[j] * frame[j]
		}

		if lowFreqEnergy > 0 {
			tilt[i] = -10 * math.Log10(highFreqEnergy/lowFreqEnergy) // Negative slope
		}
	}

	return tilt
}

func (s *SpeechFeatureExtractor) extractPauseDurations(pcm []float64) []float64 {
	// Simple silence-based pause detection using energy thresholding
	energies := s.energy.ComputeShortTimeEnergy(pcm)
	if len(energies) == 0 {
		return []float64{}
	}

	// Find silence threshold (10th percentile of energy)
	sortedEnergies := make([]float64, len(energies))
	copy(sortedEnergies, energies)
	for i := range len(sortedEnergies) {
		for j := 0; j < len(sortedEnergies)-1-i; j++ {
			if sortedEnergies[j] > sortedEnergies[j+1] {
				sortedEnergies[j], sortedEnergies[j+1] = sortedEnergies[j+1], sortedEnergies[j]
			}
		}
	}

	threshold := sortedEnergies[len(sortedEnergies)/10] // 10th percentile

	// Find pause segments
	var pauseDurations []float64
	inPause := false
	pauseStart := 0
	frameTimeS := float64(s.config.HopSize) / float64(s.config.SampleRate)

	for i, energy := range energies {
		if energy <= threshold {
			if !inPause {
				inPause = true
				pauseStart = i
			}
		} else {
			if inPause {
				pauseDuration := float64(i-pauseStart) * frameTimeS
				if pauseDuration > 0.1 { // Only count pauses longer than 100ms
					pauseDurations = append(pauseDurations, pauseDuration)
				}
				inPause = false
			}
		}
	}

	// Handle case where signal ends in a pause
	if inPause {
		pauseDuration := float64(len(energies)-pauseStart) * frameTimeS
		if pauseDuration > 0.1 {
			pauseDurations = append(pauseDurations, pauseDuration)
		}
	}

	return pauseDurations
}

func (s *SpeechFeatureExtractor) calculateSilenceRatio(pcm []float64) float64 {
	energies := s.energy.ComputeShortTimeEnergy(pcm)
	if len(energies) == 0 {
		return 0.0
	}

	// Find silence threshold (similar to pause detection)
	sortedEnergies := make([]float64, len(energies))
	copy(sortedEnergies, energies)
	for i := range len(sortedEnergies) {
		for j := 0; j < len(sortedEnergies)-1-i; j++ {
			if sortedEnergies[j] > sortedEnergies[j+1] {
				sortedEnergies[j], sortedEnergies[j+1] = sortedEnergies[j+1], sortedEnergies[j]
			}
		}
	}

	threshold := sortedEnergies[len(sortedEnergies)/10] // 10th percentile

	silentFrames := 0
	for _, energy := range energies {
		if energy <= threshold {
			silentFrames++
		}
	}

	return float64(silentFrames) / float64(len(energies))
}

// detectOnsets NOTE: hopSize and sampleRate will be integrated when I decide to
// scale onset indices to time or frequency. Not needed for MVP
func (s *SpeechFeatureExtractor) detectOnsets(energies []float64, hopSize, sampleRate int) []int {
	if len(energies) < 3 {
		return []int{}
	}

	// Calculate energy derivative for onset detection
	derivative := s.energy.ComputeEnergyDerivative(energies)

	// Find peaks in the derivative
	var onsets []int
	threshold := s.calculateAdaptiveThreshold(derivative)

	for i := 1; i < len(derivative)-1; i++ {
		if derivative[i] > derivative[i-1] &&
			derivative[i] > derivative[i+1] &&
			derivative[i] > threshold {
			onsets = append(onsets, i)
		}
	}

	return onsets
}

func (s *SpeechFeatureExtractor) calculateAdaptiveThreshold(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	// Calculate mean and standard deviation
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	stdDev := math.Sqrt(variance / float64(len(values)))

	// Threshold is mean + 2*stddev
	return mean + 2*stdDev
}

func (s *SpeechFeatureExtractor) calculateAttackTimes(onsets []int, energies []float64) []float64 {
	if len(onsets) == 0 {
		return []float64{}
	}

	attackTimes := make([]float64, len(onsets))
	frameTimeS := float64(s.config.HopSize) / float64(s.config.SampleRate)

	for i, onset := range onsets {
		// Simple attack time estimation: time to reach 90% of peak energy
		peakEnergy := energies[onset]
		// TODO: unused
		// targetEnergy := 0.9 * peakEnergy

		// Look backwards from onset to find attack start
		attackStart := onset
		for j := onset - 1; j >= 0 && j > onset-10; j-- {
			if energies[j] < 0.1*peakEnergy {
				attackStart = j
				break
			}
		}

		// Attack time is the duration from start to target energy
		attackTimes[i] = float64(onset-attackStart) * frameTimeS
		if attackTimes[i] > 0.1 { // Clamp to reasonable values
			attackTimes[i] = 0.1
		}
	}

	return attackTimes
}

func (s *SpeechFeatureExtractor) extractSimpleEnvelope(pcm []float64) []float64 {
	// Extract amplitude envelope using sliding window RMS
	windowSize := 512
	hopSize := 256

	if len(pcm) < windowSize {
		return []float64{}
	}

	numFrames := (len(pcm)-windowSize)/hopSize + 1
	envelope := make([]float64, numFrames)

	for i := range numFrames {
		start := i * hopSize
		end := start + windowSize
		end = min(end, len(pcm))

		// Calculate RMS for this window
		sum := 0.0
		for j := start; j < end; j++ {
			sum += pcm[j] * pcm[j]
		}
		envelope[i] = math.Sqrt(sum / float64(end-start))
	}

	return envelope
}

func (s *SpeechFeatureExtractor) estimateSpeechRate(pcm []float64, speechResult *speech.SpeechAnalysisResult) float64 {
	if speechResult == nil || !speechResult.IsSpeech {
		return 0.0
	}

	// Simple speech rate estimation
	signalDuration := float64(len(pcm)) / float64(s.config.SampleRate)

	// Use silence ratio to estimate speech rate
	silenceRatio := s.calculateSilenceRatio(pcm)
	speechTime := signalDuration * (1.0 - silenceRatio)

	if speechTime > 0 {
		// Rough approximation: 3-5 syllables per second for normal speech
		return 4.0 * speechTime / signalDuration // Syllables per second, normalized
	}

	return 3.0 // Default speech rate
}

func (s *SpeechFeatureExtractor) getContentSubtype() string {
	if s.isNews {
		return "news"
	}
	return "talk"
}
