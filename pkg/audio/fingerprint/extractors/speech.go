package extractors

import (
	"fmt"
	"math"
	"sort"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/logging"
)

// SpeechFeatureExtractor extracts features optimized for talk/news content
type SpeechFeatureExtractor struct {
	config *config.FeatureConfig
	logger logging.Logger
	isNews bool
}

// NewSpeechFeatureExtractor creates a speech-specific feature extractor
func NewSpeechFeatureExtractor(config *config.FeatureConfig, isNews bool) *SpeechFeatureExtractor {
	return &SpeechFeatureExtractor{
		config: config,
		logger: logging.WithFields(logging.Fields{
			"component": "speech_feature_extractor",
		}),
		isNews: isNews,
	}
}

func (s *SpeechFeatureExtractor) GetName() string {
	return "SpeechFeatureExtractor"
}

func (s *SpeechFeatureExtractor) GetContentType() config.ContentType {
	if s.isNews {
		return config.ContentNews
	} else {
		return config.ContentTalk
	}
}

func (s *SpeechFeatureExtractor) GetFeatureWeights() map[string]float64 {
	if s.config.SimilarityWeights != nil {
		return s.config.SimilarityWeights
	}

	// Default weights for speech content
	return map[string]float64{
		"mfcc":     0.50,
		"speech":   0.25,
		"spectral": 0.15,
		"temporal": 0.10,
	}
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
		"function": "ExtractFeatures",
		"frames":   spectrogram.TimeFrames,
	})

	logger.Debug("Extracting speech-specific features")

	features := &ExtractedFeatures{
		ExtractionMetadata: make(map[string]any),
	}

	if s.config.EnableSpeechFeatures {
		speechFeatures, err := s.extractSpeechFeatures(spectrogram, pcm, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract speech features")

			// Create empty speech features instead of leaving nil
			speechFeatures = &SpeechFeatures{
				FormantFrequencies: make([][]float64, spectrogram.TimeFrames),
				VoicingProbability: make([]float64, spectrogram.TimeFrames),
				SpectralTilt:       make([]float64, spectrogram.TimeFrames),
				PauseDuration:      make([]float64, 0),
			}
		}
		features.SpeechFeatures = speechFeatures
	}

	// Extract basic spectral features (focused on speech range)
	spectralFeatures, err := s.extractSpeechSpectralFeatures(spectrogram, pcm)
	if err != nil {
		logger.Error(err, "Failed to extract spectral features")

		// Create empty spectral features instead of leaving nil
		spectralFeatures = &SpectralFeatures{
			SpectralCentroid:  make([]float64, spectrogram.TimeFrames),
			SpectralRolloff:   make([]float64, spectrogram.TimeFrames),
			SpectralBandwidth: make([]float64, spectrogram.TimeFrames),
			SpectralFlatness:  make([]float64, spectrogram.TimeFrames),
			SpectralCrest:     make([]float64, spectrogram.TimeFrames),
			SpectralSlope:     make([]float64, spectrogram.TimeFrames),
			ZeroCrossingRate:  make([]float64, spectrogram.TimeFrames),
			SpectralFlux:      make([]float64, max(0, spectrogram.TimeFrames-1)),
		}
	}
	features.SpectralFeatures = spectralFeatures

	// Extract temporal features (pauses, speech rate)
	if s.config.EnableTemporalFeatures {
		temporalFeatures, err := s.extractSpeechTemporalFeatures(pcm, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract temporal features")

			// Create empty temporal features instead of leaving nil
			temporalFeatures = &TemporalFeatures{
				RMSEnergy:        make([]float64, 0),
				DynamicRange:     0.0,
				SilenceRatio:     0.0,
				PeakAmplitude:    0.0,
				AverageAmplitude: 0.0,
				OnsetDensity:     0.0,
			}
		}
		features.TemporalFeatures = temporalFeatures
	}

	// Step 5: Extract energy features
	logger.Info("Step 5: Extracting energy features...")
	energyFeatures, err := s.extractSpeechEnergyFeatures(pcm, sampleRate)
	if err != nil {
		logger.Error(err, "Failed to extract energy features")
		// Create empty energy features instead of leaving nil
		energyFeatures = &EnergyFeatures{
			ShortTimeEnergy: make([]float64, 0),
			EnergyEntropy:   make([]float64, 0),
			CrestFactor:     make([]float64, 0),
			EnergyVariance:  0.0,
			LoudnessRange:   0.0,
		}
	}
	features.EnergyFeatures = energyFeatures
	logger.Info("Energy features extraction completed")

	// Add extraction metadata
	features.ExtractionMetadata["extractor_type"] = "speech"
	features.ExtractionMetadata["mfcc_coefficients"] = s.config.MFCCCoefficients
	features.ExtractionMetadata["speech_features_enabled"] = s.config.EnableSpeechFeatures

	logger.Info("Speech feature extraction completed")
	return features, nil
}

// extractMFCC computes MFCC optimized for speech (300-4000Hz range)
// TODO: this isn't called anywhere
func (s *SpeechFeatureExtractor) extractMFCC(spectrogram *analyzers.SpectrogramResult) ([][]float64, error) {
	logger := s.logger.WithFields(logging.Fields{
		"function":          "extractMFCC",
		"mfcc_coefficients": s.config.MFCCCoefficients,
	})

	numCoeffs := s.config.MFCCCoefficients
	if numCoeffs == 0 {
		numCoeffs = 13 // Standard number for speech
	}

	mfcc := make([][]float64, spectrogram.TimeFrames)

	// Create mel filter bank optimized for speech (300-4000Hz)
	numMelFilters := 26
	speechLowFreq := 300.0   // Lower bound for speech
	speechHighFreq := 4000.0 // Upper bound for speech
	melFilters := s.createSpeechMelFilterBank(numMelFilters, speechLowFreq, speechHighFreq, spectrogram.FreqBins, spectrogram.SampleRate)

	// Process each time frame
	for t := 0; t < spectrogram.TimeFrames; t++ {
		magnitude := spectrogram.Magnitude[t]

		// Apply mel filter bank
		melSpectrum := s.applyMelFilters(magnitude, melFilters)

		// Pre-emphasis for speech (optional but common)
		melSpectrum = s.applyPreEmphasis(melSpectrum)

		// Take logarithm
		logMelSpectrum := make([]float64, len(melSpectrum))
		for i, val := range melSpectrum {
			if val > 1e-10 {
				logMelSpectrum[i] = math.Log(val)
			} else {
				logMelSpectrum[i] = math.Log(1e-10) // Floor value
			}
		}

		// Apply DCT (Discrete Cosine Transform)
		mfcc[t] = s.applyDCT(logMelSpectrum, numCoeffs)

		// Apply liftering (optional cepstral smoothing for speech)
		mfcc[t] = s.applyLiftering(mfcc[t])
	}

	logger.Debug("Speech MFCC features extracted", logging.Fields{
		"mfcc_frames": len(mfcc),
		"mfcc_coeffs": numCoeffs,
		"freq_range":  fmt.Sprintf("%.0f-%.0fHz", speechLowFreq, speechHighFreq),
	})

	return mfcc, nil
}

// extractSpeechFeatures extracts speech-specific characteristics
func (s *SpeechFeatureExtractor) extractSpeechFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*SpeechFeatures, error) {
	logger := s.logger.WithFields(logging.Fields{
		"function": "extractSpeechFeatures",
	})

	features := &SpeechFeatures{
		FormantFrequencies: make([][]float64, spectrogram.TimeFrames),
		VoicingProbability: make([]float64, spectrogram.TimeFrames),
		SpectralTilt:       make([]float64, spectrogram.TimeFrames),
		PauseDuration:      make([]float64, 0),
	}

	// Generate frequency bins
	freqs := make([]float64, spectrogram.FreqBins)
	for i := range spectrogram.FreqBins {
		freqs[i] = float64(i) * float64(sampleRate) / float64((spectrogram.FreqBins-1)*2)
	}

	// Process each frame
	for t := 0; t < spectrogram.TimeFrames; t++ {
		magnitude := spectrogram.Magnitude[t]

		// Extract formant frequencies (first 3 formants)
		features.FormantFrequencies[t] = s.extractFormants(magnitude, freqs)

		// Calculate voicing probability
		features.VoicingProbability[t] = s.calculateVoicingProbability(magnitude, freqs)

		// Calculate spectral tilt (speech characteristic)
		features.SpectralTilt[t] = s.calculateSpectralTilt(magnitude, freqs)
	}

	// Extract global speech characteristics
	features.SpeechRate = s.calculateSpeechRate(pcm, sampleRate)
	features.PauseDuration = s.extractPauseDurations(pcm, sampleRate)
	features.VocalTractLength = s.estimateVocalTractLength(features.FormantFrequencies)

	logger.Debug("Speech-specific features extracted", logging.Fields{
		"avg_voicing":        s.calculateMean(features.VoicingProbability),
		"speech_rate":        features.SpeechRate,
		"vocal_tract_length": features.VocalTractLength,
		"pause_count":        len(features.PauseDuration),
	})

	return features, nil
}

// extractSpeechSpectralFeatures extracts spectral features focused on speech range
func (s *SpeechFeatureExtractor) extractSpeechSpectralFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64) (*SpectralFeatures, error) {
	logger := s.logger.WithFields(logging.Fields{
		"function": "extractSpeechSpectralFeatures",
	})

	features := &SpectralFeatures{
		SpectralCentroid:  make([]float64, spectrogram.TimeFrames),
		SpectralRolloff:   make([]float64, spectrogram.TimeFrames),
		SpectralBandwidth: make([]float64, spectrogram.TimeFrames),
		SpectralFlatness:  make([]float64, spectrogram.TimeFrames),
		SpectralCrest:     make([]float64, spectrogram.TimeFrames),
		SpectralSlope:     make([]float64, spectrogram.TimeFrames),
		ZeroCrossingRate:  make([]float64, spectrogram.TimeFrames),
	}

	// Generate frequency bins focused on speech range (300-4000Hz)
	freqs := make([]float64, spectrogram.FreqBins)
	speechFreqs := make([]float64, 0)
	speechMags := make([][]float64, spectrogram.TimeFrames)

	for i := 0; i < spectrogram.FreqBins; i++ {
		freq := float64(i) * float64(spectrogram.SampleRate) / float64((spectrogram.FreqBins-1)*2)
		freqs[i] = freq

		// Keep only speech-relevant frequencies
		if freq >= 300 && freq <= 4000 {
			speechFreqs = append(speechFreqs, freq)
		}
	}

	hopSize := spectrogram.HopSize
	frameSize := hopSize * 2

	// Extract speech-band magnitude for each frame
	for t := 0; t < spectrogram.TimeFrames; t++ {
		speechMag := make([]float64, 0)
		for i, freq := range freqs {
			if freq >= 300 && freq <= 4000 && i < len(spectrogram.Magnitude[t]) {
				speechMag = append(speechMag, spectrogram.Magnitude[t][i])
			}
		}
		speechMags[t] = speechMag

		// Calculate features on speech-band only
		features.SpectralCentroid[t] = s.calculateSpectralCentroid(speechMag, speechFreqs)
		features.SpectralRolloff[t] = s.calculateSpectralRolloff(speechMag, speechFreqs, 0.90) // Higher rolloff for speech
		features.SpectralBandwidth[t] = s.calculateSpectralBandwidth(speechMag, speechFreqs, features.SpectralCentroid[t])
		features.SpectralFlatness[t] = s.calculateSpectralFlatness(speechMag)
		features.SpectralCrest[t] = s.calculateSpectralCrest(speechMag)
		features.SpectralSlope[t] = s.calculateSpectralSlope(speechMag, speechFreqs)

		// Calculate zero crossing rate
		start := t * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		if start < len(pcm) {
			pcmFrame := pcm[start:end]
			features.ZeroCrossingRate[t] = calculateZeroCrossingRate(pcmFrame)
		}
	}

	// Calculate spectral flux for speech activity detection
	features.SpectralFlux = s.calculateSpeechSpectralFlux(speechMags)

	logger.Debug("Speech spectral features extracted", logging.Fields{
		"avg_centroid": s.calculateMean(features.SpectralCentroid),
		"avg_rolloff":  s.calculateMean(features.SpectralRolloff),
		"speech_bands": len(speechFreqs),
	})

	return features, nil
}

// Helper functions for logging
func (s *SpeechFeatureExtractor) findMin(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	min := values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
	}
	return min
}

func (s *SpeechFeatureExtractor) findMax(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}

// extractSpeechTemporalFeatures extracts temporal features relevant to speech
func (s *SpeechFeatureExtractor) extractSpeechTemporalFeatures(pcm []float64, sampleRate int) (*TemporalFeatures, error) {
	logger := s.logger.WithFields(logging.Fields{
		"function":    "extractSpeechTemporalFeatures",
		"sample_rate": sampleRate,
	})

	features := &TemporalFeatures{}

	// Frame size optimized for speech (typically 20-30ms)
	frameSize := sampleRate * 25 / 1000 // 25ms frames
	hopSize := frameSize / 2            // 50% overlap

	numFrames := (len(pcm)-frameSize)/hopSize + 1
	if numFrames <= 0 {
		return features, fmt.Errorf("insufficient audio length for speech temporal analysis")
	}

	features.RMSEnergy = make([]float64, numFrames)
	energies := make([]float64, numFrames)

	// Calculate frame-based features
	for i := range numFrames {
		start := i * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		frame := pcm[start:end]

		// RMS Energy per frame
		rms := 0.0
		for _, sample := range frame {
			rms += sample * sample
		}
		rms = math.Sqrt(rms / float64(len(frame)))
		features.RMSEnergy[i] = rms
		energies[i] = rms
	}

	// Speech-specific temporal characteristics
	features.DynamicRange = s.calculateSpeechDynamicRange(pcm)
	features.SilenceRatio = s.calculateSpeechSilenceRatio(energies, sampleRate)
	features.PeakAmplitude = s.calculatePeakAmplitude(pcm)
	features.AverageAmplitude = s.calculateAverageAmplitude(pcm)

	// Voice activity and speech segmentation
	voiceActivityRatio := s.calculateVoiceActivityRatio(energies)
	features.OnsetDensity = s.calculateSpeechOnsetDensity(energies, float64(sampleRate)/float64(hopSize))

	logger.Debug("Speech temporal features extracted", logging.Fields{
		"dynamic_range":        features.DynamicRange,
		"silence_ratio":        features.SilenceRatio,
		"voice_activity":       voiceActivityRatio,
		"speech_onset_density": features.OnsetDensity,
	})

	return features, nil
}

// extractSpeechEnergyFeatures extracts energy features for voice activity detection
func (s *SpeechFeatureExtractor) extractSpeechEnergyFeatures(pcm []float64, sampleRate int) (*EnergyFeatures, error) {
	logger := s.logger.WithFields(logging.Fields{
		"function": "extractSpeechEnergyFeatures",
	})

	features := &EnergyFeatures{}

	// Speech-optimized frame size (25ms)
	frameSize := sampleRate * 25 / 1000
	hopSize := frameSize / 2

	numFrames := (len(pcm)-frameSize)/hopSize + 1
	if numFrames <= 0 {
		return features, fmt.Errorf("insufficient audio length for speech energy analysis")
	}

	features.ShortTimeEnergy = make([]float64, numFrames)
	features.EnergyEntropy = make([]float64, numFrames)
	features.CrestFactor = make([]float64, numFrames)

	energies := make([]float64, numFrames)

	// Calculate frame-based energy features
	for i := range numFrames {
		start := i * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		frame := pcm[start:end]

		// Short-time energy
		energy := 0.0
		maxVal := 0.0
		for _, sample := range frame {
			energy += sample * sample
			if math.Abs(sample) > maxVal {
				maxVal = math.Abs(sample)
			}
		}
		energy /= float64(len(frame))
		features.ShortTimeEnergy[i] = energy
		energies[i] = energy

		// Crest factor (peak-to-RMS ratio)
		rms := math.Sqrt(energy)
		if rms > 0 {
			features.CrestFactor[i] = maxVal / rms
		}

		// Energy entropy (speech vs noise discrimination)
		features.EnergyEntropy[i] = s.calculateSpeechEnergyEntropy(frame)
	}

	// Global energy characteristics for speech
	features.EnergyVariance = s.calculateVariance(energies)
	features.LoudnessRange = s.calculateSpeechLoudnessRange(energies)

	logger.Debug("Speech energy features extracted", logging.Fields{
		"energy_variance": features.EnergyVariance,
		"loudness_range":  features.LoudnessRange,
		"avg_crest":       s.calculateMean(features.CrestFactor),
	})

	return features, nil
}

// Speech-specific helper methods

func (s *SpeechFeatureExtractor) createSpeechMelFilterBank(numFilters int, lowFreq, highFreq float64, freqBins, sampleRate int) [][]float64 {
	// Convert to mel scale
	lowMel := 2595.0 * math.Log10(1.0+lowFreq/700.0)
	highMel := 2595.0 * math.Log10(1.0+highFreq/700.0)

	// Create equally spaced mel frequencies
	melPoints := make([]float64, numFilters+2)
	melStep := (highMel - lowMel) / float64(numFilters+1)
	for i := range melPoints {
		melPoints[i] = lowMel + float64(i)*melStep
	}

	// Convert back to Hz
	freqPoints := make([]float64, len(melPoints))
	for i, mel := range melPoints {
		freqPoints[i] = 700.0 * (math.Pow(10, mel/2595.0) - 1.0)
	}

	// Create filter bank
	filterBank := make([][]float64, numFilters)
	for i := range numFilters {
		filter := make([]float64, freqBins)

		leftFreq := freqPoints[i]
		centerFreq := freqPoints[i+1]
		rightFreq := freqPoints[i+2]

		for j := range freqBins {
			freq := float64(j) * float64(sampleRate) / float64(freqBins*2)

			if freq >= leftFreq && freq <= rightFreq {
				if freq <= centerFreq {
					// Rising edge
					if centerFreq > leftFreq {
						filter[j] = (freq - leftFreq) / (centerFreq - leftFreq)
					}
				} else {
					// Falling edge
					if rightFreq > centerFreq {
						filter[j] = (rightFreq - freq) / (rightFreq - centerFreq)
					}
				}
			}
		}

		filterBank[i] = filter
	}

	return filterBank
}

func (s *SpeechFeatureExtractor) applyPreEmphasis(melSpectrum []float64) []float64 {
	// Apply high-frequency emphasis common in speech processing
	alpha := 0.97
	emphasized := make([]float64, len(melSpectrum))

	if len(melSpectrum) > 0 {
		emphasized[0] = melSpectrum[0]
		for i := 1; i < len(melSpectrum); i++ {
			emphasized[i] = melSpectrum[i] - alpha*melSpectrum[i-1]
		}
	}

	return emphasized
}

func (s *SpeechFeatureExtractor) applyLiftering(mfcc []float64) []float64 {
	// Apply liftering to smooth cepstral coefficients
	L := 22 // Liftering parameter
	liftered := make([]float64, len(mfcc))

	for i := range mfcc {
		if i == 0 {
			liftered[i] = mfcc[i] // Don't lifter C0
		} else {
			lifter := 1 + float64(L)/2*math.Sin(math.Pi*float64(i)/float64(L))
			liftered[i] = mfcc[i] * lifter
		}
	}

	return liftered
}

func (s *SpeechFeatureExtractor) extractFormants(magnitude, freqs []float64) []float64 {
	// Simple formant extraction using peak picking
	formants := make([]float64, 3) // F1, F2, F3

	// Find peaks in typical formant ranges
	f1Range := [2]float64{200, 1000}  // F1 range
	f2Range := [2]float64{800, 2500}  // F2 range
	f3Range := [2]float64{1500, 4000} // F3 range

	ranges := [][2]float64{f1Range, f2Range, f3Range}

	for formantIdx, fRange := range ranges {
		maxMag := 0.0
		maxFreq := 0.0

		for i, freq := range freqs {
			if freq >= fRange[0] && freq <= fRange[1] && i < len(magnitude) {
				if magnitude[i] > maxMag {
					maxMag = magnitude[i]
					maxFreq = freq
				}
			}
		}

		formants[formantIdx] = maxFreq
	}

	return formants
}

func (s *SpeechFeatureExtractor) calculateVoicingProbability(magnitude, freqs []float64) float64 {
	// Estimate voicing based on harmonic structure
	harmonicEnergy := 0.0
	totalEnergy := 0.0

	// Look for harmonic peaks in voiced speech range (80-400Hz fundamental)
	for i, freq := range freqs {
		if freq >= 80 && freq <= 400 && i < len(magnitude) {
			energy := magnitude[i] * magnitude[i]
			totalEnergy += energy

			// Check for harmonics
			for harmonic := 2; harmonic <= 10; harmonic++ {
				harmonicFreq := freq * float64(harmonic)
				if harmonicFreq <= 4000 {
					// Find closest frequency bin
					closestIdx := s.findClosestFreqBin(harmonicFreq, freqs)
					if closestIdx >= 0 && closestIdx < len(magnitude) {
						harmonicEnergy += magnitude[closestIdx] * magnitude[closestIdx]
					}
				}
			}
		}
	}

	if totalEnergy == 0 {
		return 0
	}

	return math.Min(1.0, harmonicEnergy/totalEnergy)
}

func (s *SpeechFeatureExtractor) calculateSpectralTilt(magnitude, freqs []float64) float64 {
	// Calculate spectral tilt (slope) in speech range
	if len(magnitude) != len(freqs) || len(magnitude) < 2 {
		return 0
	}

	// Focus on speech range (300-4000Hz)
	var speechMag, speechFreqs []float64
	for i, freq := range freqs {
		if freq >= 300 && freq <= 4000 && i < len(magnitude) {
			speechFreqs = append(speechFreqs, freq)
			speechMag = append(speechMag, magnitude[i])
		}
	}

	if len(speechMag) < 2 {
		return 0
	}

	// Linear regression in log domain
	n := 0
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumXX := 0.0

	for i := range speechMag {
		if speechMag[i] > 1e-10 && speechFreqs[i] > 0 {
			x := math.Log10(speechFreqs[i])
			y := math.Log10(speechMag[i])

			sumX += x
			sumY += y
			sumXY += x * y
			sumXX += x * x
			n++
		}
	}

	if n < 2 {
		return 0
	}

	denominator := float64(n)*sumXX - sumX*sumX
	if denominator == 0 {
		return 0
	}

	return (float64(n)*sumXY - sumX*sumY) / denominator
}

func (s *SpeechFeatureExtractor) calculateSpeechRate(pcm []float64, sampleRate int) float64 {
	// Estimate speech rate based on energy fluctuations
	frameSize := sampleRate * 25 / 1000 // 25ms frames
	hopSize := frameSize / 2

	if len(pcm) < frameSize {
		return 0
	}

	// Calculate frame energies
	energies := make([]float64, 0)
	for i := 0; i < len(pcm)-frameSize; i += hopSize {
		energy := 0.0
		for j := 0; j < frameSize && i+j < len(pcm); j++ {
			energy += pcm[i+j] * pcm[i+j]
		}
		energies = append(energies, math.Sqrt(energy/float64(frameSize)))
	}

	if len(energies) < 10 {
		return 0
	}

	// Count syllable-like peaks (simplified approach)
	threshold := s.calculateMean(energies) * 1.2
	peaks := 0

	for i := 1; i < len(energies)-1; i++ {
		if energies[i] > threshold && energies[i] > energies[i-1] && energies[i] > energies[i+1] {
			peaks++
		}
	}

	// Convert to syllables per second (rough speech rate estimate)
	duration := float64(len(energies)) * float64(hopSize) / float64(sampleRate)
	if duration > 0 {
		return float64(peaks) / duration
	}

	return 0
}

func (s *SpeechFeatureExtractor) extractPauseDurations(pcm []float64, sampleRate int) []float64 {
	// Extract pause durations for speech analysis
	frameSize := sampleRate / 20 // 50ms frames
	silenceThreshold := 0.01     // RMS threshold for silence

	pauses := make([]float64, 0)
	currentPauseDuration := 0.0
	frameTimeMs := 50.0 // 50ms per frame

	for i := 0; i < len(pcm)-frameSize; i += frameSize / 2 {
		rms := 0.0
		for j := 0; j < frameSize && i+j < len(pcm); j++ {
			rms += pcm[i+j] * pcm[i+j]
		}
		rms = math.Sqrt(rms / float64(frameSize))

		if rms < silenceThreshold {
			currentPauseDuration += frameTimeMs / 2 // Half frame due to overlap
		} else {
			if currentPauseDuration > 100 { // Only count pauses > 100ms
				pauses = append(pauses, currentPauseDuration)
			}
			currentPauseDuration = 0
		}
	}

	// Add final pause if it exists
	if currentPauseDuration > 100 {
		pauses = append(pauses, currentPauseDuration)
	}

	return pauses
}

func (s *SpeechFeatureExtractor) estimateVocalTractLength(formants [][]float64) float64 {
	// Estimate vocal tract length from formant frequencies
	if len(formants) == 0 {
		return 0
	}

	// Average the first formant across all frames
	f1Sum := 0.0
	f1Count := 0

	for _, frameFormants := range formants {
		if len(frameFormants) > 0 && frameFormants[0] > 0 {
			f1Sum += frameFormants[0]
			f1Count++
		}
	}

	if f1Count == 0 {
		return 0
	}

	avgF1 := f1Sum / float64(f1Count)

	// Rough estimation: VTL ≈ c / (4 * F1) where c = speed of sound (350 m/s)
	// Result in cm
	if avgF1 > 0 {
		return 35000 / (4 * avgF1) // Convert to cm
	}

	return 0
}

// Additional speech-specific helper methods

func (s *SpeechFeatureExtractor) calculateSpeechDynamicRange(pcm []float64) float64 {
	// Calculate dynamic range specific to speech characteristics
	if len(pcm) == 0 {
		return 0
	}

	// Use percentiles instead of absolute min/max to avoid outliers
	sorted := make([]float64, len(pcm))
	for i, sample := range pcm {
		sorted[i] = math.Abs(sample)
	}

	sort.Float64s(sorted)

	// Use 5th and 95th percentiles for robust dynamic range
	p5 := sorted[len(sorted)/20]     // 5th percentile
	p95 := sorted[19*len(sorted)/20] // 95th percentile

	if p5 > 1e-10 {
		return 20 * math.Log10(p95/p5)
	}

	return 0
}

func (s *SpeechFeatureExtractor) calculateSpeechSilenceRatio(energies []float64, sampleRate int) float64 {
	// Calculate silence ratio with speech-specific thresholds
	if len(energies) == 0 {
		return 0
	}

	// Dynamic threshold based on energy statistics
	meanEnergy := s.calculateMean(energies)
	threshold := meanEnergy * 0.1 // 10% of mean energy

	silentFrames := 0
	for _, energy := range energies {
		if energy < threshold {
			silentFrames++
		}
	}

	return float64(silentFrames) / float64(len(energies))
}

func (s *SpeechFeatureExtractor) calculateVoiceActivityRatio(energies []float64) float64 {
	// Calculate voice activity detection ratio
	if len(energies) == 0 {
		return 0
	}

	// Use energy-based VAD with adaptive threshold
	meanEnergy := s.calculateMean(energies)
	stdEnergy := math.Sqrt(s.calculateVariance(energies))
	threshold := meanEnergy + 0.5*stdEnergy

	activeFrames := 0
	for _, energy := range energies {
		if energy > threshold {
			activeFrames++
		}
	}

	return float64(activeFrames) / float64(len(energies))
}

func (s *SpeechFeatureExtractor) calculateSpeechOnsetDensity(energies []float64, frameRate float64) float64 {
	// Calculate onset density specific to speech patterns
	if len(energies) < 3 {
		return 0
	}

	onsets := 0
	threshold := 1.3 // Lower threshold for speech vs music

	// Adaptive thresholding for speech onset detection
	windowSize := 3
	for i := windowSize; i < len(energies)-1; i++ {
		current := energies[i]

		// Local average (smaller window for speech)
		localAvg := 0.0
		for j := i - windowSize; j < i; j++ {
			if j >= 0 {
				localAvg += energies[j]
			}
		}
		localAvg /= float64(windowSize)

		// Check for speech onset (energy increase)
		if current > localAvg*threshold {
			onsets++
		}
	}

	// Return onsets per second
	duration := float64(len(energies)) / frameRate
	if duration == 0 {
		return 0
	}

	return float64(onsets) / duration
}

func (s *SpeechFeatureExtractor) calculateSpeechEnergyEntropy(frame []float64) float64 {
	// Calculate energy entropy for speech/noise discrimination
	if len(frame) == 0 {
		return 0
	}

	// Divide frame into sub-frames (smaller for speech)
	numSubFrames := 8 // Smaller than music for speech resolution
	subFrameSize := len(frame) / numSubFrames

	if subFrameSize == 0 {
		return 0
	}

	subFrameEnergies := make([]float64, numSubFrames)
	totalEnergy := 0.0

	// Calculate energy for each sub-frame
	for i := range numSubFrames {
		start := i * subFrameSize
		end := start + subFrameSize
		end = min(end, len(frame))

		energy := 0.0
		for j := start; j < end; j++ {
			energy += frame[j] * frame[j]
		}
		subFrameEnergies[i] = energy
		totalEnergy += energy
	}

	if totalEnergy == 0 {
		return 0
	}

	// Calculate entropy
	entropy := 0.0
	for _, energy := range subFrameEnergies {
		if energy > 0 {
			prob := energy / totalEnergy
			entropy -= prob * math.Log2(prob)
		}
	}

	return entropy
}

func (s *SpeechFeatureExtractor) calculateSpeechLoudnessRange(energies []float64) float64 {
	// Calculate loudness range for speech content
	if len(energies) == 0 {
		return 0
	}

	// Sort energies to find percentiles
	sorted := make([]float64, len(energies))
	copy(sorted, energies)
	sort.Float64s(sorted)

	// Use 10th and 90th percentiles (narrower range for speech)
	p10 := sorted[len(sorted)/10]   // 10th percentile
	p90 := sorted[9*len(sorted)/10] // 90th percentile

	if p10 > 0 {
		return 20 * math.Log10(p90/p10)
	}

	return 0
}

func (s *SpeechFeatureExtractor) calculateSpeechSpectralFlux(speechMags [][]float64) []float64 {
	// Calculate spectral flux for speech activity detection
	if len(speechMags) < 2 {
		return nil
	}

	flux := make([]float64, len(speechMags)-1)

	for t := 1; t < len(speechMags); t++ {
		sum := 0.0
		minLen := len(speechMags[t])
		minLen = min(len(speechMags[t-1]), minLen)

		for f := 0; f < minLen; f++ {
			diff := speechMags[t][f] - speechMags[t-1][f]
			if diff > 0 { // Only positive changes
				sum += diff * diff
			}
		}
		flux[t-1] = math.Sqrt(sum)
	}

	return flux
}

func (s *SpeechFeatureExtractor) findClosestFreqBin(targetFreq float64, freqs []float64) int {
	// Find the frequency bin closest to target frequency
	minDiff := math.Inf(1)
	closestIdx := -1

	for i, freq := range freqs {
		diff := math.Abs(freq - targetFreq)
		if diff < minDiff {
			minDiff = diff
			closestIdx = i
		}
	}

	return closestIdx
}

// Common helper methods shared with other extractors

func (s *SpeechFeatureExtractor) applyMelFilters(magnitude []float64, filterBank [][]float64) []float64 {
	melSpectrum := make([]float64, len(filterBank))

	for i, filter := range filterBank {
		sum := 0.0
		for j, coeff := range filter {
			if j < len(magnitude) {
				sum += magnitude[j] * coeff
			}
		}
		melSpectrum[i] = sum
	}

	return melSpectrum
}

func (s *SpeechFeatureExtractor) applyDCT(logMelSpectrum []float64, numCoeffs int) []float64 {
	mfcc := make([]float64, numCoeffs)
	N := float64(len(logMelSpectrum))

	for k := range numCoeffs {
		sum := 0.0
		for n := range len(logMelSpectrum) {
			sum += logMelSpectrum[n] * math.Cos(math.Pi*float64(k)*(float64(n)+0.5)/N)
		}
		mfcc[k] = sum
	}

	return mfcc
}

func (s *SpeechFeatureExtractor) calculateSpectralCentroid(magnitude, freqs []float64) float64 {
	numerator := 0.0
	denominator := 0.0

	for i := 0; i < len(magnitude) && i < len(freqs); i++ {
		numerator += freqs[i] * magnitude[i]
		denominator += magnitude[i]
	}

	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

func (s *SpeechFeatureExtractor) calculateSpectralRolloff(magnitude, freqs []float64, threshold float64) float64 {
	totalEnergy := 0.0
	for _, mag := range magnitude {
		totalEnergy += mag * mag
	}

	if totalEnergy == 0 {
		return 0
	}

	targetEnergy := threshold * totalEnergy
	cumulativeEnergy := 0.0

	for i := 0; i < len(magnitude) && i < len(freqs); i++ {
		cumulativeEnergy += magnitude[i] * magnitude[i]
		if cumulativeEnergy >= targetEnergy {
			return freqs[i]
		}
	}

	if len(freqs) > 0 {
		return freqs[len(freqs)-1]
	}
	return 0
}

func (s *SpeechFeatureExtractor) calculateSpectralBandwidth(magnitude, freqs []float64, centroid float64) float64 {
	numerator := 0.0
	denominator := 0.0

	for i := 0; i < len(magnitude) && i < len(freqs); i++ {
		diff := freqs[i] - centroid
		numerator += diff * diff * magnitude[i]
		denominator += magnitude[i]
	}

	if denominator == 0 {
		return 0
	}
	return math.Sqrt(numerator / denominator)
}

func (s *SpeechFeatureExtractor) calculateSpectralFlatness(magnitude []float64) float64 {
	if len(magnitude) == 0 {
		return 0
	}

	// Geometric mean
	logSum := 0.0
	count := 0
	for _, mag := range magnitude {
		if mag > 1e-10 {
			logSum += math.Log(mag)
			count++
		}
	}

	if count == 0 {
		return 0
	}

	geometricMean := math.Exp(logSum / float64(count))

	// Arithmetic mean
	arithmeticMean := 0.0
	for _, mag := range magnitude {
		arithmeticMean += mag
	}
	arithmeticMean /= float64(len(magnitude))

	if arithmeticMean == 0 {
		return 0
	}

	return geometricMean / arithmeticMean
}

func (s *SpeechFeatureExtractor) calculateSpectralCrest(magnitude []float64) float64 {
	if len(magnitude) == 0 {
		return 0
	}

	maxVal := 0.0
	sumSquares := 0.0

	for _, mag := range magnitude {
		if mag > maxVal {
			maxVal = mag
		}
		sumSquares += mag * mag
	}

	rms := math.Sqrt(sumSquares / float64(len(magnitude)))
	if rms == 0 {
		return 0
	}

	return maxVal / rms
}

func (s *SpeechFeatureExtractor) calculateSpectralSlope(magnitude, freqs []float64) float64 {
	if len(magnitude) != len(freqs) || len(magnitude) < 2 {
		return 0
	}

	// Linear regression in log domain
	n := 0
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumXX := 0.0

	for i := range len(magnitude) {
		if magnitude[i] > 1e-10 && freqs[i] > 0 {
			x := math.Log10(freqs[i])
			y := math.Log10(magnitude[i])

			sumX += x
			sumY += y
			sumXY += x * y
			sumXX += x * x
			n++
		}
	}

	if n < 2 {
		return 0
	}

	denominator := float64(n)*sumXX - sumX*sumX
	if denominator == 0 {
		return 0
	}

	return (float64(n)*sumXY - sumX*sumY) / denominator
}

func (s *SpeechFeatureExtractor) calculatePeakAmplitude(pcm []float64) float64 {
	maxVal := 0.0
	for _, sample := range pcm {
		if math.Abs(sample) > maxVal {
			maxVal = math.Abs(sample)
		}
	}
	return maxVal
}

func (s *SpeechFeatureExtractor) calculateAverageAmplitude(pcm []float64) float64 {
	if len(pcm) == 0 {
		return 0
	}

	sum := 0.0
	for _, sample := range pcm {
		sum += math.Abs(sample)
	}
	return sum / float64(len(pcm))
}

func (s *SpeechFeatureExtractor) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, val := range values {
		sum += val
	}
	return sum / float64(len(values))
}

func (s *SpeechFeatureExtractor) calculateVariance(values []float64) float64 {
	if len(values) <= 1 {
		return 0
	}

	mean := s.calculateMean(values)
	variance := 0.0

	for _, val := range values {
		diff := val - mean
		variance += diff * diff
	}

	return variance / float64(len(values))
}
