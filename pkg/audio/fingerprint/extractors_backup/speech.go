package extractors

import (
	"fmt"
	"math"
	"sort"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
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

	logger.Info("Extractor received spectrogram", logging.Fields{
		"spectrogram_frames":      spectrogram.TimeFrames,
		"spectrogram_freq_bins":   spectrogram.FreqBins,
		"spectrogram_hop_size":    spectrogram.HopSize,
		"spectrogram_window_size": spectrogram.WindowSize,
		"pcm_length":              len(pcm),
		"sample_rate":             sampleRate,
	})

	features := &ExtractedFeatures{
		ExtractionMetadata: make(map[string]any),
	}

	// Extract MFCC features with validation
	if s.config.EnableMFCC {
		logger.Info("Extracting MFCC features...")
		mfccFeatures, err := s.extractMFCC(spectrogram)
		if err != nil {
			logger.Error(err, "Failed to extract MFCC features")
			// Create empty MFCC instead of failing completely
			features.MFCC = make([][]float64, spectrogram.TimeFrames)
			for i := range features.MFCC {
				features.MFCC[i] = make([]float64, s.config.MFCCCoefficients)
			}
		} else {
			// VALIDATE MFCC values before accepting them
			err = s.validateMFCCValues(mfccFeatures)
			if err != nil {
				logger.Error(err, "MFCC validation failed")
				return nil, fmt.Errorf("MFCC validation failed: %w", err)
			}

			features.MFCC = mfccFeatures
			logger.Info("MFCC features extracted and validated successfully", logging.Fields{
				"frames": len(mfccFeatures),
				"coeffs": len(mfccFeatures[0]),
			})
		}
	}

	// Extract speech-specific features if enabled
	if s.config.EnableSpeechFeatures {
		logger.Info("Extracting speech-specific features...")
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

	// Extract spectral features (focused on speech range)
	logger.Info("Extracting spectral features...")
	spectralFeatures, err := s.extractSpeechSpectralFeatures(spectrogram, pcm)
	if err != nil {
		logger.Error(err, "Failed to extract spectral features")
		return nil, fmt.Errorf("spectral feature extraction failed: %w", err)
	}
	features.SpectralFeatures = spectralFeatures

	// Extract temporal features (pauses, speech rate)
	if s.config.EnableTemporalFeatures {
		logger.Info("Extracting temporal features...")
		temporalFeatures, err := s.extractSpeechTemporalFeatures(pcm, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract temporal features")
			return nil, fmt.Errorf("temporal feature extraction failed: %w", err)
		}
		features.TemporalFeatures = temporalFeatures
	}

	// Extract energy features (critical for alignment)
	logger.Info("Extracting energy features...")
	energyFeatures, err := s.extractSpeechEnergyFeatures(pcm, spectrogram)
	if err != nil {
		logger.Error(err, "Failed to extract energy features")
		return nil, fmt.Errorf("energy feature extraction failed: %w", err)
	}
	features.EnergyFeatures = energyFeatures

	// Add comprehensive extraction metadata
	features.ExtractionMetadata["extractor_type"] = "speech"
	features.ExtractionMetadata["mfcc_enabled"] = s.config.EnableMFCC
	features.ExtractionMetadata["mfcc_coefficients"] = s.config.MFCCCoefficients
	features.ExtractionMetadata["speech_features_enabled"] = s.config.EnableSpeechFeatures
	features.ExtractionMetadata["temporal_features_enabled"] = s.config.EnableTemporalFeatures

	// Frame count validation
	features.ExtractionMetadata["spectrogram_frames"] = spectrogram.TimeFrames
	features.ExtractionMetadata["energy_frames"] = len(energyFeatures.ShortTimeEnergy)
	features.ExtractionMetadata["spectral_frames"] = len(spectralFeatures.SpectralCentroid)
	if features.MFCC != nil {
		features.ExtractionMetadata["mfcc_frames"] = len(features.MFCC)
	}
	if features.TemporalFeatures != nil {
		features.ExtractionMetadata["temporal_frames"] = len(features.TemporalFeatures.RMSEnergy)
	}

	// Timing metadata
	features.ExtractionMetadata["spectrogram_hop_size"] = spectrogram.HopSize
	features.ExtractionMetadata["spectrogram_window_size"] = spectrogram.WindowSize
	features.ExtractionMetadata["sample_rate"] = sampleRate

	logger.Info("Speech feature extraction completed successfully")

	// VALIDATE that everything is properly aligned
	err = s.validateFeatureConsistency(features, spectrogram)
	if err != nil {
		logger.Error(err, "Feature consistency validation failed")
		return nil, fmt.Errorf("feature validation failed: %w", err)
	}

	return features, nil
}

// Things we need:
// - DCT TYPE
// - Silence Threshold
// - Warping formula
// - Weighting
// - LogType ('dbpow', 'dbamp', 'log', 'natural')
// - High frequency bound
func (s *SpeechFeatureExtractor) extractMFCC(spectrogram *analyzers.SpectrogramResult) ([][]float64, error) {
	// logger := s.logger.WithFields(logging.Fields{
	// "function":          "extractMFCC",
	// "mfcc_coefficients": s.config.MFCCCoefficients,
	// })

	numCoeffs := s.config.MFCCCoefficients
	if numCoeffs == 0 {
		numCoeffs = 13
	}

	if numCoeffs < 1 || numCoeffs > 50 {
		return nil, fmt.Errorf("invalid number of MFCC coefficients: %d (should be 1-50)", numCoeffs)
	}

	// Mel filter bank parameters
	numMelFilters := 26
	speechLowFreq := 300.0
	speechHighFreq := minFloat64(4000.0, float64(spectrogram.SampleRate)/2.0)

	melFilters, err := s.createMelFilterBank(numMelFilters, speechLowFreq, speechHighFreq, spectrogram.FreqBins, spectrogram.SampleRate)
	if err != nil {
		return nil, fmt.Errorf("failed to create mel filter bank: %w", err)
	}

	mfcc := make([][]float64, spectrogram.TimeFrames)

	// Process each time frame
	for t := 0; t < spectrogram.TimeFrames; t++ {
		if t >= len(spectrogram.Magnitude) {
			mfcc[t] = make([]float64, numCoeffs)
			continue
		}

		magnitude := spectrogram.Magnitude[t]
		if len(magnitude) == 0 {
			mfcc[t] = make([]float64, numCoeffs)
			continue
		}

		// Apply mel filter bank
		melSpectrum := s.applyMelFilters(magnitude, melFilters)

		// Convert to log mel spectrum
		logMelSpectrum := s.computeLogMelSpectrum(melSpectrum)

		// Apply DCT
		mfccFrame := s.applyDCT(logMelSpectrum, numCoeffs)

		// Apply liftering
		mfcc[t] = s.applyLiftering(mfccFrame)
	}

	// Apply normalization to handle extreme values
	mfcc = s.normalizeMFCC(mfcc)

	return mfcc, nil
}

func (s *SpeechFeatureExtractor) hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

func (s *SpeechFeatureExtractor) melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10, mel/2595.0) - 1.0)
}

func (s *SpeechFeatureExtractor) validateMFCCValues(mfcc [][]float64) error {
	if len(mfcc) == 0 {
		return fmt.Errorf("empty MFCC array")
	}

	for t, frame := range mfcc {
		for c, coeff := range frame {
			if math.IsNaN(coeff) || math.IsInf(coeff, 0) {
				return fmt.Errorf("invalid MFCC at frame %d, coeff %d: %f", t, c, coeff)
			}
			if math.Abs(coeff) > 200 {
				return fmt.Errorf("extreme MFCC at frame %d, coeff %d: %f (should be -50 to +50)", t, c, coeff)
			}
		}
	}

	// Calculate statistics
	var allValues []float64
	for _, frame := range mfcc {
		allValues = append(allValues, frame...)
	}

	if len(allValues) > 0 {
		mean := 0.0
		for _, val := range allValues {
			mean += val
		}
		mean /= float64(len(allValues))

		variance := 0.0
		for _, val := range allValues {
			variance += (val - mean) * (val - mean)
		}
		variance /= float64(len(allValues))
		stdDev := math.Sqrt(variance)

		s.logger.Info("MFCC validation statistics", logging.Fields{
			"frames":  len(mfcc),
			"coeffs":  len(mfcc[0]),
			"mean":    mean,
			"std_dev": stdDev,
			"min":     s.findMinInMFCC(mfcc),
			"max":     s.findMaxInMFCC(mfcc),
		})
	}

	return nil
}

func (s *SpeechFeatureExtractor) findMinInMFCC(mfcc [][]float64) float64 {
	if len(mfcc) == 0 || len(mfcc[0]) == 0 {
		return 0
	}

	min := mfcc[0][0]
	for _, frame := range mfcc {
		for _, coeff := range frame {
			if coeff < min {
				min = coeff
			}
		}
	}
	return min
}

func (s *SpeechFeatureExtractor) findMaxInMFCC(mfcc [][]float64) float64 {
	if len(mfcc) == 0 || len(mfcc[0]) == 0 {
		return 0
	}

	max := mfcc[0][0]
	for _, frame := range mfcc {
		for _, coeff := range frame {
			if coeff > max {
				max = coeff
			}
		}
	}
	return max
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

	// Generate frequency bins
	freqs := make([]float64, spectrogram.FreqBins)
	for i := 0; i < spectrogram.FreqBins; i++ {
		freqs[i] = float64(i) * float64(spectrogram.SampleRate) / float64(spectrogram.WindowSize)
	}

	hopSize := spectrogram.HopSize
	frameSize := spectrogram.WindowSize

	// Process each frame
	for t := range spectrogram.TimeFrames {
		// Get magnitude spectrum for this frame
		magnitude := spectrogram.Magnitude[t]

		// Filter to speech range (300-4000Hz) - FIXED indexing
		speechMag := make([]float64, 0)
		speechFreqs := make([]float64, 0)

		for i, freq := range freqs {
			if freq >= 300 && freq <= 4000 && i < len(magnitude) {
				speechMag = append(speechMag, magnitude[i])
				speechFreqs = append(speechFreqs, freq)
			}
		}

		if len(speechMag) == 0 {
			// No speech frequencies available - use low frequency range
			maxIdx := min(len(magnitude), len(freqs)/4) // Use first quarter of spectrum
			speechMag = magnitude[:maxIdx]
			speechFreqs = freqs[:maxIdx]
		}

		// Calculate features on speech-relevant frequencies
		features.SpectralCentroid[t] = s.calculateSpectralCentroid(speechMag, speechFreqs)
		features.SpectralRolloff[t] = s.calculateSpectralRolloff(speechMag, speechFreqs, 0.85)
		features.SpectralBandwidth[t] = s.calculateSpectralBandwidth(speechMag, speechFreqs, features.SpectralCentroid[t])
		features.SpectralFlatness[t] = s.calculateSpectralFlatness(speechMag)
		features.SpectralCrest[t] = s.calculateSpectralCrest(speechMag)
		features.SpectralSlope[t] = s.calculateSpectralSlope(speechMag, speechFreqs)

		// Calculate zero crossing rate - FIXED frame boundaries
		start := t * hopSize
		end := min(start+frameSize, len(pcm))

		if start < len(pcm) && end > start {
			pcmFrame := pcm[start:end]
			features.ZeroCrossingRate[t] = calculateZeroCrossingRate(pcmFrame)
		}
	}

	// Calculate spectral flux
	features.SpectralFlux = s.calculateSpectralFlux(spectrogram)

	logger.Debug("Speech spectral features extracted", logging.Fields{
		"avg_centroid": s.calculateMean(features.SpectralCentroid),
		"avg_rolloff":  s.calculateMean(features.SpectralRolloff),
		"frames":       spectrogram.TimeFrames,
		"freq_bins":    spectrogram.FreqBins,
		"sample_rate":  spectrogram.SampleRate,
	})

	return features, nil
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
	features.SilenceRatio = s.calculateSpeechSilenceRatio(energies)
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
func (s *SpeechFeatureExtractor) extractSpeechEnergyFeatures(pcm []float64, spectrogram *analyzers.SpectrogramResult) (*EnergyFeatures, error) {
	logger := s.logger.WithFields(logging.Fields{
		"function": "extractSpeechEnergyFeatures",
	})

	// CRITICAL FIX: Use the SAME timing as the spectrogram
	hopSize := spectrogram.HopSize
	frameSize := spectrogram.WindowSize
	expectedFrames := spectrogram.TimeFrames

	logger.Info("Energy extraction using spectrogram timing", logging.Fields{
		"hop_size":        hopSize,
		"frame_size":      frameSize,
		"expected_frames": expectedFrames,
		"pcm_length":      len(pcm),
	})

	features := &EnergyFeatures{
		ShortTimeEnergy: make([]float64, expectedFrames),
		EnergyEntropy:   make([]float64, expectedFrames),
		CrestFactor:     make([]float64, expectedFrames),
	}

	energies := make([]float64, expectedFrames)

	// Extract energy using EXACT same framing as spectrogram
	for i := range expectedFrames {
		start := i * hopSize
		end := min(start+frameSize, len(pcm))

		if start >= len(pcm) {
			// Handle edge case - pad with zeros
			features.ShortTimeEnergy[i] = 0.0
			features.EnergyEntropy[i] = 0.0
			features.CrestFactor[i] = 0.0
			energies[i] = 0.0
			continue
		}

		// Extract frame (with zero padding if needed)
		frame := make([]float64, frameSize)
		copyLen := min(end-start, frameSize)
		copy(frame, pcm[start:start+copyLen])
		// Rest is already zero-padded

		// Short-time energy
		energy := 0.0
		maxVal := 0.0
		for _, sample := range frame {
			energy += sample * sample
			if math.Abs(sample) > maxVal {
				maxVal = math.Abs(sample)
			}
		}
		energy /= float64(len(frame)) // Normalize by frame length
		features.ShortTimeEnergy[i] = energy
		energies[i] = energy

		// Crest factor (peak-to-RMS ratio)
		rms := math.Sqrt(energy)
		if rms > 1e-10 {
			features.CrestFactor[i] = maxVal / rms
		} else {
			features.CrestFactor[i] = 0.0
		}

		// Energy entropy
		features.EnergyEntropy[i] = s.calculateSpeechEnergyEntropy(frame)
	}

	// Global energy characteristics
	features.EnergyVariance = s.calculateVariance(energies)
	features.LoudnessRange = s.calculateSpeechLoudnessRange(energies)

	logger.Info("Speech energy features extracted with correct timing", logging.Fields{
		"frames_extracted": len(features.ShortTimeEnergy),
		"energy_variance":  features.EnergyVariance,
		"loudness_range":   features.LoudnessRange,
		"avg_energy":       s.calculateMean(energies),
	})

	return features, nil
}

// Speech-specific helper methods

func (s *SpeechFeatureExtractor) validateFeatureConsistency(features *ExtractedFeatures, spectrogram *analyzers.SpectrogramResult) error {
	expectedFrames := spectrogram.TimeFrames

	// Check MFCC consistency
	if features.MFCC != nil {
		if len(features.MFCC) != expectedFrames {
			return fmt.Errorf("MFCC frame count mismatch: expected %d, got %d", expectedFrames, len(features.MFCC))
		}
		for i, frame := range features.MFCC {
			if len(frame) != s.config.MFCCCoefficients {
				return fmt.Errorf("MFCC coefficient count mismatch at frame %d: expected %d, got %d",
					i, s.config.MFCCCoefficients, len(frame))
			}
			for j, coeff := range frame {
				if math.IsNaN(coeff) || math.IsInf(coeff, 0) {
					return fmt.Errorf("invalid MFCC coefficient at frame %d, coeff %d: %f", i, j, coeff)
				}
			}
		}
	}

	// Check energy features consistency
	if features.EnergyFeatures != nil {
		if len(features.EnergyFeatures.ShortTimeEnergy) != expectedFrames {
			return fmt.Errorf("energy frame count mismatch: expected %d, got %d",
				expectedFrames, len(features.EnergyFeatures.ShortTimeEnergy))
		}
	}

	// Check spectral features consistency
	if features.SpectralFeatures != nil {
		if len(features.SpectralFeatures.SpectralCentroid) != expectedFrames {
			return fmt.Errorf("spectral frame count mismatch: expected %d, got %d",
				expectedFrames, len(features.SpectralFeatures.SpectralCentroid))
		}
	}

	return nil
}

func (s *SpeechFeatureExtractor) applyLiftering(mfcc []float64) []float64 {
	L := 22.0
	liftered := make([]float64, len(mfcc))

	for i := range mfcc {
		if i == 0 {
			liftered[i] = mfcc[i]
		} else {
			lifter := 1.0 + L/2.0*math.Sin(math.Pi*float64(i)/L)
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

func (s *SpeechFeatureExtractor) calculateSpeechSilenceRatio(energies []float64) float64 {
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

func (s *SpeechFeatureExtractor) computeMagnitudeStatistics(magnitude [][]float64) map[string]float64 {
	var allMagnitudes []float64

	for _, frame := range magnitude {
		for _, mag := range frame {
			if mag > 0 {
				allMagnitudes = append(allMagnitudes, mag)
			}
		}
	}

	if len(allMagnitudes) == 0 {
		return map[string]float64{
			"mean":   1.0,
			"max":    1.0,
			"median": 1.0,
		}
	}

	// Sort for percentiles
	sort.Float64s(allMagnitudes)

	// Calculate statistics
	mean := s.calculateMean(allMagnitudes)
	max := allMagnitudes[len(allMagnitudes)-1]
	median := allMagnitudes[len(allMagnitudes)/2]
	p95 := allMagnitudes[int(0.95*float64(len(allMagnitudes)))]

	return map[string]float64{
		"mean":   mean,
		"max":    max,
		"median": median,
		"p95":    p95,
	}
}

func (s *SpeechFeatureExtractor) createMelFilterBank(numFilters int, lowFreq, highFreq float64, freqBins, sampleRate int) ([][]float64, error) {
	if numFilters <= 0 || freqBins <= 0 || sampleRate <= 0 {
		return nil, fmt.Errorf("invalid parameters: filters=%d, bins=%d, rate=%d", numFilters, freqBins, sampleRate)
	}

	nyquist := float64(sampleRate) / 2.0
	if highFreq > nyquist {
		highFreq = nyquist
	}

	// Convert to mel scale
	lowMel := s.hzToMel(lowFreq)
	highMel := s.hzToMel(highFreq)

	// Create mel points
	melPoints := make([]float64, numFilters+2)
	melStep := (highMel - lowMel) / float64(numFilters+1)
	for i := range melPoints {
		melPoints[i] = lowMel + float64(i)*melStep
	}

	// Convert back to Hz
	freqPoints := make([]float64, len(melPoints))
	for i, mel := range melPoints {
		freqPoints[i] = s.melToHz(mel)
	}

	// FIXED: Correct frequency resolution for FFT
	freqResolution := float64(sampleRate) / float64(2*(freqBins-1))

	// Create normalized filter bank
	filterBank := make([][]float64, numFilters)
	for i := 0; i < numFilters; i++ {
		filter := make([]float64, freqBins)
		filterSum := 0.0

		leftFreq := freqPoints[i]
		centerFreq := freqPoints[i+1]
		rightFreq := freqPoints[i+2]

		for j := 0; j < freqBins; j++ {
			freq := float64(j) * freqResolution

			if freq >= leftFreq && freq <= rightFreq {
				var weight float64
				if freq <= centerFreq {
					if centerFreq > leftFreq {
						weight = (freq - leftFreq) / (centerFreq - leftFreq)
					}
				} else {
					if rightFreq > centerFreq {
						weight = (rightFreq - freq) / (rightFreq - centerFreq)
					}
				}
				filter[j] = weight
				filterSum += weight
			}
		}

		// Normalize filter to preserve energy
		if filterSum > 0 {
			for j := range filter {
				filter[j] /= filterSum
			}
		}

		filterBank[i] = filter
	}

	return filterBank, nil
}

func (s *SpeechFeatureExtractor) applyMelFilters(magnitude []float64, filterBank [][]float64) []float64 {
	melSpectrum := make([]float64, len(filterBank))

	for i, filter := range filterBank {
		sum := 0.0
		for j := 0; j < len(filter) && j < len(magnitude); j++ {
			sum += magnitude[j] * filter[j]
		}

		// Ensure minimum value to prevent log issues
		if sum < 1e-8 {
			sum = 1e-8
		}

		melSpectrum[i] = sum
	}

	return melSpectrum
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

func (s *SpeechFeatureExtractor) calculateSpectralFlux(spectrogram *analyzers.SpectrogramResult) []float64 {
	if spectrogram.TimeFrames < 2 {
		return make([]float64, max(0, spectrogram.TimeFrames-1))
	}

	flux := make([]float64, spectrogram.TimeFrames-1)

	for t := 1; t < spectrogram.TimeFrames; t++ {
		sum := 0.0
		minLen := min(len(spectrogram.Magnitude[t]), len(spectrogram.Magnitude[t-1]))

		for f := range minLen {
			diff := spectrogram.Magnitude[t][f] - spectrogram.Magnitude[t-1][f]
			if diff > 0 { // Only positive changes (energy increases)
				sum += diff * diff
			}
		}
		flux[t-1] = math.Sqrt(sum)
	}

	return flux
}

func (s *SpeechFeatureExtractor) AuditFeatureUsage(features *ExtractedFeatures) {
	logger := s.logger.WithFields(logging.Fields{
		"function": "AuditFeatureUsage",
	})

	logger.Info("=== FEATURE USAGE AUDIT ===")

	if len(features.MFCC) > 0 {
		logger.Info("✅ MFCC features extracted", logging.Fields{
			"frames": len(features.MFCC),
			"coeffs": len(features.MFCC[0]),
		})
	} else {
		logger.Info("❌ MFCC features missing or empty")
	}

	if features.EnergyFeatures != nil && len(features.EnergyFeatures.ShortTimeEnergy) > 0 {
		logger.Info("✅ Energy features extracted", logging.Fields{
			"frames": len(features.EnergyFeatures.ShortTimeEnergy),
		})
	} else {
		logger.Info("❌ Energy features missing or empty")
	}

	if features.SpectralFeatures != nil && len(features.SpectralFeatures.SpectralCentroid) > 0 {
		logger.Info("✅ Spectral features extracted", logging.Fields{
			"frames": len(features.SpectralFeatures.SpectralCentroid),
		})
	} else {
		logger.Info("❌ Spectral features missing or empty")
	}

	if features.SpeechFeatures != nil {
		logger.Info("✅ Speech-specific features extracted")
	} else {
		logger.Info("❌ Speech-specific features missing")
	}

	if features.TemporalFeatures != nil && len(features.TemporalFeatures.RMSEnergy) > 0 {
		logger.Info("✅ Temporal features extracted", logging.Fields{
			"frames": len(features.TemporalFeatures.RMSEnergy),
		})
	} else {
		logger.Info("❌ Temporal features missing or empty")
	}

	logger.Info("=== END FEATURE AUDIT ===")
}

func (s *SpeechFeatureExtractor) normalizeMagnitude(magnitude []float64, stats map[string]float64) []float64 {
	normalized := make([]float64, len(magnitude))

	// Use 95th percentile as normalization factor to avoid outliers
	normFactor := stats["p95"]
	if normFactor == 0 {
		normFactor = 1.0
	}

	for i, mag := range magnitude {
		// Normalize and add small offset
		normalized[i] = (mag / normFactor) + 1e-8
	}

	return normalized
}

func (s *SpeechFeatureExtractor) computeLogMelSpectrum(melSpectrum []float64) []float64 {
	logMelSpectrum := make([]float64, len(melSpectrum))

	for i, val := range melSpectrum {
		// Use natural log for standard MFCC
		logMelSpectrum[i] = math.Log(val)
	}

	return logMelSpectrum
}

func (s *SpeechFeatureExtractor) applyDCT(logMelSpectrum []float64, numCoeffs int) []float64 {
	if len(logMelSpectrum) == 0 {
		return make([]float64, numCoeffs)
	}

	mfcc := make([]float64, numCoeffs)
	N := float64(len(logMelSpectrum))

	for k := 0; k < numCoeffs; k++ {
		sum := 0.0
		for n := 0; n < len(logMelSpectrum); n++ {
			sum += logMelSpectrum[n] * math.Cos(math.Pi*float64(k)*(float64(n)+0.5)/N)
		}

		// Standard DCT normalization
		normFactor := math.Sqrt(2.0 / N)
		if k == 0 {
			normFactor = math.Sqrt(1.0 / N)
		}

		mfcc[k] = sum * normFactor
	}

	return mfcc
}

func (s *SpeechFeatureExtractor) postProcessMFCC(mfcc [][]float64) [][]float64 {
	if len(mfcc) == 0 {
		return mfcc
	}

	// Calculate global statistics
	var allCoeffs []float64
	for _, frame := range mfcc {
		allCoeffs = append(allCoeffs, frame...)
	}

	if len(allCoeffs) == 0 {
		return mfcc
	}

	// Calculate mean and std
	mean := s.calculateMean(allCoeffs)
	std := math.Sqrt(s.calculateVariance(allCoeffs))

	// Z-score normalization if values are too extreme
	if std > 20 || math.Abs(mean) > 10 {
		s.logger.Info("Applying z-score normalization to MFCC", logging.Fields{
			"original_mean": mean,
			"original_std":  std,
		})

		for i := range mfcc {
			for j := range mfcc[i] {
				if std > 0 {
					mfcc[i][j] = (mfcc[i][j] - mean) / std * 5.0 // Scale to reasonable range
				}
			}
		}
	}

	// Final clamp with less aggressive bounds
	for i := range mfcc {
		for j := range mfcc[i] {
			if math.IsNaN(mfcc[i][j]) || math.IsInf(mfcc[i][j], 0) {
				mfcc[i][j] = 0.0
			} else if mfcc[i][j] > 25 {
				mfcc[i][j] = 25.0
			} else if mfcc[i][j] < -25 {
				mfcc[i][j] = -25.0
			}
		}
	}

	return mfcc
}

func (s *SpeechFeatureExtractor) normalizeMFCC(mfcc [][]float64) [][]float64 {
	if len(mfcc) == 0 {
		return mfcc
	}

	// Calculate statistics per coefficient
	numCoeffs := len(mfcc[0])
	coeffStats := make([]struct{ mean, std float64 }, numCoeffs)

	for c := 0; c < numCoeffs; c++ {
		values := make([]float64, len(mfcc))
		for t := 0; t < len(mfcc); t++ {
			values[t] = mfcc[t][c]
		}

		mean := s.calculateMean(values)
		std := math.Sqrt(s.calculateVariance(values))

		coeffStats[c].mean = mean
		coeffStats[c].std = std
	}

	// Apply coefficient-wise normalization
	for t := 0; t < len(mfcc); t++ {
		for c := 0; c < numCoeffs; c++ {
			// Handle extreme values based on coefficient type
			if c == 0 {
				// C0 (energy) - clamp to reasonable range
				if mfcc[t][c] > 20 {
					mfcc[t][c] = 20.0
				} else if mfcc[t][c] < -20 {
					mfcc[t][c] = -20.0
				}
			} else {
				// Other coefficients - use z-score normalization if needed
				if coeffStats[c].std > 15 {
					if coeffStats[c].std > 0 {
						mfcc[t][c] = (mfcc[t][c] - coeffStats[c].mean) / coeffStats[c].std * 5.0
					}
				}

				// Final clamp
				if mfcc[t][c] > 15 {
					mfcc[t][c] = 15.0
				} else if mfcc[t][c] < -15 {
					mfcc[t][c] = -15.0
				}
			}

			// Handle NaN/Inf
			if math.IsNaN(mfcc[t][c]) || math.IsInf(mfcc[t][c], 0) {
				mfcc[t][c] = 0.0
			}
		}
	}

	return mfcc
}
