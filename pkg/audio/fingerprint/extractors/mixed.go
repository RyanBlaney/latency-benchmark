package extractors

import (
	"fmt"
	"math"
	"sort"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/logging"
)

// MixedFeatureExtractor handles mixed content with robust, content-agnostic features
type MixedFeatureExtractor struct {
	config *config.FeatureConfig
	logger logging.Logger
}

// NewMixedFeatureExtractor creates a mixed content feature extractor
func NewMixedFeatureExtractor(featureConfig *config.FeatureConfig) *MixedFeatureExtractor {
	return &MixedFeatureExtractor{
		config: featureConfig,
		logger: logging.WithFields(logging.Fields{
			"component": "mixed_feature_extractor",
		}),
	}
}

func (m *MixedFeatureExtractor) GetName() string {
	return "MixedFeatureExtractor"
}

func (m *MixedFeatureExtractor) GetContentType() config.ContentType {
	return config.ContentMixed
}

func (m *MixedFeatureExtractor) GetFeatureWeights() map[string]float64 {
	if m.config.SimilarityWeights != nil {
		return m.config.SimilarityWeights
	}

	// Conservative weights for mixed content - emphasize most stable features
	return map[string]float64{
		"spectral": 0.30, // Most stable across all content types
		"energy":   0.25, // Universal applicability
		"mfcc":     0.20, // Reasonably stable across content
		"temporal": 0.15, // Content-dependent but useful
		"chroma":   0.10, // Least reliable for unknown content
	}
}

func (m *MixedFeatureExtractor) ExtractFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*ExtractedFeatures, error) {
	if spectrogram == nil {
		return nil, fmt.Errorf("spectrogram cannot be nil")
	}
	if len(pcm) == 0 {
		return nil, fmt.Errorf("PCM data cannot be empty")
	}
	if sampleRate <= 0 {
		return nil, fmt.Errorf("sample rate must be positive")
	}

	logger := m.logger.WithFields(logging.Fields{
		"function":  "ExtractFeatures",
		"frames":    spectrogram.TimeFrames,
		"freq_bins": spectrogram.FreqBins,
	})

	logger.Debug("Extracting mixed content features with robust approach")

	features := &ExtractedFeatures{
		ExtractionMetadata: make(map[string]any),
	}

	// Content analysis for adaptive feature extraction
	contentAnalysis := m.analyzeContentType(spectrogram, pcm, sampleRate)
	features.ExtractionMetadata["content_analysis"] = contentAnalysis

	// Extract robust spectral features (highest priority)
	spectralFeatures, err := m.extractRobustSpectralFeatures(spectrogram)
	if err != nil {
		logger.Error(err, "Failed to extract robust spectral features")
		return nil, err
	}
	features.SpectralFeatures = spectralFeatures

	// Extract energy features (universally applicable)
	energyFeatures, err := m.extractRobustEnergyFeatures(pcm, sampleRate)
	if err != nil {
		logger.Error(err, "Failed to extract energy features")
		return nil, err
	}
	features.EnergyFeatures = energyFeatures

	// Extract MFCC (reasonably stable across content types)
	if m.config.EnableMFCC {
		mfcc, err := m.extractAdaptiveMFCC(spectrogram, sampleRate, contentAnalysis)
		if err != nil {
			logger.Error(err, "Failed to extract MFCC")
		} else {
			features.MFCC = mfcc
		}
	}

	// Extract conservative temporal features
	if m.config.EnableTemporalFeatures {
		temporalFeatures, err := m.extractRobustTemporalFeatures(pcm, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract temporal features")
		} else {
			features.TemporalFeatures = temporalFeatures
		}
	}

	// Extract chroma features if enabled (lowest priority)
	if m.config.EnableChroma {
		chromaFeatures, err := m.extractConservativeChromaFeatures(spectrogram, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract chroma features")
		} else {
			features.ChromaFeatures = chromaFeatures
		}
	}

	// Add extraction metadata
	features.ExtractionMetadata["extractor_type"] = "mixed"
	features.ExtractionMetadata["robust_features"] = true
	features.ExtractionMetadata["adaptive_approach"] = true
	features.ExtractionMetadata["content_confidence"] = contentAnalysis.Confidence

	logger.Info("Mixed content feature extraction completed", logging.Fields{
		"content_type_guess": contentAnalysis.PredominantType,
		"confidence":         contentAnalysis.Confidence,
	})

	return features, nil
}

// ContentAnalysis represents the analysis of mixed content
type ContentAnalysis struct {
	PredominantType   string             `json:"predominant_type"`
	Confidence        float64            `json:"confidence"`
	ContentRatios     map[string]float64 `json:"content_ratios"`
	SpectralVariation float64            `json:"spectral_variation"`
	EnergyVariation   float64            `json:"energy_variation"`
	IsHighlyMixed     bool               `json:"is_highly_mixed"`
}

// analyzeContentType performs basic content analysis to guide feature extraction
func (m *MixedFeatureExtractor) analyzeContentType(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) *ContentAnalysis {
	logger := m.logger.WithFields(logging.Fields{
		"function": "analyzeContentType",
	})

	analysis := &ContentAnalysis{
		ContentRatios: make(map[string]float64),
	}

	// Analyze spectral characteristics
	spectralAnalysis := m.analyzeSpectralContent(spectrogram, sampleRate)

	// Analyze temporal characteristics
	temporalAnalysis := m.analyzeTemporalContent(pcm, sampleRate)

	// Analyze energy characteristics
	energyAnalysis := m.analyzeEnergyContent(pcm, sampleRate)

	// Determine predominant content type using all analyses
	analysis.ContentRatios["music"] = spectralAnalysis.MusicLikelihood
	analysis.ContentRatios["speech"] = spectralAnalysis.SpeechLikelihood
	analysis.ContentRatios["sports"] = energyAnalysis.SportsLikelihood
	analysis.ContentRatios["noise"] = spectralAnalysis.NoiseLikelihood

	// Adjust speech likelihood based on temporal characteristics
	if temporalAnalysis.SilenceRatio > 0.1 && temporalAnalysis.DynamicRange > 20 {
		analysis.ContentRatios["speech"] *= 1.2 // Boost speech likelihood
	}

	// Adjust music likelihood based on rhythm strength
	if temporalAnalysis.RhythmStrength > 0.5 {
		analysis.ContentRatios["music"] *= 1.1 // Boost music likelihood
	}

	// Find predominant type
	maxRatio := 0.0
	for contentType, ratio := range analysis.ContentRatios {
		if ratio > maxRatio {
			maxRatio = ratio
			analysis.PredominantType = contentType
		}
	}

	analysis.Confidence = maxRatio
	analysis.SpectralVariation = spectralAnalysis.Variation
	analysis.EnergyVariation = energyAnalysis.Variation
	analysis.IsHighlyMixed = maxRatio < 0.6 // No clear predominant type

	logger.Debug("Content analysis completed", logging.Fields{
		"predominant_type": analysis.PredominantType,
		"confidence":       analysis.Confidence,
		"highly_mixed":     analysis.IsHighlyMixed,
		"silence_ratio":    temporalAnalysis.SilenceRatio,
		"rhythm_strength":  temporalAnalysis.RhythmStrength,
	})

	return analysis
}

// SpectralAnalysis holds spectral content analysis results
type SpectralAnalysis struct {
	MusicLikelihood  float64
	SpeechLikelihood float64
	NoiseLikelihood  float64
	Variation        float64
}

// TemporalAnalysis holds temporal content analysis results
type TemporalAnalysis struct {
	DynamicRange   float64
	SilenceRatio   float64
	RhythmStrength float64
}

// EnergyAnalysis holds energy content analysis results
type EnergyAnalysis struct {
	SportsLikelihood float64
	Variation        float64
	PeakDensity      float64
}

func (m *MixedFeatureExtractor) analyzeSpectralContent(spectrogram *analyzers.SpectrogramResult, sampleRate int) *SpectralAnalysis {
	analysis := &SpectralAnalysis{}

	// Calculate average spectral characteristics
	avgCentroid := 0.0
	avgRolloff := 0.0
	avgFlatness := 0.0
	spectralVariances := make([]float64, 0)

	freqs := make([]float64, spectrogram.FreqBins)
	for i := 0; i < spectrogram.FreqBins; i++ {
		freqs[i] = float64(i) * float64(sampleRate) / float64((spectrogram.FreqBins-1)*2)
	}

	for t := 0; t < spectrogram.TimeFrames; t++ {
		magnitude := spectrogram.Magnitude[t]

		centroid := m.calculateSpectralCentroid(magnitude, freqs)
		rolloff := m.calculateSpectralRolloff(magnitude, freqs, 0.85)
		flatness := m.calculateSpectralFlatness(magnitude)

		avgCentroid += centroid
		avgRolloff += rolloff
		avgFlatness += flatness

		// Calculate spectral variance for this frame
		variance := m.calculateSpectralVariance(magnitude)
		spectralVariances = append(spectralVariances, variance)
	}

	if spectrogram.TimeFrames > 0 {
		avgCentroid /= float64(spectrogram.TimeFrames)
		avgRolloff /= float64(spectrogram.TimeFrames)
		avgFlatness /= float64(spectrogram.TimeFrames)
	}

	// Determine content likelihoods based on spectral characteristics

	// Music likelihood: moderate centroid, high spectral variance, low flatness
	musicScore := 0.0
	if avgCentroid > 1000 && avgCentroid < 3000 {
		musicScore += 0.4
	}
	if avgFlatness < 0.5 {
		musicScore += 0.3
	}
	avgSpectralVariance := m.calculateMean(spectralVariances)
	if avgSpectralVariance > 0.1 {
		musicScore += 0.3
	}
	analysis.MusicLikelihood = musicScore

	// Speech likelihood: centroid in speech range, moderate flatness
	speechScore := 0.0
	if avgCentroid > 300 && avgCentroid < 2000 {
		speechScore += 0.5
	}
	if avgFlatness > 0.3 && avgFlatness < 0.7 {
		speechScore += 0.3
	}
	if avgRolloff > 1000 && avgRolloff < 4000 {
		speechScore += 0.2
	}
	analysis.SpeechLikelihood = speechScore

	// Noise likelihood: high flatness, low spectral structure
	noiseScore := 0.0
	if avgFlatness > 0.7 {
		noiseScore += 0.6
	}
	if avgSpectralVariance < 0.05 {
		noiseScore += 0.4
	}
	analysis.NoiseLikelihood = noiseScore

	// Overall spectral variation
	analysis.Variation = m.calculateVariance(spectralVariances)

	return analysis
}

func (m *MixedFeatureExtractor) analyzeTemporalContent(pcm []float64, sampleRate int) *TemporalAnalysis {
	analysis := &TemporalAnalysis{}

	// Calculate frame-based energy
	frameSize := sampleRate / 20 // 50ms frames
	hopSize := frameSize / 2

	energies := make([]float64, 0)
	for i := 0; i < len(pcm)-frameSize; i += hopSize {
		energy := 0.0
		for j := 0; j < frameSize && i+j < len(pcm); j++ {
			energy += pcm[i+j] * pcm[i+j]
		}
		energies = append(energies, math.Sqrt(energy/float64(frameSize)))
	}

	if len(energies) > 0 {
		analysis.DynamicRange = m.calculateDynamicRange(energies)
		analysis.SilenceRatio = m.calculateSilenceRatio(energies)
		analysis.RhythmStrength = m.calculateRhythmStrength(energies)
	}

	return analysis
}

func (m *MixedFeatureExtractor) analyzeEnergyContent(pcm []float64, sampleRate int) *EnergyAnalysis {
	analysis := &EnergyAnalysis{}

	// Calculate frame-based energy analysis
	frameSize := sampleRate / 25 // 40ms frames
	hopSize := frameSize / 2

	energies := make([]float64, 0)
	peaks := make([]float64, 0)

	for i := 0; i < len(pcm)-frameSize; i += hopSize {
		energy := 0.0
		maxVal := 0.0
		for j := 0; j < frameSize && i+j < len(pcm); j++ {
			sample := pcm[i+j]
			energy += sample * sample
			if math.Abs(sample) > maxVal {
				maxVal = math.Abs(sample)
			}
		}
		energy = math.Sqrt(energy / float64(frameSize))
		energies = append(energies, energy)
		peaks = append(peaks, maxVal)
	}

	if len(energies) > 0 {
		// Sports likelihood: high energy variation, frequent peaks
		energyVariation := m.calculateVariance(energies)
		meanEnergy := m.calculateMean(energies)

		peakCount := 0
		threshold := meanEnergy * 2.0
		for _, energy := range energies {
			if energy > threshold {
				peakCount++
			}
		}

		analysis.PeakDensity = float64(peakCount) / float64(len(energies))
		analysis.Variation = energyVariation

		// Sports likelihood based on high variation and peak density
		sportsScore := 0.0
		if energyVariation > meanEnergy*0.5 {
			sportsScore += 0.4
		}
		if analysis.PeakDensity > 0.1 {
			sportsScore += 0.6
		}
		analysis.SportsLikelihood = sportsScore
	}

	return analysis
}

// extractRobustSpectralFeatures extracts spectral features optimized for mixed content
func (m *MixedFeatureExtractor) extractRobustSpectralFeatures(spectrogram *analyzers.SpectrogramResult) (*SpectralFeatures, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function": "extractRobustSpectralFeatures",
	})

	features := &SpectralFeatures{
		SpectralCentroid:  make([]float64, spectrogram.TimeFrames),
		SpectralRolloff:   make([]float64, spectrogram.TimeFrames),
		SpectralBandwidth: make([]float64, spectrogram.TimeFrames),
		SpectralFlatness:  make([]float64, spectrogram.TimeFrames),
		SpectralCrest:     make([]float64, spectrogram.TimeFrames),
		SpectralSlope:     make([]float64, spectrogram.TimeFrames),
		ZeroCrossingRate:  make([]float64, spectrogram.TimeFrames),
		SpectralContrast:  make([][]float64, spectrogram.TimeFrames),
	}

	// Generate frequency bins
	freqs := make([]float64, spectrogram.FreqBins)
	for i := 0; i < spectrogram.FreqBins; i++ {
		freqs[i] = float64(i) * float64(spectrogram.SampleRate) / float64((spectrogram.FreqBins-1)*2)
	}

	// Extract robust spectral features for each frame
	for t := 0; t < spectrogram.TimeFrames; t++ {
		magnitude := spectrogram.Magnitude[t]

		// Standard spectral features with robust calculations
		features.SpectralCentroid[t] = m.calculateRobustSpectralCentroid(magnitude, freqs)
		features.SpectralRolloff[t] = m.calculateSpectralRolloff(magnitude, freqs, 0.85)
		features.SpectralBandwidth[t] = m.calculateSpectralBandwidth(magnitude, freqs, features.SpectralCentroid[t])
		features.SpectralFlatness[t] = m.calculateSpectralFlatness(magnitude)
		features.SpectralCrest[t] = m.calculateSpectralCrest(magnitude)
		features.SpectralSlope[t] = m.calculateSpectralSlope(magnitude, freqs)

		// Conservative spectral contrast (fewer bands, more stable)
		contrastBands := 4 // Conservative for mixed content
		features.SpectralContrast[t] = m.calculateRobustSpectralContrast(magnitude, contrastBands)
	}

	// Calculate robust spectral flux
	features.SpectralFlux = m.calculateRobustSpectralFlux(spectrogram)

	logger.Debug("Robust spectral features extracted", logging.Fields{
		"avg_centroid": m.calculateMean(features.SpectralCentroid),
		"avg_rolloff":  m.calculateMean(features.SpectralRolloff),
		"avg_flatness": m.calculateMean(features.SpectralFlatness),
	})

	return features, nil
}

// extractRobustEnergyFeatures extracts energy features suitable for all content types
func (m *MixedFeatureExtractor) extractRobustEnergyFeatures(pcm []float64, sampleRate int) (*EnergyFeatures, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function": "extractRobustEnergyFeatures",
	})

	features := &EnergyFeatures{}

	// Conservative frame size (30ms - between music and speech)
	frameSize := sampleRate * 30 / 1000
	hopSize := frameSize / 2

	numFrames := (len(pcm)-frameSize)/hopSize + 1
	if numFrames <= 0 {
		return features, fmt.Errorf("insufficient audio length for mixed content energy analysis")
	}

	features.ShortTimeEnergy = make([]float64, numFrames)
	features.EnergyEntropy = make([]float64, numFrames)
	features.CrestFactor = make([]float64, numFrames)

	energies := make([]float64, numFrames)

	// Calculate robust frame-based energy features
	for i := 0; i < numFrames; i++ {
		start := i * hopSize
		end := start + frameSize
		if end > len(pcm) {
			end = len(pcm)
		}

		frame := pcm[start:end]

		// Short-time energy with robust calculation
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

		// Crest factor (robust across content types)
		rms := math.Sqrt(energy)
		if rms > 0 {
			features.CrestFactor[i] = maxVal / rms
		}

		// Energy entropy (universal content discriminator)
		features.EnergyEntropy[i] = m.calculateRobustEnergyEntropy(frame)
	}

	// Global energy characteristics
	features.EnergyVariance = m.calculateVariance(energies)
	features.LoudnessRange = m.calculateRobustLoudnessRange(energies)

	logger.Debug("Robust energy features extracted", logging.Fields{
		"energy_variance": features.EnergyVariance,
		"loudness_range":  features.LoudnessRange,
		"avg_crest":       m.calculateMean(features.CrestFactor),
	})

	return features, nil
}

// extractAdaptiveMFCC extracts MFCC adapted based on content analysis
func (m *MixedFeatureExtractor) extractAdaptiveMFCC(spectrogram *analyzers.SpectrogramResult, sampleRate int, contentAnalysis *ContentAnalysis) ([][]float64, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function":         "extractAdaptiveMFCC",
		"predominant_type": contentAnalysis.PredominantType,
	})

	numCoeffs := m.config.MFCCCoefficients
	if numCoeffs == 0 {
		numCoeffs = 13
	}

	mfcc := make([][]float64, spectrogram.TimeFrames)

	// Adaptive frequency range based on content analysis
	var lowFreq, highFreq float64
	if contentAnalysis.PredominantType == "speech" {
		lowFreq = 300.0
		highFreq = 4000.0
	} else if contentAnalysis.PredominantType == "music" {
		lowFreq = 80.0
		highFreq = 8000.0
	} else {
		// Conservative range for mixed/unknown content
		lowFreq = 200.0
		highFreq = 6000.0
	}

	// Create adaptive mel filter bank
	numMelFilters := 26
	melFilters := m.createAdaptiveMelFilterBank(numMelFilters, lowFreq, highFreq, spectrogram.FreqBins, sampleRate)

	// Process each time frame
	for t := 0; t < spectrogram.TimeFrames; t++ {
		magnitude := spectrogram.Magnitude[t]

		// Apply mel filter bank
		melSpectrum := m.applyMelFilters(magnitude, melFilters)

		// Take logarithm with robust floor
		logMelSpectrum := make([]float64, len(melSpectrum))
		for i, val := range melSpectrum {
			if val > 1e-10 {
				logMelSpectrum[i] = math.Log(val)
			} else {
				logMelSpectrum[i] = math.Log(1e-10)
			}
		}

		// Apply DCT
		mfcc[t] = m.applyDCT(logMelSpectrum, numCoeffs)
	}

	logger.Debug("Adaptive MFCC features extracted", logging.Fields{
		"mfcc_frames": len(mfcc),
		"mfcc_coeffs": numCoeffs,
		"freq_range":  fmt.Sprintf("%.0f-%.0fHz", lowFreq, highFreq),
	})

	return mfcc, nil
}

// extractRobustTemporalFeatures extracts temporal features robust across content types
func (m *MixedFeatureExtractor) extractRobustTemporalFeatures(pcm []float64, sampleRate int) (*TemporalFeatures, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function": "extractRobustTemporalFeatures",
	})

	features := &TemporalFeatures{}

	// Conservative frame size for mixed content
	frameSize := sampleRate * 30 / 1000 // 30ms frames
	hopSize := frameSize / 2

	numFrames := (len(pcm)-frameSize)/hopSize + 1
	if numFrames <= 0 {
		return features, fmt.Errorf("insufficient audio length for temporal analysis")
	}

	features.RMSEnergy = make([]float64, numFrames)
	features.AttackTime = make([]float64, 0)
	features.DecayTime = make([]float64, 0)

	energies := make([]float64, numFrames)

	// Calculate frame-based features
	for i := 0; i < numFrames; i++ {
		start := i * hopSize
		end := start + frameSize
		if end > len(pcm) {
			end = len(pcm)
		}

		frame := pcm[start:end]

		// RMS Energy
		rms := 0.0
		for _, sample := range frame {
			rms += sample * sample
		}
		rms = math.Sqrt(rms / float64(len(frame)))
		features.RMSEnergy[i] = rms
		energies[i] = rms
	}

	// Conservative temporal characteristics
	features.DynamicRange = m.calculateRobustDynamicRange(pcm)
	features.SilenceRatio = m.calculateRobustSilenceRatio(energies)
	features.PeakAmplitude = m.calculatePeakAmplitude(pcm)
	features.AverageAmplitude = m.calculateAverageAmplitude(pcm)

	// Conservative onset detection
	features.OnsetDensity = m.calculateRobustOnsetDensity(energies, float64(sampleRate)/float64(hopSize))
	features.TempoVariation = m.calculateRobustTempoVariation(energies)

	logger.Debug("Robust temporal features extracted", logging.Fields{
		"dynamic_range":   features.DynamicRange,
		"silence_ratio":   features.SilenceRatio,
		"onset_density":   features.OnsetDensity,
		"tempo_variation": features.TempoVariation,
	})

	return features, nil
}

// extractConservativeChromaFeatures extracts chroma features with conservative approach
func (m *MixedFeatureExtractor) extractConservativeChromaFeatures(spectrogram *analyzers.SpectrogramResult, sampleRate int) ([][]float64, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function":    "extractConservativeChromaFeatures",
		"chroma_bins": m.config.ChromaBins,
	})

	chromaBins := m.config.ChromaBins
	if chromaBins == 0 {
		chromaBins = 12 // Standard chromagram
	}

	chroma := make([][]float64, spectrogram.TimeFrames)

	// Generate frequency bins
	freqs := make([]float64, spectrogram.FreqBins)
	for i := 0; i < spectrogram.FreqBins; i++ {
		freqs[i] = float64(i) * float64(sampleRate) / float64((spectrogram.FreqBins-1)*2)
	}

	// Process each time frame with conservative approach
	for t := 0; t < spectrogram.TimeFrames; t++ {
		chroma[t] = make([]float64, chromaBins)
		magnitude := spectrogram.Magnitude[t]

		// Conservative frequency range (avoid very low and very high frequencies)
		for f := 0; f < len(magnitude); f++ {
			freq := freqs[f]
			if freq < 100 || freq > 4000 {
				continue // Skip frequencies outside conservative range
			}

			// Convert frequency to MIDI note number
			if freq > 0 {
				midiNote := 12*math.Log2(freq/440.0) + 69
				if midiNote >= 0 {
					// Map to chroma class
					chromaClass := int(math.Round(midiNote)) % chromaBins
					if chromaClass >= 0 && chromaClass < chromaBins {
						chroma[t][chromaClass] += magnitude[f]
					}
				}
			}
		}

		// Conservative normalization
		maxVal := 0.0
		for i := range chroma[t] {
			if chroma[t][i] > maxVal {
				maxVal = chroma[t][i]
			}
		}
		if maxVal > 0 {
			for i := range chroma[t] {
				chroma[t][i] /= maxVal
			}
		}
	}

	logger.Debug("Conservative chroma features extracted", logging.Fields{
		"chroma_frames": len(chroma),
		"chroma_bins":   chromaBins,
	})

	return chroma, nil
}

// Robust helper methods for mixed content

func (m *MixedFeatureExtractor) calculateRobustSpectralCentroid(magnitude, freqs []float64) float64 {
	// Robust spectral centroid calculation with outlier handling
	numerator := 0.0
	denominator := 0.0

	// Skip very low and very high frequencies for robustness
	for i := 0; i < len(magnitude) && i < len(freqs); i++ {
		if freqs[i] < 50 || freqs[i] > 8000 {
			continue
		}
		numerator += freqs[i] * magnitude[i]
		denominator += magnitude[i]
	}

	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

func (m *MixedFeatureExtractor) calculateRobustSpectralContrast(magnitude []float64, numBands int) []float64 {
	// Conservative spectral contrast with fewer bands
	contrast := make([]float64, numBands)
	bandSize := len(magnitude) / numBands

	for band := 0; band < numBands; band++ {
		start := band * bandSize
		end := start + bandSize
		if end > len(magnitude) {
			end = len(magnitude)
		}

		if start >= end {
			continue
		}

		bandMag := magnitude[start:end]
		if len(bandMag) < 10 {
			continue
		}

		// Sort to find percentiles
		sorted := make([]float64, len(bandMag))
		copy(sorted, bandMag)
		sort.Float64s(sorted)

		// Use conservative percentiles (20th and 80th)
		valley := sorted[len(sorted)/5] // 20th percentile
		peak := sorted[4*len(sorted)/5] // 80th percentile

		if valley > 0 {
			contrast[band] = math.Log(peak / valley)
		}
	}

	return contrast
}

func (m *MixedFeatureExtractor) calculateRobustSpectralFlux(spectrogram *analyzers.SpectrogramResult) []float64 {
	// Robust spectral flux calculation
	if spectrogram.TimeFrames <= 1 {
		return nil
	}

	flux := make([]float64, spectrogram.TimeFrames-1)

	for t := 1; t < spectrogram.TimeFrames; t++ {
		sum := 0.0
		for f := 0; f < spectrogram.FreqBins; f++ {
			diff := spectrogram.Magnitude[t][f] - spectrogram.Magnitude[t-1][f]
			// Conservative approach: only consider positive changes
			if diff > 0 {
				sum += diff * diff
			}
		}
		flux[t-1] = math.Sqrt(sum)
	}

	return flux
}

func (m *MixedFeatureExtractor) calculateRobustEnergyEntropy(frame []float64) float64 {
	// Conservative energy entropy calculation
	if len(frame) == 0 {
		return 0
	}

	// Use moderate number of sub-frames
	numSubFrames := 8
	subFrameSize := len(frame) / numSubFrames

	if subFrameSize == 0 {
		return 0
	}

	subFrameEnergies := make([]float64, numSubFrames)
	totalEnergy := 0.0

	for i := 0; i < numSubFrames; i++ {
		start := i * subFrameSize
		end := start + subFrameSize
		if end > len(frame) {
			end = len(frame)
		}

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

	entropy := 0.0
	for _, energy := range subFrameEnergies {
		if energy > 0 {
			prob := energy / totalEnergy
			entropy -= prob * math.Log2(prob)
		}
	}

	return entropy
}

func (m *MixedFeatureExtractor) calculateRobustLoudnessRange(energies []float64) float64 {
	// Conservative loudness range calculation
	if len(energies) == 0 {
		return 0
	}

	sorted := make([]float64, len(energies))
	copy(sorted, energies)
	sort.Float64s(sorted)

	// Use conservative percentiles (10th and 90th)
	p10 := sorted[len(sorted)/10]   // 10th percentile
	p90 := sorted[9*len(sorted)/10] // 90th percentile

	if p10 > 0 {
		return 20 * math.Log10(p90/p10)
	}

	return 0
}

func (m *MixedFeatureExtractor) calculateRobustDynamicRange(pcm []float64) float64 {
	// Conservative dynamic range calculation
	if len(pcm) == 0 {
		return 0
	}

	sorted := make([]float64, len(pcm))
	for i, sample := range pcm {
		sorted[i] = math.Abs(sample)
	}
	sort.Float64s(sorted)

	// Use conservative percentiles (5th and 95th)
	p5 := sorted[len(sorted)/20]     // 5th percentile
	p95 := sorted[19*len(sorted)/20] // 95th percentile

	if p5 > 1e-10 {
		return 20 * math.Log10(p95/p5)
	}

	return 0
}

func (m *MixedFeatureExtractor) calculateRobustSilenceRatio(energies []float64) float64 {
	// Conservative silence ratio calculation
	if len(energies) == 0 {
		return 0
	}

	// Use median-based threshold for robustness
	sorted := make([]float64, len(energies))
	copy(sorted, energies)
	sort.Float64s(sorted)

	median := sorted[len(sorted)/2]
	threshold := median * 0.1 // 10% of median

	silentFrames := 0
	for _, energy := range energies {
		if energy < threshold {
			silentFrames++
		}
	}

	return float64(silentFrames) / float64(len(energies))
}

func (m *MixedFeatureExtractor) calculateRobustOnsetDensity(energies []float64, frameRate float64) float64 {
	// Conservative onset detection
	if len(energies) < 10 {
		return 0
	}

	onsets := 0
	threshold := 1.5 // Conservative threshold

	// Use adaptive thresholding
	windowSize := 5
	for i := windowSize; i < len(energies)-1; i++ {
		current := energies[i]

		// Local average
		localAvg := 0.0
		for j := i - windowSize; j < i; j++ {
			if j >= 0 {
				localAvg += energies[j]
			}
		}
		localAvg /= float64(windowSize)

		// Conservative onset detection
		if current > localAvg*threshold && current > energies[i+1] {
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

func (m *MixedFeatureExtractor) calculateRobustTempoVariation(energies []float64) float64 {
	// Conservative tempo variation calculation
	if len(energies) < 20 {
		return 0
	}

	// Find peaks with conservative approach
	peaks := make([]int, 0)
	windowSize := 3
	meanEnergy := m.calculateMean(energies)
	threshold := meanEnergy * 1.2 // Conservative threshold

	for i := windowSize; i < len(energies)-windowSize; i++ {
		current := energies[i]

		// Check if it's a local maximum
		isLocalMax := true
		for j := i - windowSize; j <= i+windowSize; j++ {
			if j != i && energies[j] >= current {
				isLocalMax = false
				break
			}
		}

		if isLocalMax && current > threshold {
			peaks = append(peaks, i)
		}
	}

	if len(peaks) < 2 {
		return 0
	}

	// Calculate intervals between peaks
	intervals := make([]float64, 0)
	for i := 1; i < len(peaks); i++ {
		interval := float64(peaks[i] - peaks[i-1])
		intervals = append(intervals, interval)
	}

	if len(intervals) == 0 {
		return 0
	}

	// Calculate coefficient of variation
	mean := m.calculateMean(intervals)
	variance := m.calculateVariance(intervals)

	if mean == 0 {
		return 0
	}

	return math.Sqrt(variance) / mean
}

func (m *MixedFeatureExtractor) calculateSpectralVariance(magnitude []float64) float64 {
	// Calculate spectral variance for content analysis
	if len(magnitude) <= 1 {
		return 0
	}

	mean := m.calculateMean(magnitude)
	variance := 0.0

	for _, mag := range magnitude {
		diff := mag - mean
		variance += diff * diff
	}

	return variance / float64(len(magnitude))
}

func (m *MixedFeatureExtractor) calculateDynamicRange(energies []float64) float64 {
	// Calculate dynamic range from energy values
	if len(energies) == 0 {
		return 0
	}

	maxEnergy := 0.0
	minEnergy := math.Inf(1)

	for _, energy := range energies {
		if energy > maxEnergy {
			maxEnergy = energy
		}
		if energy < minEnergy && energy > 1e-10 {
			minEnergy = energy
		}
	}

	if minEnergy == 0 || minEnergy == math.Inf(1) {
		return 0
	}

	return 20 * math.Log10(maxEnergy/minEnergy)
}

func (m *MixedFeatureExtractor) calculateSilenceRatio(energies []float64) float64 {
	// Calculate silence ratio from energy values
	if len(energies) == 0 {
		return 0
	}

	meanEnergy := m.calculateMean(energies)
	threshold := meanEnergy * 0.1

	silentFrames := 0
	for _, energy := range energies {
		if energy < threshold {
			silentFrames++
		}
	}

	return float64(silentFrames) / float64(len(energies))
}

func (m *MixedFeatureExtractor) calculateRhythmStrength(energies []float64) float64 {
	// Simple rhythm strength calculation
	if len(energies) < 10 {
		return 0
	}

	// Calculate autocorrelation-like measure
	rhythmStrength := 0.0
	for lag := 1; lag < len(energies)/4; lag++ {
		correlation := 0.0
		count := 0
		for i := 0; i < len(energies)-lag; i++ {
			correlation += energies[i] * energies[i+lag]
			count++
		}
		if count > 0 {
			correlation /= float64(count)
		}
		rhythmStrength += correlation
	}

	return rhythmStrength
}

// Mel filter bank creation for mixed content
func (m *MixedFeatureExtractor) createAdaptiveMelFilterBank(numFilters int, lowFreq, highFreq float64, freqBins, sampleRate int) [][]float64 {
	// Create mel filter bank with adaptive frequency range
	lowMel := 2595.0 * math.Log10(1.0+lowFreq/700.0)
	highMel := 2595.0 * math.Log10(1.0+highFreq/700.0)

	melPoints := make([]float64, numFilters+2)
	melStep := (highMel - lowMel) / float64(numFilters+1)
	for i := range melPoints {
		melPoints[i] = lowMel + float64(i)*melStep
	}

	freqPoints := make([]float64, len(melPoints))
	for i, mel := range melPoints {
		freqPoints[i] = 700.0 * (math.Pow(10, mel/2595.0) - 1.0)
	}

	filterBank := make([][]float64, numFilters)
	for i := 0; i < numFilters; i++ {
		filter := make([]float64, freqBins)

		leftFreq := freqPoints[i]
		centerFreq := freqPoints[i+1]
		rightFreq := freqPoints[i+2]

		for j := 0; j < freqBins; j++ {
			freq := float64(j) * float64(sampleRate) / float64(freqBins*2)

			if freq >= leftFreq && freq <= rightFreq {
				if freq <= centerFreq {
					if centerFreq > leftFreq {
						filter[j] = (freq - leftFreq) / (centerFreq - leftFreq)
					}
				} else {
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

// Common helper methods

func (m *MixedFeatureExtractor) applyMelFilters(magnitude []float64, filterBank [][]float64) []float64 {
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

func (m *MixedFeatureExtractor) applyDCT(logMelSpectrum []float64, numCoeffs int) []float64 {
	mfcc := make([]float64, numCoeffs)
	N := float64(len(logMelSpectrum))

	for k := 0; k < numCoeffs; k++ {
		sum := 0.0
		for n := 0; n < len(logMelSpectrum); n++ {
			sum += logMelSpectrum[n] * math.Cos(math.Pi*float64(k)*(float64(n)+0.5)/N)
		}
		mfcc[k] = sum
	}

	return mfcc
}

func (m *MixedFeatureExtractor) calculateSpectralCentroid(magnitude, freqs []float64) float64 {
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

func (m *MixedFeatureExtractor) calculateSpectralRolloff(magnitude, freqs []float64, threshold float64) float64 {
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

func (m *MixedFeatureExtractor) calculateSpectralBandwidth(magnitude, freqs []float64, centroid float64) float64 {
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

func (m *MixedFeatureExtractor) calculateSpectralFlatness(magnitude []float64) float64 {
	if len(magnitude) == 0 {
		return 0
	}

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

func (m *MixedFeatureExtractor) calculateSpectralCrest(magnitude []float64) float64 {
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

func (m *MixedFeatureExtractor) calculateSpectralSlope(magnitude, freqs []float64) float64 {
	if len(magnitude) != len(freqs) || len(magnitude) < 2 {
		return 0
	}

	n := 0
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumXX := 0.0

	for i := 0; i < len(magnitude); i++ {
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

func (m *MixedFeatureExtractor) calculatePeakAmplitude(pcm []float64) float64 {
	maxVal := 0.0
	for _, sample := range pcm {
		if math.Abs(sample) > maxVal {
			maxVal = math.Abs(sample)
		}
	}
	return maxVal
}

func (m *MixedFeatureExtractor) calculateAverageAmplitude(pcm []float64) float64 {
	if len(pcm) == 0 {
		return 0
	}

	sum := 0.0
	for _, sample := range pcm {
		sum += math.Abs(sample)
	}
	return sum / float64(len(pcm))
}

func (m *MixedFeatureExtractor) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, val := range values {
		sum += val
	}
	return sum / float64(len(values))
}

func (m *MixedFeatureExtractor) calculateVariance(values []float64) float64 {
	if len(values) <= 1 {
		return 0
	}

	mean := m.calculateMean(values)
	variance := 0.0

	for _, val := range values {
		diff := val - mean
		variance += diff * diff
	}

	return variance / float64(len(values))
}
