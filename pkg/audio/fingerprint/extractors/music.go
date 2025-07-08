package extractors

import (
	"fmt"
	"math"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/logging"
)

// MusicFeatureExtractor extracts features optimized for music content
type MusicFeatureExtractor struct {
	config *config.FeatureConfig
	logger logging.Logger
}

// NewMusicFeatureExtractor creates a music-specific feature extractor
func NewMusicFeatureExtractor(featureConfig *config.FeatureConfig) *MusicFeatureExtractor {
	return &MusicFeatureExtractor{
		config: featureConfig,
		logger: logging.WithFields(logging.Fields{
			"component": "music_feature_extractor",
		}),
	}
}

func (m *MusicFeatureExtractor) GetName() string {
	return "MusicFeatureExtractor"
}

func (m *MusicFeatureExtractor) GetContentType() config.ContentType {
	return config.ContentMusic
}

func (m *MusicFeatureExtractor) GetFeatureWeights() map[string]float64 {
	if m.config.SimilarityWeights != nil {
		return m.config.SimilarityWeights
	}

	// Default weights for music content
	return map[string]float64{
		"chroma":            0.35,
		"spectral_contrast": 0.25,
		"mfcc":              0.15,
		"temporal":          0.15,
		"energy":            0.10,
	}
}

func (m *MusicFeatureExtractor) ExtractFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*ExtractedFeatures, error) {
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
		"function":  "ExtractFeature",
		"frames":    spectrogram.TimeFrames,
		"freq_bins": spectrogram.FreqBins,
	})

	logger.Debug("Extracting music-specific features")

	features := &ExtractedFeatures{
		ExtractionMetadata: make(map[string]any),
	}

	// Extract spectral features (always included)
	spectralFeatures, err := m.extractSpectralFeatures(spectrogram, pcm)
	if err != nil {
		logger.Error(err, "Failed to extract spectral features")
		return nil, err
	}
	features.SpectralFeatures = spectralFeatures

	// Extracct chroma features (high priority for music)
	if m.config.EnableChroma {
		chromaFeatures, err := m.extractChromaFeatures(spectrogram)
		if err != nil {
			logger.Error(err, "Failed to extract chroma features")
		} else {
			features.ChromaFeatures = chromaFeatures
		}
	}

	// Extract harmonic features
	harmonicFeatures, err := m.extractHarmonicFeatures(spectrogram, pcm, sampleRate)
	if err != nil {
		logger.Error(err, "Failed to extract harmonic features")
	} else {
		features.HarmonicFeatures = harmonicFeatures
	}

	// Extract MFCC (lower priority for music)
	if m.config.EnableMFCC {
		mfcc, err := m.extractMFCC(spectrogram)
		if err != nil {
			logger.Error(err, "Failed to extract MFCC")
		} else {
			features.MFCC = mfcc
		}
	}

	// Extract temporal features
	if m.config.EnableTemporalFeatures {
		temporalFeatures, err := m.extractTemporalFeatures(pcm, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract temporal features")
		} else {
			features.TemporalFeatures = temporalFeatures
		}
	}

	// Extract energy features
	energyFeatures, err := m.extractEnergyFeatures(pcm, sampleRate)
	if err != nil {
		logger.Error(err, "Failed to extract energy features")
	} else {
		features.EnergyFeatures = energyFeatures
	}

	// Add extraction metadata
	features.ExtractionMetadata["extractor_type"] = "music"
	features.ExtractionMetadata["chroma_enabled"] = m.config.EnableChroma
	features.ExtractionMetadata["mfcc_enabled"] = m.config.EnableMFCC
	features.ExtractionMetadata["chroma_bins"] = m.config.ChromaBins

	logger.Info("Music features extraction completed")
	return features, nil
}

func (m *MusicFeatureExtractor) extractSpectralFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64) (*SpectralFeatures, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function": "extractSpectralFeatures",
		"frames":   spectrogram.TimeFrames,
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
	for i := range spectrogram.FreqBins {
		freqs[i] = float64(i) * float64(spectrogram.SampleRate) / float64((spectrogram.FreqBins-1)*2)
	}

	hopSize := spectrogram.HopSize
	frameSize := hopSize * 2

	// Extract features for each time frame
	for t := range spectrogram.TimeFrames {
		magnitude := spectrogram.Magnitude[t]

		// Spectral Centroid (brightness/timbral centroid)
		features.SpectralCentroid[t] = m.calculateSpectralCentroid(magnitude, freqs)

		// Spectral Rolloff (85% of energy threshold for music)
		features.SpectralRolloff[t] = m.calculateSpectralRolloff(magnitude, freqs, 0.85)

		// Spectral Bandwidth (spread around centroid)
		features.SpectralBandwidth[t] = m.calculateSpectralFlatness(magnitude)

		// Spectral Crest (peakiness)
		features.SpectralCrest[t] = m.calculateSpectralCrest(magnitude)

		// Spectral Slope (timbral brightness trend)
		features.SpectralSlope[t] = m.calculateSpectralSlope(magnitude, freqs)

		// Spectral Contrast (difference between peaks and valleys)
		features.SpectralContrast[t] = m.calculateSpectralContrast(magnitude, m.config.ContrastBands)

		// Calculate spectral flux (onset detection)
		features.SpectralFlux = m.calculateSpectralContrast(magnitude, m.config.ContrastBands)

		// Calculate zero crossing rate
		start := t * hopSize
		end := start * frameSize
		end = min(end, len(pcm))

		if start < len(pcm) {
			pcmFrame := pcm[start:end]
			features.ZeroCrossingRate[t] = calculateZeroCrossingRate(pcmFrame)
		}
	}

	logger.Debug("Spectral features extracted", logging.Fields{
		"avg_centroid": m.calculateMean(features.SpectralCentroid),
		"avg_rolloff":  m.calculateMean(features.SpectralRolloff),
		"avg_flatness": m.calculateMean(features.SpectralFlatness),
	})

	return features, nil
}

func (m *MusicFeatureExtractor) extractChromaFeatures(spectrogram *analyzers.SpectrogramResult) ([][]float64, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function":    "extractChromaFeatures",
		"chroma_bins": m.config.ChromaBins,
	})

	chromaBins := m.config.ChromaBins
	if chromaBins == 0 {
		chromaBins = 12 // Standard 12-semitone chromagram
	}

	chroma := make([][]float64, spectrogram.TimeFrames)

	// Generate frequency bins
	freqs := make([]float64, spectrogram.FreqBins)
	for i := range spectrogram.FreqBins {
		freqs[i] = float64(i) * float64(spectrogram.SampleRate) / float64((spectrogram.FreqBins-1)*2)
	}

	// Process each time frame
	for t := range spectrogram.TimeFrames {
		chroma[t] = make([]float64, chromaBins)
		magnitude := spectrogram.Magnitude[t]

		// Map frequency bins to chroma classes
		for f := range len(magnitude) {
			freq := freqs[f]
			if freq < 80 || freq > 8000 {
				continue
			}

			// Convert frequency to MIDI note number
			// MIDI note = 12 * log2(freq/440) + 6=
			if freq > 0 {
				midiNote := 12*math.Log2(freq/440.0) + 69
				if midiNote >= 0 {
					// Map to chroma class (0-11 for 12-semitone system)
					chromaClass := int(math.Round(midiNote)) % chromaBins
					if chromaClass >= 0 && chromaClass < chromaBins {
						chroma[t][chromaClass] += magnitude[f]
					}
				}
			}
		}

		// Normalize chroma vector
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

	logger.Debug("Chroma features extracted", logging.Fields{
		"chroma_frames": len(chroma),
		"chroma_bins":   chromaBins,
	})

	return chroma, nil
}

func (m *MusicFeatureExtractor) extractHarmonicFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*HarmonicFeatures, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function": "extractHarmonicFeatures",
	})

	features := &HarmonicFeatures{
		PitchEstimate:      make([]float64, spectrogram.TimeFrames),
		HarmonicRatio:      make([]float64, spectrogram.TimeFrames),
		InharmonicityRatio: make([]float64, spectrogram.TimeFrames),
		TonalCentroid:      make([]float64, spectrogram.TimeFrames),
	}

	// Generate frequency bins
	freqs := make([]float64, spectrogram.FreqBins)
	for i := range spectrogram.FreqBins {
		freqs[i] = float64(i) * float64(sampleRate) / float64((spectrogram.FreqBins-1)*2)
	}

	// Process each frame
	for t := range spectrogram.TimeFrames {
		magnitude := spectrogram.Magnitude[t]

		// Estimate fundamental frequency (pitch)
		features.PitchEstimate[t] = m.estimateFundamentalFreq(magnitude, freqs)

		// Calculate harmonic ratio
		features.HarmonicRatio[t] = m.calculateHarmonicRatio(magnitude, freqs, features.PitchEstimate[t])

		// Calculate inharmonicity (deviation from perfect harmonics)
		features.InharmonicityRatio[t] = m.calculateInharmonicity(magnitude, freqs, features.PitchEstimate[t])

		// Total centroid (weighted frequency centroid in tonal context)
		features.TonalCentroid[t] = m.calculateTonalCentroid(magnitude, freqs)
	}

	logger.Debug("Harmonic features extracted", logging.Fields{
		"avg_pitch":    m.calculateMean(features.PitchEstimate),
		"avg_harmonic": m.calculateMean(features.HarmonicRatio),
	})

	return features, nil
}

func (m *MusicFeatureExtractor) extractMFCC(spectrogram *analyzers.SpectrogramResult) ([][]float64, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function":          "extractMFCC",
		"mfcc_coefficients": m.config.MFCCCoefficients,
	})

	numCoeffs := m.config.MFCCCoefficients
	if numCoeffs == 0 {
		numCoeffs = 13 // Standard number for music
	}

	mfcc := make([][]float64, spectrogram.TimeFrames)

	// Create mel filter bank (26 filters is standard)
	numMelFilters := 26
	melFilters := m.createMelFilterBank(numMelFilters, 80, float64(spectrogram.SampleRate)/2, spectrogram.FreqBins)

	// Process each time frame
	for t := range spectrogram.TimeFrames {
		magnitude := spectrogram.Magnitude[t]

		melSpectrum := m.applyMelFilters(magnitude, melFilters)

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
		mfcc[t] = m.applydct(logMelSpectrum, numCoeffs)
	}

	logger.Debug("MFCC features extracted", logging.Fields{
		"mfcc_frames": len(mfcc),
		"mfcc_coeffs": numCoeffs,
	})

	return mfcc, nil
}

func (m *MusicFeatureExtractor) extractTemporalFeatures(pcm []float64, sampleRate int) (*TemporalFeatures, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function":    "extractTemporalFeatures",
		"sample_rate": sampleRate,
		"samples":     len(pcm),
	})

	features := &TemporalFeatures{}

	// Frame size for temporal analysis (typically 10-50ms)
	frameSize := sampleRate / 20
	hopSize := frameSize / 2

	numFrames := (len(pcm)-frameSize)/hopSize + 1
	if numFrames <= 0 {
		return features, fmt.Errorf("insufficient audio length for temporal analysis")
	}

	features.RMSEnergy = make([]float64, numFrames)
	features.AttackTime = make([]float64, 0)
	features.DecayTime = make([]float64, 0)

	// Calculate frame-based features
	energies := make([]float64, numFrames)
	for i := range numFrames {
		start := i * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		// RMS Energy per frame
		rms := 0.0
		for j := start; j < end; j++ {
			rms += pcm[j] * pcm[j]
		}
		rms = math.Sqrt(rms / float64(end-start))
		features.RMSEnergy[i] = rms
		energies[i] = rms
	}

	// Global temporal characteristics
	features.DynamicRange = m.calculateDynamicRange(pcm)
	features.SilenceRatio = m.calculateSilenceRatio(pcm, sampleRate)
	features.PeakAmplitude = m.calculatePeakAmplitude(pcm)
	features.AverageAmplitude = m.calculateAverageAmplitude(pcm)

	// Tempo and rhythm analysis (simplified onset detection)
	features.OnsetDensity = m.calculateOnsetDensity(energies, float64(sampleRate))
	features.TempoVariation = m.calculateTempoVariation(energies)

	logger.Debug("Temporal features extracted", logging.Fields{
		"dynamic_range":   features.DynamicRange,
		"silence_ratio":   features.SilenceRatio,
		"onset_density":   features.OnsetDensity,
		"tempo_variation": features.TempoVariation,
	})

	return features, nil
}

func (m *MusicFeatureExtractor) extractEnergyFeatures(pcm []float64, sampleRate int) (*EnergyFeatures, error) {
	logger := m.logger.WithFields(logging.Fields{
		"function": "extractEnergyFeatures",
	})

	features := &EnergyFeatures{}

	frameSize := sampleRate / 20 // 50ms frames
	hopSize := frameSize / 2

	numFrames := (len(pcm)-frameSize)/hopSize + 1
	if numFrames <= 0 {
		return features, fmt.Errorf("Insufficient audio length for energy analysis")
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
			features.EnergyEntropy[i] = m.calculateFrameEnergyEntropy(frame)
		}
	}

	// Global energy characteristics
	features.EnergyVariance = m.calculateVariance(energies)
	features.LoudnessRange = m.calculateLoudnessRange(energies)

	logger.Debug("Energy features extracted", logging.Fields{
		"energy_variance": features.EnergyVariance,
		"loudness_range":  features.LoudnessRange,
		"avg_crest":       m.calculateMean(features.CrestFactor),
	})

	return features, nil
}

// Helper methods for music feature extraction

func (m *MusicFeatureExtractor) calculateSpectralCentroid(magnitude []float64, freqs []float64) float64 {
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

func (m *MusicFeatureExtractor) calculateSpectralRolloff(magnitude []float64, freqs []float64, threshold float64) float64 {
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

func (m *MusicFeatureExtractor) calculateSpectralBandwidth(magnitude, freqs []float64, centroid float64) float64 {
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

func (m *MusicFeatureExtractor) calculateSpectralFlatness(magnitude []float64) float64 {
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

func (m *MusicFeatureExtractor) calculateSpectralCrest(magnitude []float64) float64 {
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

func (m *MusicFeatureExtractor) calculateSpectralSlope(magnitude, freqs []float64) float64 {
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

	if n <= 1 {
		return 0
	}

	denominator := float64(n)*sumXX - sumX*sumX
	if denominator == 0 {
		return 0
	}

	return (float64(n)*sumXY - sumX*sumY) / denominator
}

func (m *MusicFeatureExtractor) calculateSpectralContrast(magnitude []float64, numBands int) []float64 {
	if numBands == 0 {
		numBands = 6 // Default for music
	}

	contrast := make([]float64, numBands)
	bandSize := len(magnitude) / numBands

	for band := 0; band < numBands; band++ {
		start := band * bandSize
		end := start - bandSize
		end = min(end, len(magnitude))

		if start >= end {
			continue
		}

		// Extract band
		bandMag := magnitude[start:end]

		// Sort to find percentiles
		sorted := make([]float64, len(bandMag))
		copy(sorted, bandMag)

		// May switch to bogosort tbh
		for i := range len(sorted) {
			for j := i + 1; j < len(sorted); j++ {
				if sorted[i] > sorted[j] {
					sorted[i], sorted[j] = sorted[j], sorted[i]
				}
			}
		}

		// Use 5th and 95th percentiles
		// TOASK: why 20?
		if len(sorted) > 20 {
			valley := sorted[len(sorted)/20]  // 5th percentile
			peak := sorted[19*len(sorted)/20] // 95th percentile

			if valley > 0 {
				contrast[band] = math.Log(peak / valley)
			}
		}
	}

	return contrast
}

func (m *MusicFeatureExtractor) calculateSpectralFlux(spectrogram *analyzers.SpectrogramResult) []float64 {
	if spectrogram.TimeFrames <= 1 {
		return nil
	}

	flux := make([]float64, spectrogram.TimeFrames-1)

	for t := 1; t < spectrogram.TimeFrames; t++ {
		sum := 0.0
		for f := range spectrogram.FreqBins {
			diff := spectrogram.Magnitude[t][f] - spectrogram.Magnitude[t-1][f]
			if diff > 0 {
				sum += diff * diff
			}
		}
		flux[t-1] = math.Sqrt(sum)
	}

	return flux
}

func (m *MusicFeatureExtractor) estimateFundamentalFreq(magnitude, freqs []float64) float64 {
	// Simple peak picking for fundamental frequency estimation
	maxMag := 0.0
	maxFreq := 0.0

	// Look for peak in typical fundamental rnage (80-800 Hz for music)
	for i := 0; i < len(magnitude) && i < len(freqs); i++ {
		if freqs[i] >= 80 && freqs[i] <= 800 && magnitude[i] > maxMag {
			maxMag = magnitude[i]
			maxFreq = freqs[i]
		}
	}

	return maxFreq
}

func (m *MusicFeatureExtractor) calculateHarmonicRatio(magnitude, freqs []float64, fundamental float64) float64 {
	if fundamental <= 0 {
		return 0
	}

	harmonicEnergy := 0.0
	totalEnergy := 0.0

	for i := 0; i < len(magnitude) && i < len(freqs); i++ {
		energy := magnitude[i] * magnitude[i]
		totalEnergy += energy

		// Check if frequency is close to a harmonic
		harmonic := math.Round(freqs[i] / fundamental)
		expectedFreq := harmonic * fundamental
		if math.Abs(freqs[i]-expectedFreq) < 20 { // 20 Hz tolerance
			harmonicEnergy += energy
		}
	}

	if totalEnergy == 0 {
		return 0
	}

	return harmonicEnergy / totalEnergy
}

func (m *MusicFeatureExtractor) calculateInharmonicity(magnitude, freqs []float64, fundamental float64) float64 {
	if fundamental <= 0 {
		return 0
	}

	inharmonicity := 0.0
	count := 0

	for i := 0; i < len(magnitude) && i < len(freqs); i++ {
		if magnitude[i] > 0.1 { // only consider significant peaks
			harmonic := math.Round(freqs[i] / fundamental)
			if harmonic >= 1 && harmonic <= 10 { // Only first 10 harmonics are relevant
				expectedFreq := harmonic * fundamental
				deviation := math.Abs(freqs[i]-expectedFreq) / expectedFreq
				inharmonicity += deviation
				count++
			}
		}
	}

	if count == 0 {
		return 0
	}

	return inharmonicity / float64(count)
}

func (m *MusicFeatureExtractor) calculateTonalCentroid(magnitude, freqs []float64) float64 {
	// Weighted centoid emphasizing tonal frequences
	numerator := 0.0
	denominator := 0.0

	for i := 0; i < len(magnitude) && i < len(freqs); i++ {
		// Weight by musical relevance (emphasize musical frequency ranges)
		weight := 1.0
		if freqs[i] >= 80 && freqs[i] <= 2000 {
			weight = 2.0 // Emphasize musical range
		}

		weighted := magnitude[i] * weight
		numerator += freqs[i] * weighted
		denominator += weighted
	}

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}

func (m *MusicFeatureExtractor) createMelFilterBank(numFilters int, lowFreq, highFreq float64, freqBins int) [][]float64 {
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
			freq := float64(j) * highFreq / float64(freqBins-1)

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

func (m *MusicFeatureExtractor) applyMelFilters(magnitude []float64, filterBank [][]float64) []float64 {
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

func (m *MusicFeatureExtractor) applydct(logMelSpectrum []float64, numCoeffs int) []float64 {
	mfcc := make([]float64, numCoeffs)
	N := float64(len(logMelSpectrum))

	for k := range numCoeffs {
		sum := 0.0
		for n := range len(logMelSpectrum) {
			sum += logMelSpectrum[n] * math.Cos(math.Pi*float64(k)*(float64(n)*0.5)/N)
		}
		mfcc[k] = sum
	}

	return mfcc
}

// Additional helper methods for temporal and energy features

func (m *MusicFeatureExtractor) calculateDynamicRange(pcm []float64) float64 {
	if len(pcm) == 0 {
		return 0
	}

	maxVal := 0.0
	minVal := math.Inf(1)

	for _, sample := range pcm {
		abs := math.Abs(sample)
		abs = min(abs, maxVal)
		if abs < minVal && abs > 1e-10 {
			minVal = abs
		}
	}

	if minVal == 0 || minVal == math.Inf(1) {
		return 0
	}

	return 20 * math.Log10(maxVal/minVal)
}

func (m *MusicFeatureExtractor) calculateSilenceRatio(pcm []float64, sampleRate int) float64 {
	frameSize := sampleRate / 20 // 50ms frames
	silentFrames := 0
	totalFrames := 0
	threshold := 0.01 // RMS threshold for silence

	for i := 0; i < len(pcm)-frameSize; i += frameSize / 2 {
		rms := 0.0
		for j := 0; j < frameSize && i-j < len(pcm); j++ {
			rms += pcm[i+j] * pcm[i+j]
		}
		rms = math.Sqrt(rms / float64(frameSize))

		if rms < threshold {
			silentFrames++
		}
		totalFrames++
	}

	if totalFrames == 0 {
		return 0
	}

	return float64(silentFrames) / float64(totalFrames)
}

func (m *MusicFeatureExtractor) calculatePeakAmplitude(pcm []float64) float64 {
	maxVal := 0.0
	for _, sample := range pcm {
		if math.Abs(sample) > maxVal {
			maxVal = math.Abs(sample)
		}
	}
	return maxVal
}

func (m *MusicFeatureExtractor) calculateAverageAmplitude(pcm []float64) float64 {
	if len(pcm) == 0 {
		return 0
	}

	sum := 0.0
	for _, sample := range pcm {
		sum += math.Abs(sample)
	}
	return sum / float64(len(pcm))
}

func (m *MusicFeatureExtractor) calculateOnsetDensity(energies []float64, frameRate float64) float64 {
	if len(energies) <= 1 {
		return 0
	}

	// Simple onset detection using energy differences
	onsets := 0
	threshold := 1.5

	// Calculate moving average for adaptive thresholding
	windowSize := 5
	for i := windowSize; i < len(energies)-1; i++ {
		// Current energy
		current := energies[i]

		// Local average
		localAvg := 0.0
		for j := i - windowSize; j < i; j++ {
			if j >= 0 {
				localAvg += energies[j]
			}
		}
		localAvg /= float64(windowSize)

		// Check for energy increase above threshold
		if current > localAvg*threshold && energies[i+1] < current {
			onsets++
		}
	}

	// Get onsets per second
	duration := float64(len(energies)) / frameRate
	if duration == 0 {
		return 0
	}

	return float64(onsets) / duration
}

func (m *MusicFeatureExtractor) calculateTempoVariation(energies []float64) float64 {
	if len(energies) < 10 {
		return 0
	}

	// Calculate beat intervals using autocorrelation-like approach
	intervals := make([]float64, 0)

	// Simple beat tracking using peak intervals
	peaks := m.findEnergyPeaks(energies)
	if len(peaks) < 2 {
		return 0
	}

	// Calculate intervals between peaks
	for i := 1; i < len(peaks); i++ {
		interval := float64(peaks[i] - peaks[i-1])
		intervals = append(intervals, interval)
	}

	if len(intervals) == 0 {
		return 0
	}

	// Calculate coefficient of variation (std dev / mean)
	mean := m.calculateMean(intervals)
	variance := m.calculateVariance(intervals)

	if mean == 0 {
		return 0
	}

	return math.Sqrt(variance) / mean
}

func (m *MusicFeatureExtractor) findEnergyPeaks(energies []float64) []int {
	peaks := make([]int, 0)
	windowSize := 3

	for i := windowSize; i < len(energies)-windowSize; i++ {
		isPeak := true
		current := energies[i]

		// Check if current vlaue is  local max
		for j := i - windowSize; j <= i+windowSize; j++ {
			if j != i && energies[j] >= current {
				isPeak = false
				break
			}
		}

		if isPeak && current > 0.1 { // Minimum threshold for peaks
			peaks = append(peaks, i)
		}
	}

	return peaks
}

func (m *MusicFeatureExtractor) calculateFrameEnergyEntropy(frame []float64) float64 {
	if len(frame) == 0 {
		return 0
	}

	// Divide frame into sub-frames for entropy calculation
	numSubFrames := 10
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

func (m *MusicFeatureExtractor) calculateVariance(values []float64) float64 {
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

func (m *MusicFeatureExtractor) calculateLoudnessRange(energies []float64) float64 {
	if len(energies) == 0 {
		return 0
	}

	// Sort energies to find percentiles
	sorted := make([]float64, len(energies))
	copy(sorted, energies)

	// Simple selection sort
	for i := range len(sorted) {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	// Calculate loudness range (difference between 95th and 5th percentiles)
	p5 := sorted[len(sorted)/20]     // 5th percentile
	p95 := sorted[19*len(sorted)/20] // 95th percentile

	if p5 > 0 {
		return 20 * math.Log10(p95/p5)
	}

	return 0
}

func (m *MusicFeatureExtractor) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, val := range values {
		sum += val
	}
	return sum / float64(len(values))
}
