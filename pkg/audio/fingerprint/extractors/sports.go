package extractors

import (
	"fmt"
	"math"
	"sort"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/logging"
)

// SportsFeatureExtractor extracts features optimized for sports content
type SportsFeatureExtractor struct {
	config *config.FeatureConfig
	logger logging.Logger
}

// NewSportsFeatureExtractor creates a sports-specific feature extractor
func NewSportsFeatureExtractor(featureConfig *config.FeatureConfig) *SportsFeatureExtractor {
	return &SportsFeatureExtractor{
		config: featureConfig,
		logger: logging.WithFields(logging.Fields{
			"component": "sports_feature_extractor",
		}),
	}
}

func (sp *SportsFeatureExtractor) GetName() string {
	return "SportsFeatureExtractor"
}

func (sp *SportsFeatureExtractor) GetContentType() config.ContentType {
	return config.ContentSports
}

func (sp *SportsFeatureExtractor) GetFeatureWeights() map[string]float64 {
	if sp.config.SimilarityWeights != nil {
		return sp.config.SimilarityWeights
	}

	// Default weights for sports content
	return map[string]float64{
		"energy":            0.35, // High weight for crowd noise/energy dynamics
		"spectral_contrast": 0.25, // Important for distinguishing elements
		"temporal":          0.20, // Dynamic changes and crowd reactions
		"mfcc":              0.15, // Commentary analysis
		"spectral":          0.05, // Basic characteristics
	}
}

func (sp *SportsFeatureExtractor) ExtractFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*ExtractedFeatures, error) {
	if spectrogram == nil {
		return nil, fmt.Errorf("spectrogram cannot be nil")
	}
	if len(pcm) == 0 {
		return nil, fmt.Errorf("PCM data cannot be empty")
	}
	if sampleRate <= 0 {
		return nil, fmt.Errorf("sample rate must be positive")
	}

	logger := sp.logger.WithFields(logging.Fields{
		"function":  "ExtractFeatures",
		"frames":    spectrogram.TimeFrames,
		"freq_bins": spectrogram.FreqBins,
	})

	logger.Debug("Extracting sports-specific features")

	features := &ExtractedFeatures{
		ExtractionMetadata: make(map[string]any),
	}

	// Extract energy features (highest priority for sports)
	energyFeatures, err := sp.extractSportsEnergyFeatures(pcm, sampleRate)
	if err != nil {
		logger.Error(err, "Failed to extract energy features for sports")
		return nil, err
	}
	features.EnergyFeatures = energyFeatures

	// Extract spectral contrast (crowd vs commentary)
	if sp.config.EnableSpectralContrast {
		spectralFeatures, err := sp.extractSportsSpectralFeatures(spectrogram, pcm)
		if err != nil {
			logger.Error(err, "Failed to extract spectral features")
		} else {
			features.SpectralFeatures = spectralFeatures
		}
	}

	// Extract temporal features (crowd reactions, commentary patterns)
	if sp.config.EnableTemporalFeatures {
		temporalFeatures, err := sp.extractSportsTemporalFeatures(pcm, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract temporal features")
		} else {
			features.TemporalFeatures = temporalFeatures
		}
	}

	// Extract MFCC for commentary analysis
	if sp.config.EnableMFCC {
		mfcc, err := sp.extractSportsMFCC(spectrogram)
		if err != nil {
			logger.Error(err, "Failed to extract MFCC")
		} else {
			features.MFCC = mfcc
		}
	}

	// Extract crowd analysis features
	crowdFeatures, err := sp.extractCrowdFeatures(spectrogram, pcm, sampleRate)
	if err != nil {
		logger.Error(err, "Failed to extract crowd features")
	} else {
		// Store crowd features in metadata since they're sports-specific
		features.ExtractionMetadata["crowd_features"] = crowdFeatures
	}

	// Add extraction metadata
	features.ExtractionMetadata["extractor_type"] = "sports"
	features.ExtractionMetadata["energy_focus"] = true
	features.ExtractionMetadata["crowd_analysis"] = true
	features.ExtractionMetadata["commentary_analysis"] = sp.config.EnableMFCC

	logger.Info("Sports feature extraction completed")
	return features, nil
}

// CrowdFeatures represents crowd-specific characteristics
type CrowdFeatures struct {
	CrowdNoiseLevel   []float64 `json:"crowd_noise_level"`
	ExcitementLevel   []float64 `json:"excitement_level"`
	CrowdDensity      float64   `json:"crowd_density"`
	ReactionIntensity []float64 `json:"reaction_intensity"`
	BackgroundNoise   float64   `json:"background_noise"`
	CrowdConsistency  float64   `json:"crowd_consistency"`
	PeakReactions     []float64 `json:"peak_reactions"`
	EnergyBursts      int       `json:"energy_bursts"`
}

// extractSportsEnergyFeatures extracts energy features optimized for sports dynamics
func (sp *SportsFeatureExtractor) extractSportsEnergyFeatures(pcm []float64, sampleRate int) (*EnergyFeatures, error) {
	logger := sp.logger.WithFields(logging.Fields{
		"function": "extractSportsEnergyFeatures",
	})

	features := &EnergyFeatures{}

	// Sports-optimized frame size (shorter for dynamic content)
	frameSize := sampleRate / 25 // 40ms frames (vs 50ms for music)
	hopSize := frameSize / 4     // 75% overlap for better temporal resolution

	numFrames := (len(pcm)-frameSize)/hopSize + 1
	if numFrames <= 0 {
		return features, fmt.Errorf("insufficient audio length for sports energy analysis")
	}

	features.ShortTimeEnergy = make([]float64, numFrames)
	features.EnergyEntropy = make([]float64, numFrames)
	features.CrestFactor = make([]float64, numFrames)

	energies := make([]float64, numFrames)
	instantaneousPeaks := make([]float64, numFrames)

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
		instantaneousPeaks[i] = maxVal

		// Crest factor (peak-to-RMS ratio) - important for crowd dynamics
		rms := math.Sqrt(energy)
		if rms > 0 {
			features.CrestFactor[i] = maxVal / rms
		}

		// Energy entropy (crowd vs commentary discrimination)
		features.EnergyEntropy[i] = sp.calculateSportsEnergyEntropy(frame)
	}

	// Global energy characteristics for sports
	features.EnergyVariance = sp.calculateSportsEnergyVariance(energies)
	features.LoudnessRange = sp.calculateSportsLoudnessRange(energies)

	// Sports-specific energy metrics
	sp.addSportsEnergyMetrics(features, energies, instantaneousPeaks)

	logger.Debug("Sports energy features extracted", logging.Fields{
		"energy_variance": features.EnergyVariance,
		"loudness_range":  features.LoudnessRange,
		"avg_crest":       sp.calculateMean(features.CrestFactor),
	})

	return features, nil
}

// extractSportsSpectralFeatures extracts spectral features for sports analysis
func (sp *SportsFeatureExtractor) extractSportsSpectralFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64) (*SpectralFeatures, error) {
	logger := sp.logger.WithFields(logging.Fields{
		"function": "extractSportsSpectralFeatures",
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
	var contrastBands int
	for t := 0; t < spectrogram.TimeFrames; t++ {
		magnitude := spectrogram.Magnitude[t]

		// Standard spectral features
		features.SpectralCentroid[t] = sp.calculateSpectralCentroid(magnitude, freqs)
		features.SpectralRolloff[t] = sp.calculateSpectralRolloff(magnitude, freqs, 0.85)
		features.SpectralBandwidth[t] = sp.calculateSpectralBandwidth(magnitude, freqs, features.SpectralCentroid[t])
		features.SpectralFlatness[t] = sp.calculateSpectralFlatness(magnitude)
		features.SpectralCrest[t] = sp.calculateSpectralCrest(magnitude)
		features.SpectralSlope[t] = sp.calculateSpectralSlope(magnitude, freqs)

		// Sports-specific spectral contrast (more bands for better discrimination)
		contrastBands = sp.config.ContrastBands
		if contrastBands == 0 {
			contrastBands = 8 // More bands for sports (vs 6 for music)
		}
		features.SpectralContrast[t] = sp.calculateSportsSpectralContrast(magnitude, contrastBands)

		// Calculate zero crossing rate
		start := t * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		if start < len(pcm) {
			pcmFrame := pcm[start:end]
			features.ZeroCrossingRate[t] = calculateZeroCrossingRate(pcmFrame)
		}
	}

	// Calculate spectral flux for event detection
	features.SpectralFlux = sp.calculateSportsSpectralFlux(spectrogram)

	logger.Debug("Sports spectral features extracted", logging.Fields{
		"avg_centroid":   sp.calculateMean(features.SpectralCentroid),
		"avg_rolloff":    sp.calculateMean(features.SpectralRolloff),
		"contrast_bands": contrastBands,
	})

	return features, nil
}

// extractSportsTemporalFeatures extracts temporal features for sports dynamics
func (sp *SportsFeatureExtractor) extractSportsTemporalFeatures(pcm []float64, sampleRate int) (*TemporalFeatures, error) {
	logger := sp.logger.WithFields(logging.Fields{
		"function":    "extractSportsTemporalFeatures",
		"sample_rate": sampleRate,
	})

	features := &TemporalFeatures{}

	// Sports-optimized frame size (shorter for dynamic events)
	frameSize := sampleRate / 25 // 40ms frames
	hopSize := frameSize / 4     // High overlap for event detection

	numFrames := (len(pcm)-frameSize)/hopSize + 1
	if numFrames <= 0 {
		return features, fmt.Errorf("insufficient audio length for sports temporal analysis")
	}

	features.RMSEnergy = make([]float64, numFrames)
	features.AttackTime = make([]float64, 0)
	features.DecayTime = make([]float64, 0)

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

	// Sports-specific temporal characteristics
	features.DynamicRange = sp.calculateSportsDynamicRange(pcm)
	features.SilenceRatio = sp.calculateSportsSilenceRatio(energies)
	features.PeakAmplitude = sp.calculatePeakAmplitude(pcm)
	features.AverageAmplitude = sp.calculateAverageAmplitude(pcm)

	// Sports event detection
	features.OnsetDensity = sp.calculateSportsEventDensity(energies, float64(sampleRate)/float64(hopSize))
	features.TempoVariation = sp.calculateCrowdReactionVariation(energies)

	// Detect attack and decay times for crowd reactions
	attackDecayTimes := sp.extractCrowdReactionTimes(energies, float64(hopSize)/float64(sampleRate))
	features.AttackTime = attackDecayTimes.AttackTimes
	features.DecayTime = attackDecayTimes.DecayTimes

	logger.Debug("Sports temporal features extracted", logging.Fields{
		"dynamic_range":      features.DynamicRange,
		"silence_ratio":      features.SilenceRatio,
		"event_density":      features.OnsetDensity,
		"reaction_variation": features.TempoVariation,
		"reaction_events":    len(features.AttackTime),
	})

	return features, nil
}

// extractSportsMFCC extracts MFCC for commentary analysis in sports
func (sp *SportsFeatureExtractor) extractSportsMFCC(spectrogram *analyzers.SpectrogramResult) ([][]float64, error) {
	logger := sp.logger.WithFields(logging.Fields{
		"function":          "extractSportsMFCC",
		"mfcc_coefficients": sp.config.MFCCCoefficients,
	})

	numCoeffs := sp.config.MFCCCoefficients
	if numCoeffs == 0 {
		numCoeffs = 13 // Standard number
	}

	mfcc := make([][]float64, spectrogram.TimeFrames)

	// Create mel filter bank optimized for mixed content (broader range than speech)
	numMelFilters := 26
	lowFreq := 200.0   // Lower than speech to capture crowd noise
	highFreq := 6000.0 // Higher than speech for stadium acoustics
	melFilters := sp.createSportsMelFilterBank(numMelFilters, lowFreq, highFreq, spectrogram.FreqBins, spectrogram.SampleRate)

	// Process each time frame
	for t := 0; t < spectrogram.TimeFrames; t++ {
		magnitude := spectrogram.Magnitude[t]

		// Apply mel filter bank
		melSpectrum := sp.applyMelFilters(magnitude, melFilters)

		// Take logarithm
		logMelSpectrum := make([]float64, len(melSpectrum))
		for i, val := range melSpectrum {
			if val > 1e-10 {
				logMelSpectrum[i] = math.Log(val)
			} else {
				logMelSpectrum[i] = math.Log(1e-10)
			}
		}

		// Apply DCT
		mfcc[t] = sp.applyDCT(logMelSpectrum, numCoeffs)
	}

	logger.Debug("Sports MFCC features extracted", logging.Fields{
		"mfcc_frames": len(mfcc),
		"mfcc_coeffs": numCoeffs,
		"freq_range":  fmt.Sprintf("%.0f-%.0fHz", lowFreq, highFreq),
	})

	return mfcc, nil
}

// extractCrowdFeatures extracts crowd-specific features for sports analysis
func (sp *SportsFeatureExtractor) extractCrowdFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*CrowdFeatures, error) {
	logger := sp.logger.WithFields(logging.Fields{
		"function": "extractCrowdFeatures",
	})

	features := &CrowdFeatures{
		CrowdNoiseLevel:   make([]float64, spectrogram.TimeFrames),
		ExcitementLevel:   make([]float64, spectrogram.TimeFrames),
		ReactionIntensity: make([]float64, spectrogram.TimeFrames),
		PeakReactions:     make([]float64, 0),
	}

	// Analyze crowd characteristics in frequency domain
	for t := 0; t < spectrogram.TimeFrames; t++ {
		magnitude := spectrogram.Magnitude[t]

		// Calculate crowd noise level (energy in crowd frequency range)
		features.CrowdNoiseLevel[t] = sp.calculateCrowdNoiseLevel(magnitude, spectrogram.SampleRate)

		// Calculate excitement level (high-frequency energy + spectral spread)
		features.ExcitementLevel[t] = sp.calculateExcitementLevel(magnitude, spectrogram.SampleRate)

		// Calculate reaction intensity (sudden energy changes)
		if t > 0 {
			features.ReactionIntensity[t] = sp.calculateReactionIntensity(
				spectrogram.Magnitude[t-1], magnitude)
		}
	}

	// Global crowd characteristics
	features.CrowdDensity = sp.calculateCrowdDensity(features.CrowdNoiseLevel)
	features.BackgroundNoise = sp.calculateBackgroundNoise(features.CrowdNoiseLevel)
	features.CrowdConsistency = sp.calculateCrowdConsistency(features.CrowdNoiseLevel)
	features.PeakReactions = sp.extractPeakReactions(features.ReactionIntensity)
	features.EnergyBursts = sp.countEnergyBursts(features.ExcitementLevel)

	logger.Debug("Crowd features extracted", logging.Fields{
		"crowd_density":    features.CrowdDensity,
		"background_noise": features.BackgroundNoise,
		"peak_reactions":   len(features.PeakReactions),
		"energy_bursts":    features.EnergyBursts,
	})

	return features, nil
}

// Sports-specific helper methods

type AttackDecayTimes struct {
	AttackTimes []float64
	DecayTimes  []float64
}

func (sp *SportsFeatureExtractor) addSportsEnergyMetrics(features *EnergyFeatures, energies, peaks []float64) {
	// Add sports-specific energy metrics to metadata
	// These could be stored in ExtractionMetadata if needed

	// Energy burst detection
	burstThreshold := sp.calculateMean(energies) + 2*math.Sqrt(sp.calculateVariance(energies))
	burstCount := 0
	for _, energy := range energies {
		if energy > burstThreshold {
			burstCount++
		}
	}

	// Peak consistency (how consistent are the peaks)
	peakVariance := sp.calculateVariance(peaks)

	// Could add these to features metadata if needed
	_ = burstCount
	_ = peakVariance
}

func (sp *SportsFeatureExtractor) calculateSportsEnergyVariance(energies []float64) float64 {
	// Sports-specific energy variance calculation
	if len(energies) <= 1 {
		return 0
	}

	// Use rolling variance to capture dynamics
	windowSize := 20 // Approximately 1 second at 25fps
	windowSize = min(windowSize, len(energies))

	maxVariance := 0.0
	for i := 0; i <= len(energies)-windowSize; i++ {
		window := energies[i : i+windowSize]
		variance := sp.calculateVariance(window)
		if variance > maxVariance {
			maxVariance = variance
		}
	}

	return maxVariance
}

func (sp *SportsFeatureExtractor) calculateSportsLoudnessRange(energies []float64) float64 {
	// Sports-specific loudness range (wider percentiles due to dynamics)
	if len(energies) == 0 {
		return 0
	}

	sorted := make([]float64, len(energies))
	copy(sorted, energies)
	sort.Float64s(sorted)

	// Use 2nd and 98th percentiles for sports (wider range)
	p2 := sorted[len(sorted)/50]     // 2nd percentile
	p98 := sorted[49*len(sorted)/50] // 98th percentile

	if p2 > 0 {
		return 20 * math.Log10(p98/p2)
	}

	return 0
}

func (sp *SportsFeatureExtractor) calculateSportsEnergyEntropy(frame []float64) float64 {
	// Sports-specific energy entropy calculation
	if len(frame) == 0 {
		return 0
	}

	// Use more sub-frames for sports to capture rapid changes
	numSubFrames := 12
	subFrameSize := len(frame) / numSubFrames

	if subFrameSize == 0 {
		return 0
	}

	subFrameEnergies := make([]float64, numSubFrames)
	totalEnergy := 0.0

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

	entropy := 0.0
	for _, energy := range subFrameEnergies {
		if energy > 0 {
			prob := energy / totalEnergy
			entropy -= prob * math.Log2(prob)
		}
	}

	return entropy
}

func (sp *SportsFeatureExtractor) calculateSportsSpectralContrast(magnitude []float64, numBands int) []float64 {
	// Sports-specific spectral contrast with emphasis on mid frequencies
	contrast := make([]float64, numBands)
	bandSize := len(magnitude) / numBands

	for band := range numBands {
		start := band * bandSize
		end := start + bandSize
		end = min(end, len(magnitude))

		if start >= end {
			continue
		}

		bandMag := magnitude[start:end]

		// Sort to find percentiles
		sorted := make([]float64, len(bandMag))
		copy(sorted, bandMag)
		sort.Float64s(sorted)

		// Use different percentiles for different bands
		var valley, peak float64
		if band < numBands/2 {
			// Lower frequency bands (crowd noise) - use wider percentiles
			valley = sorted[len(sorted)/30]  // 3rd percentile
			peak = sorted[29*len(sorted)/30] // 97th percentile
		} else {
			// Higher frequency bands (commentary) - use narrower percentiles
			valley = sorted[len(sorted)/10] // 10th percentile
			peak = sorted[9*len(sorted)/10] // 90th percentile
		}

		if valley > 0 {
			contrast[band] = math.Log(peak / valley)
		}
	}

	return contrast
}

func (sp *SportsFeatureExtractor) calculateSportsSpectralFlux(spectrogram *analyzers.SpectrogramResult) []float64 {
	// Sports-specific spectral flux for event detection
	if spectrogram.TimeFrames < 2 {
		return nil
	}

	flux := make([]float64, spectrogram.TimeFrames-1)

	for t := 1; t < spectrogram.TimeFrames; t++ {
		sum := 0.0
		for f := 0; f < spectrogram.FreqBins; f++ {
			diff := spectrogram.Magnitude[t][f] - spectrogram.Magnitude[t-1][f]
			// For sports, consider both increases and decreases
			sum += diff * diff
		}
		flux[t-1] = math.Sqrt(sum)
	}

	return flux
}

func (sp *SportsFeatureExtractor) calculateSportsDynamicRange(pcm []float64) float64 {
	// Sports-specific dynamic range calculation
	if len(pcm) == 0 {
		return 0
	}

	// Use percentiles approach for robust measurement
	sorted := make([]float64, len(pcm))
	for i, sample := range pcm {
		sorted[i] = math.Abs(sample)
	}
	sort.Float64s(sorted)

	// Use 1st and 99th percentiles for sports (very wide range)
	p1 := sorted[len(sorted)/100]     // 1st percentile
	p99 := sorted[99*len(sorted)/100] // 99th percentile

	if p1 > 1e-10 {
		return 20 * math.Log10(p99/p1)
	}

	return 0
}

func (sp *SportsFeatureExtractor) calculateSportsSilenceRatio(energies []float64) float64 {
	// Sports-specific silence ratio (very little actual silence expected)
	if len(energies) == 0 {
		return 0
	}

	// Use a much lower threshold for sports (background crowd noise)
	meanEnergy := sp.calculateMean(energies)
	threshold := meanEnergy * 0.05 // 5% of mean energy

	silentFrames := 0
	for _, energy := range energies {
		if energy < threshold {
			silentFrames++
		}
	}

	return float64(silentFrames) / float64(len(energies))
}

func (sp *SportsFeatureExtractor) calculateSportsEventDensity(energies []float64, frameRate float64) float64 {
	// Calculate sports event density (crowd reactions, scoring events, etc.)
	if len(energies) < 5 {
		return 0
	}

	events := 0
	threshold := 2.0 // Higher threshold for sports events

	// Use adaptive thresholding with larger window
	windowSize := 10 // Larger window for sports events
	for i := windowSize; i < len(energies)-1; i++ {
		current := energies[i]

		// Local statistics
		localMean := 0.0
		localMax := 0.0
		for j := i - windowSize; j < i; j++ {
			if j >= 0 {
				localMean += energies[j]
				if energies[j] > localMax {
					localMax = energies[j]
				}
			}
		}
		localMean /= float64(windowSize)

		// Check for significant energy increase (crowd reaction)
		if current > localMean*threshold && current > localMax*0.8 {
			events++
		}
	}

	// Return events per second
	duration := float64(len(energies)) / frameRate
	if duration == 0 {
		return 0
	}

	return float64(events) / duration
}

func (sp *SportsFeatureExtractor) calculateCrowdReactionVariation(energies []float64) float64 {
	// Calculate crowd reaction variation (different from tempo variation)
	if len(energies) < 20 {
		return 0
	}

	// Look for crowd reaction patterns
	reactionPeaks := sp.findCrowdReactionPeaks(energies)
	if len(reactionPeaks) < 2 {
		return 0
	}

	// Calculate intervals between reactions
	intervals := make([]float64, 0)
	for i := 1; i < len(reactionPeaks); i++ {
		interval := float64(reactionPeaks[i] - reactionPeaks[i-1])
		intervals = append(intervals, interval)
	}

	if len(intervals) == 0 {
		return 0
	}

	// Calculate coefficient of variation
	mean := sp.calculateMean(intervals)
	variance := sp.calculateVariance(intervals)

	if mean == 0 {
		return 0
	}

	return math.Sqrt(variance) / mean
}

func (sp *SportsFeatureExtractor) findCrowdReactionPeaks(energies []float64) []int {
	// Find crowd reaction peaks (different from energy peaks)
	peaks := make([]int, 0)
	windowSize := 5

	// Calculate adaptive threshold
	meanEnergy := sp.calculateMean(energies)
	stdEnergy := math.Sqrt(sp.calculateVariance(energies))
	threshold := meanEnergy + 1.5*stdEnergy

	for i := windowSize; i < len(energies)-windowSize; i++ {
		current := energies[i]

		// Check if it's a local maximum above threshold
		isPeak := true
		for j := i - windowSize; j <= i+windowSize; j++ {
			if j != i && energies[j] >= current {
				isPeak = false
				break
			}
		}

		if isPeak && current > threshold {
			peaks = append(peaks, i)
		}
	}

	return peaks
}

func (sp *SportsFeatureExtractor) extractCrowdReactionTimes(energies []float64, frameTimeSeconds float64) *AttackDecayTimes {
	// Extract attack and decay times for crowd reactions
	result := &AttackDecayTimes{
		AttackTimes: make([]float64, 0),
		DecayTimes:  make([]float64, 0),
	}

	peaks := sp.findCrowdReactionPeaks(energies)

	for _, peakIdx := range peaks {
		// Find attack time (rise to peak)
		attackStart := peakIdx
		peakEnergy := energies[peakIdx]
		threshold := peakEnergy * 0.1 // 10% of peak energy

		for i := peakIdx - 1; i >= 0; i-- {
			if energies[i] < threshold {
				attackStart = i
				break
			}
		}

		attackTime := float64(peakIdx-attackStart) * frameTimeSeconds
		result.AttackTimes = append(result.AttackTimes, attackTime)

		// Find decay time (fall from peak)
		decayEnd := peakIdx
		for i := peakIdx + 1; i < len(energies); i++ {
			if energies[i] < threshold {
				decayEnd = i
				break
			}
		}

		decayTime := float64(decayEnd-peakIdx) * frameTimeSeconds
		result.DecayTimes = append(result.DecayTimes, decayTime)
	}

	return result
}

// Crowd-specific analysis methods

func (sp *SportsFeatureExtractor) calculateCrowdNoiseLevel(magnitude []float64, sampleRate int) float64 {
	// Calculate crowd noise level in typical crowd frequency range (100-1000Hz)
	crowdEnergy := 0.0
	totalEnergy := 0.0

	for i, mag := range magnitude {
		freq := float64(i) * float64(sampleRate) / float64((len(magnitude)-1)*2)
		energy := mag * mag
		totalEnergy += energy

		if freq >= 100 && freq <= 1000 {
			crowdEnergy += energy
		}
	}

	if totalEnergy == 0 {
		return 0
	}

	return crowdEnergy / totalEnergy
}

func (sp *SportsFeatureExtractor) calculateExcitementLevel(magnitude []float64, sampleRate int) float64 {
	// Calculate excitement level based on high-frequency energy and spectral spread
	highFreqEnergy := 0.0
	totalEnergy := 0.0

	for i, mag := range magnitude {
		freq := float64(i) * float64(sampleRate) / float64((len(magnitude)-1)*2)
		energy := mag * mag
		totalEnergy += energy

		// High frequency range for excitement (1kHz-4kHz)
		if freq >= 1000 && freq <= 4000 {
			highFreqEnergy += energy
		}
	}

	if totalEnergy == 0 {
		return 0
	}

	// Combine high-frequency ratio with spectral spread
	highFreqRatio := highFreqEnergy / totalEnergy
	spectralSpread := sp.calculateSpectralSpread(magnitude, sampleRate)

	return highFreqRatio * (1 + spectralSpread/1000) // Normalize spread
}

func (sp *SportsFeatureExtractor) calculateSpectralSpread(magnitude []float64, sampleRate int) float64 {
	// Calculate spectral spread (bandwidth)
	centroid := 0.0
	totalMag := 0.0

	// Calculate centroid first
	for i, mag := range magnitude {
		freq := float64(i) * float64(sampleRate) / float64((len(magnitude)-1)*2)
		centroid += freq * mag
		totalMag += mag
	}

	if totalMag == 0 {
		return 0
	}
	centroid /= totalMag

	// Calculate spread around centroid
	spread := 0.0
	for i, mag := range magnitude {
		freq := float64(i) * float64(sampleRate) / float64((len(magnitude)-1)*2)
		diff := freq - centroid
		spread += diff * diff * mag
	}

	return math.Sqrt(spread / totalMag)
}

func (sp *SportsFeatureExtractor) calculateReactionIntensity(prevMagnitude, currMagnitude []float64) float64 {
	// Calculate reaction intensity as sudden spectral change
	if len(prevMagnitude) != len(currMagnitude) {
		return 0
	}

	totalChange := 0.0
	for i := range len(currMagnitude) {
		diff := currMagnitude[i] - prevMagnitude[i]
		if diff > 0 { // Only positive changes (energy increases)
			totalChange += diff * diff
		}
	}

	return math.Sqrt(totalChange)
}

func (sp *SportsFeatureExtractor) calculateCrowdDensity(crowdNoiseLevels []float64) float64 {
	// Calculate crowd density based on consistency of crowd noise
	if len(crowdNoiseLevels) == 0 {
		return 0
	}

	mean := sp.calculateMean(crowdNoiseLevels)
	variance := sp.calculateVariance(crowdNoiseLevels)

	// Higher density = higher mean with lower variance
	if variance == 0 {
		return mean
	}

	return mean / (1 + math.Sqrt(variance))
}

func (sp *SportsFeatureExtractor) calculateBackgroundNoise(crowdNoiseLevels []float64) float64 {
	// Calculate background noise level (minimum sustained crowd noise)
	if len(crowdNoiseLevels) == 0 {
		return 0
	}

	// Use 10th percentile as background noise estimate
	sorted := make([]float64, len(crowdNoiseLevels))
	copy(sorted, crowdNoiseLevels)
	sort.Float64s(sorted)

	return sorted[len(sorted)/10]
}

func (sp *SportsFeatureExtractor) calculateCrowdConsistency(crowdNoiseLevels []float64) float64 {
	// Calculate crowd consistency (how stable the crowd noise is)
	if len(crowdNoiseLevels) <= 1 {
		return 0
	}

	mean := sp.calculateMean(crowdNoiseLevels)
	variance := sp.calculateVariance(crowdNoiseLevels)

	if mean == 0 {
		return 0
	}

	// Consistency = 1 / coefficient_of_variation
	cv := math.Sqrt(variance) / mean
	return 1 / (1 + cv)
}

func (sp *SportsFeatureExtractor) extractPeakReactions(reactionIntensity []float64) []float64 {
	// Extract peak reaction intensities
	if len(reactionIntensity) == 0 {
		return nil
	}

	peaks := make([]float64, 0)
	threshold := sp.calculateMean(reactionIntensity) + math.Sqrt(sp.calculateVariance(reactionIntensity))

	for i := 1; i < len(reactionIntensity)-1; i++ {
		current := reactionIntensity[i]
		if current > threshold &&
			current > reactionIntensity[i-1] &&
			current > reactionIntensity[i+1] {
			peaks = append(peaks, current)
		}
	}

	return peaks
}

func (sp *SportsFeatureExtractor) countEnergyBursts(excitementLevels []float64) int {
	// Count energy bursts (sustained high excitement periods)
	if len(excitementLevels) == 0 {
		return 0
	}

	threshold := sp.calculateMean(excitementLevels) + 0.5*math.Sqrt(sp.calculateVariance(excitementLevels))
	minDuration := 5 // Minimum frames for a burst

	bursts := 0
	currentBurstLength := 0

	for _, level := range excitementLevels {
		if level > threshold {
			currentBurstLength++
		} else {
			if currentBurstLength >= minDuration {
				bursts++
			}
			currentBurstLength = 0
		}
	}

	// Check final burst
	if currentBurstLength >= minDuration {
		bursts++
	}

	return bursts
}

// Sports-specific mel filter bank creation
func (sp *SportsFeatureExtractor) createSportsMelFilterBank(numFilters int, lowFreq, highFreq float64, freqBins, sampleRate int) [][]float64 {
	// Create mel filter bank optimized for sports (broader frequency range)
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
	for i := range numFilters {
		filter := make([]float64, freqBins)

		leftFreq := freqPoints[i]
		centerFreq := freqPoints[i+1]
		rightFreq := freqPoints[i+2]

		for j := range freqBins {
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

func (sp *SportsFeatureExtractor) applyMelFilters(magnitude []float64, filterBank [][]float64) []float64 {
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

func (sp *SportsFeatureExtractor) applyDCT(logMelSpectrum []float64, numCoeffs int) []float64 {
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

func (sp *SportsFeatureExtractor) calculateSpectralCentroid(magnitude, freqs []float64) float64 {
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

func (sp *SportsFeatureExtractor) calculateSpectralRolloff(magnitude, freqs []float64, threshold float64) float64 {
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

func (sp *SportsFeatureExtractor) calculateSpectralBandwidth(magnitude, freqs []float64, centroid float64) float64 {
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

func (sp *SportsFeatureExtractor) calculateSpectralFlatness(magnitude []float64) float64 {
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

func (sp *SportsFeatureExtractor) calculateSpectralCrest(magnitude []float64) float64 {
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

func (sp *SportsFeatureExtractor) calculateSpectralSlope(magnitude, freqs []float64) float64 {
	if len(magnitude) != len(freqs) || len(magnitude) < 2 {
		return 0
	}

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

func (sp *SportsFeatureExtractor) calculatePeakAmplitude(pcm []float64) float64 {
	maxVal := 0.0
	for _, sample := range pcm {
		if math.Abs(sample) > maxVal {
			maxVal = math.Abs(sample)
		}
	}
	return maxVal
}

func (sp *SportsFeatureExtractor) calculateAverageAmplitude(pcm []float64) float64 {
	if len(pcm) == 0 {
		return 0
	}

	sum := 0.0
	for _, sample := range pcm {
		sum += math.Abs(sample)
	}
	return sum / float64(len(pcm))
}

func (sp *SportsFeatureExtractor) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, val := range values {
		sum += val
	}
	return sum / float64(len(values))
}

func (sp *SportsFeatureExtractor) calculateVariance(values []float64) float64 {
	if len(values) <= 1 {
		return 0
	}

	mean := sp.calculateMean(values)
	variance := 0.0

	for _, val := range values {
		diff := val - mean
		variance += diff * diff
	}

	return variance / float64(len(values))
}
