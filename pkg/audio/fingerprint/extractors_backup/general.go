package extractors

import (
	"fmt"
	"math"
	"sort"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
)

// GeneralFeatureExtractor provides balanced feature extraction for unknown content
type GeneralFeatureExtractor struct {
	config *config.FeatureConfig
	logger logging.Logger
}

// NewGeneralFeatureExtractor creates a general-purpose feature extractor
func NewGeneralFeatureExtractor(featureConfig *config.FeatureConfig) *GeneralFeatureExtractor {
	return &GeneralFeatureExtractor{
		config: featureConfig,
		logger: logging.WithFields(logging.Fields{
			"component": "general_feature_extractor",
		}),
	}
}

func (g *GeneralFeatureExtractor) GetName() string {
	return "GeneralFeatureExtractor"
}

func (g *GeneralFeatureExtractor) GetContentType() config.ContentType {
	return config.ContentUnknown
}

func (g *GeneralFeatureExtractor) GetFeatureWeights() map[string]float64 {
	if g.config.SimilarityWeights != nil {
		return g.config.SimilarityWeights
	}

	// Balanced weights for unknown content
	return map[string]float64{
		"spectral": 0.25,
		"mfcc":     0.25,
		"temporal": 0.20,
		"energy":   0.15,
		"chroma":   0.15,
	}
}

func (g *GeneralFeatureExtractor) ExtractFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*ExtractedFeatures, error) {
	if spectrogram == nil {
		return nil, fmt.Errorf("spectrogram cannot be nil")
	}
	if len(pcm) == 0 {
		return nil, fmt.Errorf("PCM data cannot be empty")
	}
	if sampleRate <= 0 {
		return nil, fmt.Errorf("sample rate must be positive")
	}

	logger := g.logger.WithFields(logging.Fields{
		"function": "ExtractFeatures",
		"frames":   spectrogram.TimeFrames,
	})

	logger.Debug("Extracting general features with balanced approach")

	features := &ExtractedFeatures{
		ExtractionMetadata: make(map[string]any),
	}

	// Extract spectral features (always enabled)
	spectralFeatures, err := g.extractSpectralFeatures(spectrogram, pcm)
	if err != nil {
		logger.Error(err, "Failed to extract spectral features")
		return nil, err
	}
	features.SpectralFeatures = spectralFeatures

	// Extract MFCC (if enabled)
	if g.config.EnableMFCC {
		mfcc, err := g.extractMFCC(spectrogram, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract MFCC")
		} else {
			features.MFCC = mfcc
		}
	}

	// Extract temporal features (if enabled)
	if g.config.EnableTemporalFeatures {
		temporalFeatures, err := g.extractTemporalFeatures(pcm, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract temporal features")
		} else {
			features.TemporalFeatures = temporalFeatures
		}
	}

	// Extract energy features
	energyFeatures, err := g.extractEnergyFeatures(pcm, sampleRate)
	if err != nil {
		logger.Error(err, "Failed to extract energy features")
	} else {
		features.EnergyFeatures = energyFeatures
	}

	// Extract chroma features (if enabled)
	if g.config.EnableChroma {
		chromaFeatures, err := g.extractChromaFeatures(spectrogram, sampleRate)
		if err != nil {
			logger.Error(err, "Failed to extract chroma features")
		} else {
			features.ChromaFeatures = chromaFeatures
		}
	}

	// Add metadata
	features.ExtractionMetadata["extractor_type"] = "general"
	features.ExtractionMetadata["balanced_approach"] = true
	features.ExtractionMetadata["universal_features"] = true

	logger.Info("General feature extraction completed")
	return features, nil
}

// extractSpectralFeatures extracts balanced spectral features
func (g *GeneralFeatureExtractor) extractSpectralFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64) (*SpectralFeatures, error) {
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

	freqs := make([]float64, spectrogram.FreqBins)
	for i := 0; i < spectrogram.FreqBins; i++ {
		freqs[i] = float64(i) * float64(spectrogram.SampleRate) / float64((spectrogram.FreqBins-1)*2)
	}

	hopSize := spectrogram.HopSize
	frameSize := hopSize * 2

	for t := 0; t < spectrogram.TimeFrames; t++ {
		magnitude := spectrogram.Magnitude[t]

		features.SpectralCentroid[t] = g.calculateSpectralCentroid(magnitude, freqs)
		features.SpectralRolloff[t] = g.calculateSpectralRolloff(magnitude, freqs, 0.85)
		features.SpectralBandwidth[t] = g.calculateSpectralBandwidth(magnitude, freqs, features.SpectralCentroid[t])
		features.SpectralFlatness[t] = g.calculateSpectralFlatness(magnitude)
		features.SpectralCrest[t] = g.calculateSpectralCrest(magnitude)
		features.SpectralSlope[t] = g.calculateSpectralSlope(magnitude, freqs)

		// Balanced spectral contrast
		contrastBands := g.config.ContrastBands
		if contrastBands == 0 {
			contrastBands = 6
		}
		features.SpectralContrast[t] = g.calculateSpectralContrast(magnitude, contrastBands)

		// Calculate zero crossing rate
		start := t * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		if start < len(pcm) {
			pcmFrame := pcm[start:end]
			features.ZeroCrossingRate[t] = calculateZeroCrossingRate(pcmFrame)
		}
	}

	features.SpectralFlux = g.calculateSpectralFlux(spectrogram)
	return features, nil
}

// extractMFCC extracts MFCC with balanced frequency range
func (g *GeneralFeatureExtractor) extractMFCC(spectrogram *analyzers.SpectrogramResult, sampleRate int) ([][]float64, error) {
	numCoeffs := g.config.MFCCCoefficients
	if numCoeffs == 0 {
		numCoeffs = 13
	}

	mfcc := make([][]float64, spectrogram.TimeFrames)
	numMelFilters := 26
	lowFreq := 150.0
	highFreq := 6000.0
	melFilters := g.createMelFilterBank(numMelFilters, lowFreq, highFreq, spectrogram.FreqBins, sampleRate)

	for t := 0; t < spectrogram.TimeFrames; t++ {
		magnitude := spectrogram.Magnitude[t]
		melSpectrum := g.applyMelFilters(magnitude, melFilters)

		logMelSpectrum := make([]float64, len(melSpectrum))
		for i, val := range melSpectrum {
			if val > 1e-10 {
				logMelSpectrum[i] = math.Log(val)
			} else {
				logMelSpectrum[i] = math.Log(1e-10)
			}
		}

		mfcc[t] = g.applyDCT(logMelSpectrum, numCoeffs)
	}

	return mfcc, nil
}

// extractTemporalFeatures extracts balanced temporal features
func (g *GeneralFeatureExtractor) extractTemporalFeatures(pcm []float64, sampleRate int) (*TemporalFeatures, error) {
	features := &TemporalFeatures{}
	frameSize := sampleRate * 40 / 1000 // 40ms frames
	hopSize := frameSize / 2

	numFrames := (len(pcm)-frameSize)/hopSize + 1
	if numFrames <= 0 {
		return features, fmt.Errorf("insufficient audio length")
	}

	features.RMSEnergy = make([]float64, numFrames)
	features.AttackTime = make([]float64, 0)
	features.DecayTime = make([]float64, 0)

	energies := make([]float64, numFrames)
	for i := range numFrames {
		start := i * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		frame := pcm[start:end]
		rms := 0.0
		for _, sample := range frame {
			rms += sample * sample
		}
		rms = math.Sqrt(rms / float64(len(frame)))
		features.RMSEnergy[i] = rms
		energies[i] = rms
	}

	features.DynamicRange = g.calculateDynamicRange(pcm)
	features.SilenceRatio = g.calculateSilenceRatio(energies)
	features.PeakAmplitude = g.calculatePeakAmplitude(pcm)
	features.AverageAmplitude = g.calculateAverageAmplitude(pcm)
	features.OnsetDensity = g.calculateOnsetDensity(energies, float64(sampleRate)/float64(hopSize))
	features.TempoVariation = g.calculateTempoVariation(energies)

	return features, nil
}

// extractEnergyFeatures extracts universal energy features
func (g *GeneralFeatureExtractor) extractEnergyFeatures(pcm []float64, sampleRate int) (*EnergyFeatures, error) {
	features := &EnergyFeatures{}
	frameSize := sampleRate * 40 / 1000
	hopSize := frameSize / 2

	numFrames := (len(pcm)-frameSize)/hopSize + 1
	if numFrames <= 0 {
		return features, fmt.Errorf("insufficient audio length")
	}

	features.ShortTimeEnergy = make([]float64, numFrames)
	features.EnergyEntropy = make([]float64, numFrames)
	features.CrestFactor = make([]float64, numFrames)

	energies := make([]float64, numFrames)
	for i := range numFrames {
		start := i * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		frame := pcm[start:end]
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

		rms := math.Sqrt(energy)
		if rms > 0 {
			features.CrestFactor[i] = maxVal / rms
		}

		features.EnergyEntropy[i] = g.calculateEnergyEntropy(frame)
	}

	features.EnergyVariance = g.calculateVariance(energies)
	features.LoudnessRange = g.calculateLoudnessRange(energies)
	return features, nil
}

// extractChromaFeatures extracts chroma with balanced approach
func (g *GeneralFeatureExtractor) extractChromaFeatures(spectrogram *analyzers.SpectrogramResult, sampleRate int) ([][]float64, error) {
	chromaBins := g.config.ChromaBins
	if chromaBins == 0 {
		chromaBins = 12
	}

	chroma := make([][]float64, spectrogram.TimeFrames)
	freqs := make([]float64, spectrogram.FreqBins)
	for i := 0; i < spectrogram.FreqBins; i++ {
		freqs[i] = float64(i) * float64(sampleRate) / float64((spectrogram.FreqBins-1)*2)
	}

	for t := 0; t < spectrogram.TimeFrames; t++ {
		chroma[t] = make([]float64, chromaBins)
		magnitude := spectrogram.Magnitude[t]

		for f := range len(magnitude) {
			freq := freqs[f]
			if freq < 100 || freq > 5000 {
				continue
			}

			if freq > 0 {
				midiNote := 12*math.Log2(freq/440.0) + 69
				if midiNote >= 0 {
					chromaClass := int(math.Round(midiNote)) % chromaBins
					if chromaClass >= 0 && chromaClass < chromaBins {
						chroma[t][chromaClass] += magnitude[f]
					}
				}
			}
		}

		// Normalize
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

	return chroma, nil
}

// Helper methods - spectral calculations
func (g *GeneralFeatureExtractor) calculateSpectralContrast(magnitude []float64, numBands int) []float64 {
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
		if len(bandMag) < 5 {
			continue
		}

		sorted := make([]float64, len(bandMag))
		copy(sorted, bandMag)
		sort.Float64s(sorted)

		valley := sorted[len(sorted)*15/100]
		peak := sorted[len(sorted)*85/100]

		if valley > 0 {
			contrast[band] = math.Log(peak / valley)
		}
	}

	return contrast
}

func (g *GeneralFeatureExtractor) calculateSpectralFlux(spectrogram *analyzers.SpectrogramResult) []float64 {
	if spectrogram.TimeFrames <= 1 {
		return nil
	}

	flux := make([]float64, spectrogram.TimeFrames-1)
	for t := 1; t < spectrogram.TimeFrames; t++ {
		sum := 0.0
		for f := 0; f < spectrogram.FreqBins; f++ {
			diff := spectrogram.Magnitude[t][f] - spectrogram.Magnitude[t-1][f]
			sum += diff * diff
		}
		flux[t-1] = math.Sqrt(sum)
	}

	return flux
}

func (g *GeneralFeatureExtractor) calculateSpectralCentroid(magnitude, freqs []float64) float64 {
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

func (g *GeneralFeatureExtractor) calculateSpectralRolloff(magnitude, freqs []float64, threshold float64) float64 {
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

func (g *GeneralFeatureExtractor) calculateSpectralBandwidth(magnitude, freqs []float64, centroid float64) float64 {
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

func (g *GeneralFeatureExtractor) calculateSpectralFlatness(magnitude []float64) float64 {
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

func (g *GeneralFeatureExtractor) calculateSpectralCrest(magnitude []float64) float64 {
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

func (g *GeneralFeatureExtractor) calculateSpectralSlope(magnitude, freqs []float64) float64 {
	if len(magnitude) != len(freqs) || len(magnitude) < 2 {
		return 0
	}

	n := 0
	sumX, sumY, sumXY, sumXX := 0.0, 0.0, 0.0, 0.0
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

// Helper methods - temporal and energy calculations
func (g *GeneralFeatureExtractor) calculateDynamicRange(pcm []float64) float64 {
	if len(pcm) == 0 {
		return 0
	}

	sorted := make([]float64, len(pcm))
	for i, sample := range pcm {
		sorted[i] = math.Abs(sample)
	}
	sort.Float64s(sorted)

	p5 := sorted[len(sorted)/20]
	p95 := sorted[19*len(sorted)/20]
	if p5 > 1e-10 {
		return 20 * math.Log10(p95/p5)
	}
	return 0
}

func (g *GeneralFeatureExtractor) calculateSilenceRatio(energies []float64) float64 {
	if len(energies) == 0 {
		return 0
	}

	meanEnergy := g.calculateMean(energies)
	threshold := meanEnergy * 0.1
	silentFrames := 0
	for _, energy := range energies {
		if energy < threshold {
			silentFrames++
		}
	}
	return float64(silentFrames) / float64(len(energies))
}

func (g *GeneralFeatureExtractor) calculateOnsetDensity(energies []float64, frameRate float64) float64 {
	if len(energies) < 5 {
		return 0
	}

	onsets := 0
	threshold := 1.3
	windowSize := 3
	for i := windowSize; i < len(energies)-1; i++ {
		current := energies[i]
		localAvg := 0.0
		for j := i - windowSize; j < i; j++ {
			if j >= 0 {
				localAvg += energies[j]
			}
		}
		localAvg /= float64(windowSize)

		if current > localAvg*threshold && current > energies[i+1] {
			onsets++
		}
	}

	duration := float64(len(energies)) / frameRate
	if duration == 0 {
		return 0
	}
	return float64(onsets) / duration
}

func (g *GeneralFeatureExtractor) calculateTempoVariation(energies []float64) float64 {
	if len(energies) < 10 {
		return 0
	}

	peaks := make([]int, 0)
	windowSize := 2
	meanEnergy := g.calculateMean(energies)
	threshold := meanEnergy * 1.1

	for i := windowSize; i < len(energies)-windowSize; i++ {
		current := energies[i]
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

	intervals := make([]float64, 0)
	for i := 1; i < len(peaks); i++ {
		interval := float64(peaks[i] - peaks[i-1])
		intervals = append(intervals, interval)
	}

	if len(intervals) == 0 {
		return 0
	}

	mean := g.calculateMean(intervals)
	variance := g.calculateVariance(intervals)
	if mean == 0 {
		return 0
	}
	return math.Sqrt(variance) / mean
}

func (g *GeneralFeatureExtractor) calculateEnergyEntropy(frame []float64) float64 {
	if len(frame) == 0 {
		return 0
	}

	numSubFrames := 8
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

func (g *GeneralFeatureExtractor) calculateLoudnessRange(energies []float64) float64 {
	if len(energies) == 0 {
		return 0
	}

	sorted := make([]float64, len(energies))
	copy(sorted, energies)
	sort.Float64s(sorted)

	p10 := sorted[len(sorted)/10]
	p90 := sorted[9*len(sorted)/10]
	if p10 > 0 {
		return 20 * math.Log10(p90/p10)
	}
	return 0
}

// Mel filter bank and DCT methods
func (g *GeneralFeatureExtractor) createMelFilterBank(numFilters int, lowFreq, highFreq float64, freqBins, sampleRate int) [][]float64 {
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

func (g *GeneralFeatureExtractor) applyMelFilters(magnitude []float64, filterBank [][]float64) []float64 {
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

func (g *GeneralFeatureExtractor) applyDCT(logMelSpectrum []float64, numCoeffs int) []float64 {
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

// Basic utility methods
func (g *GeneralFeatureExtractor) calculatePeakAmplitude(pcm []float64) float64 {
	maxVal := 0.0
	for _, sample := range pcm {
		if math.Abs(sample) > maxVal {
			maxVal = math.Abs(sample)
		}
	}
	return maxVal
}

func (g *GeneralFeatureExtractor) calculateAverageAmplitude(pcm []float64) float64 {
	if len(pcm) == 0 {
		return 0
	}
	sum := 0.0
	for _, sample := range pcm {
		sum += math.Abs(sample)
	}
	return sum / float64(len(pcm))
}

func (g *GeneralFeatureExtractor) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, val := range values {
		sum += val
	}
	return sum / float64(len(values))
}

func (g *GeneralFeatureExtractor) calculateVariance(values []float64) float64 {
	if len(values) <= 1 {
		return 0
	}
	mean := g.calculateMean(values)
	variance := 0.0
	for _, val := range values {
		diff := val - mean
		variance += diff * diff
	}
	return variance / float64(len(values))
}
