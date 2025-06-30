package audio

import (
	"math"
	"strings"

	"github.com/tunein/cdn-benchmark-cli/pkg/audio/logging"
)

// ContentDetector handles content type detection from metadata and audio analysis
type ContentDetector struct {
	config *ContentAwareConfig
	logger logging.Logger
}

// NewContentDetector creates a new content detector
func NewContentDetector(config *ContentAwareConfig) *ContentDetector {
	logger := logging.WithFields(logging.Fields{
		"component": "content_detector",
	})

	return &ContentDetector{
		config: config,
		logger: logger,
	}
}

// DetectedContentType detects content type from AudioData using metadata and optionally acoustic analysis
func (cd *ContentDetector) DetectContentType(audioData *AudioData) ContentType {
	logger := cd.logger.WithFields(logging.Fields{
		"function": "DetectContentType",
		"url":      getURLFromAudioData(audioData),
	})

	// First try metadata-based detection
	if audioData.Metadata != nil {
		metadataType := detectContentTypeFromMetadata(audioData.Metadata)
		if metadataType != ContentUnknown {
			logger.Info("Content type detected from metadata", logging.Fields{
				"content_type":     metadataType,
				"detection_method": "metadata",
				"source":           getMetadataSource(audioData.Metadata),
			})
			return metadataType
		}
	}

	// Fall back to acoustic analysis if enabled
	if cd.config.EnableContentDetection && len(audioData.PCM) > 0 {
		acousticType := cd.DetectFromAudio(audioData.PCM, audioData.SampleRate)
		if acousticType != ContentUnknown {
			logger.Info("Content type detected from audio analysis", logging.Fields{
				"content_type":     acousticType,
				"detection_method": "acoustic",
			})
			return acousticType
		}
	}

	// Use default content type
	logging.Debug("Using default content type", logging.Fields{
		"content	gtype":    cd.config.DefaultContentType,
		"detection_method": "default",
	})

	return cd.config.DefaultContentType
}

// DetectFromAudio performs acoustic analysis to detect content type
func (cd *ContentDetector) DetectFromAudio(pcm []float64, sampleRate int) ContentType {
	if len(pcm) == 0 {
		return ContentUnknown
	}

	logger := cd.logger.WithFields(logging.Fields{
		"function":    "DetectFromAudio",
		"sample_rate": sampleRate,
		"samples":     len(pcm),
	})

	// Analyze acoustic characteristics
	features := cd.extractAcousticFeatures(pcm, sampleRate)

	logger.Debug("Acoustic features extracted", logging.Fields{
		"zero_crossing_rate": features.ZeroCrossingRate,
		"spectral_centroid":  features.SpectralCentroid,
		"energy_variance":    features.EnergyVariance,
		"silence_ratio":      features.SilenceRatio,
		"harmonic_ratio":     features.HarmonicRatio,
	})

	// Classify based on features
	contentType := cd.classifyFromFeatures(features)

	logger.Info("Content type classified from acoustic analysis", logging.Fields{
		"content_type": contentType,
		"confidence":   features.ClassificationConfidence,
	})

	return contentType
}

// AcousticFeatures holds features used for content classification
type AcousticFeatures struct {
	ZeroCrossingRate         float64 `json:"zero_crossing_rate"`
	SpectralCentroid         float64 `json:"spectral_centroid"`
	EnergyVariance           float64 `json:"energy_variance"`
	SilenceRatio             float64 `json:"silence_ratio"`
	HarmonicRatio            float64 `json:"harmonic_ratio"`
	LowFreqEnergy            float64 `json:"low_freq_energy"`
	HighFreqEnergy           float64 `json:"high_freq_energy"`
	DynamicRange             float64 `json:"dynamic_range"`
	TemporalStability        float64 `json:"temporal_stability"`
	ClassificationConfidence float64 `json:"classification_confidence"`
}

// extractAcousticFeatures extracts features for content classification
func (cd *ContentDetector) extractAcousticFeatures(pcm []float64, sampleRate int) *AcousticFeatures {
	features := &AcousticFeatures{}

	// Zero Crossing Rate (higher for speech, lower for music)
	features.ZeroCrossingRate = cd.calculateZeroCrossingRate(pcm)

	// Basic spectral analysis
	// TODO: specify in config
	windowSize := 2048
	windowSize = min(windowSize, len(pcm))

	// Use first window for quick analysis
	window := pcm[:windowSize]
	spectrum := cd.computeBasicSpectrum(window)

	// Spectral Centroid (brightness)
	features.SpectralCentroid = cd.calculateSpectralCentroid(spectrum, sampleRate)

	// Energy-based features
	features.EnergyVariance = cd.calculateEnergyVariance(pcm)
	features.SilenceRatio = cd.calculateSilenceRatio(pcm)
	features.DynamicRange = cd.calculateDynamicRange(pcm)

	// Frequency distribution
	features.LowFreqEnergy, features.HighFreqEnergy = cd.calculateFreqEnergyRatio(spectrum)

	// Harmonic content (simplified)
	features.HarmonicRatio = cd.calculateHarmonicRatio(spectrum)

	// Temporal stability
	features.TemporalStability = cd.calculateTemporalStability(pcm, sampleRate)

	return features
}

// classifyFromFeatures classifies content type based on acoustic features
func (cd *ContentDetector) classifyFromFeatures(features *AcousticFeatures) ContentType {
	scores := make(map[ContentType]float64)

	// Music characteristics: Low ZCR, high harmonic content, temporal stability
	// TODO remove magic numbers
	musicScore := 0.0
	if features.ZeroCrossingRate < 0.1 {
		musicScore += 2.0
	}
	if features.HarmonicRatio > 0.3 {
		musicScore += 2.0
	}
	if features.TemporalStability > 0.5 {
		musicScore += 1.0
	}
	if features.DynamicRange > 20 {
		musicScore += 1.0
	}
	scores[ContentMusic] = musicScore

	// Speech/News characteristics: Higher ZCR, spectral centroid in speech range, lower harmonic content
	speechScore := 0.0
	if features.ZeroCrossingRate > 0.05 && features.ZeroCrossingRate < 0.3 {
		speechScore += 2.0
	}
	if features.SpectralCentroid > 800 && features.SpectralCentroid < 3000 {
		speechScore += 2.0
	}
	if features.HarmonicRatio < 0.2 {
		speechScore += 1.0
	}
	if features.SilenceRatio > 0.1 && features.SilenceRatio < 0.4 {
		speechScore += 1.0
	}
	scores[ContentNews] = speechScore
	scores[ContentTalk] = speechScore * 0.9 // Slightly lower for talk

	// Sports characteristics: High energy variance, dynamic range, mixed content
	sportsScore := 0.0
	if features.EnergyVariance > 0.3 {
		sportsScore += 2.0
	}
	if features.DynamicRange > 30 {
		sportsScore += 1.5
	}
	if features.TemporalStability < 0.4 {
		sportsScore += 1.0
	}
	scores[ContentSports] = sportsScore

	// Find best match
	bestType := ContentUnknown
	bestScore := cd.config.AutoDetectThreshold

	for contentType, score := range scores {
		if score > bestScore {
			bestScore = score
			bestType = contentType
		}
	}

	// Set confidence
	features.ClassificationConfidence = bestScore / 6.0 // Normalize to 0-1

	return bestType
}

// calculateZeroCrossingRate computes zero crossing rate
func (cd *ContentDetector) calculateZeroCrossingRate(pcm []float64) float64 {
	if len(pcm) <= 1 {
		return 0
	}

	crossings := 0
	for i := 1; i < len(pcm); i++ {
		if (pcm[i-1] >= 0 && pcm[i] < 0) || (pcm[i-1] < 0 && pcm[i] >= 0) {
			crossings++
		}
	}

	return float64(crossings) / float64(len(pcm)-1)
}

// calculateSpectralCentroid computes spectral centroid from magnitude spectrum
func (cd *ContentDetector) calculateSpectralCentroid(spectrum []float64, sampleRate int) float64 {
	weightedSum := 0.0
	magnitudeSum := 0.0

	for i, magnitude := range spectrum {
		frequency := float64(i) * float64(sampleRate) / float64(len(spectrum)*2)
		weightedSum += frequency * magnitude
		magnitudeSum += magnitude
	}

	if magnitudeSum == 0 {
		return 0
	}

	return weightedSum / magnitudeSum
}

// calculateEnergyVariance computes energy variance over time
func (cd *ContentDetector) calculateEnergyVariance(pcm []float64) float64 {
	frameSize := 1024
	if len(pcm) < frameSize*2 {
		return 0
	}

	var energies []float64
	for i := 0; i < len(pcm)-frameSize; i += frameSize / 2 {
		energy := 0.0
		for j := 0; j < frameSize && i+j < len(pcm); j++ {
			energy += pcm[i+j] * pcm[i+j]
		}
		energies = append(energies, energy/float64(frameSize))
	}

	if len(energies) <= 1 {
		return 0
	}

	// Calculate variance
	mean := 0.0
	for _, energy := range energies {
		mean += energy
	}
	mean /= float64(len(energies))

	variance := 0.0
	for _, energy := range energies {
		diff := energy - mean
		variance += diff * diff
	}
	variance /= float64(len(energies))

	return variance
}

// calculateSilenceRatio computes ratio of silent frames
func (cd *ContentDetector) calculateSilenceRatio(pcm []float64) float64 {
	frameSize := 1024
	silentFrames := 0
	totalFrames := 0
	threshold := 0.01 // RMS threshold for silence

	for i := 0; i < len(pcm)-frameSize; i += frameSize / 2 {
		rms := 0.0
		for j := 0; j < frameSize && i+j < len(pcm); j++ {
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

// calculateDynamicRange computes dynamic range in dB
func (cd *ContentDetector) calculateDynamicRange(pcm []float64) float64 {
	if len(pcm) == 0 {
		return 0
	}

	maxVal := 0.0
	minVal := math.Inf(1)

	for _, sample := range pcm {
		abs := math.Abs(sample)
		if abs > maxVal {
			maxVal = abs
		}
		if abs < minVal && abs > 1e-10 {
			minVal = abs
		}
	}

	if minVal == 0 || minVal == math.Inf(1) {
		return 0
	}

	return 20 * math.Log10(maxVal/minVal)
}

// calculateFreqEnergyRatio computes low/high frequency energy ratio
func (cd *ContentDetector) calculateFreqEnergyRatio(spectrum []float64) (float64, float64) {
	if len(spectrum) == 0 {
		return 0, 0
	}

	splitPoint := len(spectrum) / 4 // Split at 1/4 of Nyquist frequency

	lowEnergy := 0.0
	for i := 0; i < splitPoint && i < len(spectrum); i++ {
		lowEnergy += spectrum[i] * spectrum[i]
	}

	highEnergy := 0.0
	for i := splitPoint; i < len(spectrum); i++ {
		highEnergy += spectrum[i] * spectrum[i]
	}

	totalEnergy := lowEnergy + highEnergy
	if totalEnergy == 0 {
		return 0, 0
	}

	return lowEnergy / totalEnergy, highEnergy / totalEnergy
}

// calculateHarmonicRatio estimates harmonic content
func (cd *ContentDetector) calculateHarmonicRatio(spectrum []float64) float64 {
	if len(spectrum) < 10 {
		return 0
	}

	// Find peaks in spectrum
	peaks := make([]int, 0)
	for i := 2; i < len(spectrum)-2; i++ {
		if spectrum[i] > spectrum[i-1] && spectrum[i] > spectrum[i+1] &&
			spectrum[i] > spectrum[i-2] && spectrum[i] > spectrum[i+2] {
			peaks = append(peaks, i)
		}
	}

	if len(peaks) < 2 {
		return 0
	}

	// Check for harmonic relationships (simplified)
	harmonicPeaks := 0
	fundamentalBin := peaks[0]

	for _, peak := range peaks[1:] {
		ratio := float64(peak) / float64(fundamentalBin)
		// Check if ratio is close to an integer (harmonic)
		if math.Abs(ratio-math.Round(ratio)) < 0.1 {
			harmonicPeaks++
		}
	}

	return float64(harmonicPeaks) / float64(len(peaks)-1)
}

// calculateTemporalStability measures how stable the signal is over time
func (cd *ContentDetector) calculateTemporalStability(pcm []float64, sampleRate int) float64 {
	frameSize := sampleRate / 10 // 100ms frames
	if len(pcm) < frameSize*3 {
		return 0
	}

	var frameFeatures []float64
	for i := 0; i < len(pcm)-frameSize; i += frameSize {
		// Calculate frame energy
		energy := 0.0
		for j := 0; j < frameSize && i+j < len(pcm); j++ {
			energy += pcm[i+j] * pcm[i+j]
		}
		frameFeatures = append(frameFeatures, energy)
	}

	if len(frameFeatures) <= 1 {
		return 0
	}

	// Calculate coefficient of variation (stability measure)
	mean := 0.0
	for _, feature := range frameFeatures {
		mean += feature
	}
	mean /= float64(len(frameFeatures))

	if mean == 0 {
		return 0
	}

	variance := 0.0
	for _, feature := range frameFeatures {
		diff := feature - mean
		variance += diff * diff
	}
	variance /= float64(len(frameFeatures))

	cv := math.Sqrt(variance) / mean
	return math.Max(0, 1-cv) // Higher value = more stable
}

// computeBasicSpectrum computes a basic magnitude spectrum using DFT
func (cd *ContentDetector) computeBasicSpectrum(signal []float64) []float64 {
	n := len(signal)
	spectrum := make([]float64, n/2+1)

	for k := range len(spectrum) {
		real, imag := 0.0, 0.0
		for n := range len(signal) {
			angle := -2 * math.Pi * float64(k) * float64(n) / float64(len(signal))
			real += signal[n] * math.Cos(angle)
			imag += signal[n] * math.Sin(angle)
		}
		spectrum[k] = math.Sqrt(real*real + imag*imag)
	}

	return spectrum
}

// Helper functions for metadata analysis

func getURLFromAudioData(audioData *AudioData) string {
	if audioData.Metadata != nil {
		return audioData.Metadata.URL
	}
	return "unknown"
}

func getMetadataSource(metadata *StreamMetadata) string {
	if metadata.ContentType != "" {
		return "content_type"
	}
	if metadata.Genre != "" {
		return "genre"
	}
	if metadata.Station != "" {
		return "station"
	}
	return "url_pattern"
}

// inferFromGenre infers content type from genre metadata
func inferFromGenre(genre string) ContentType {
	genre = strings.ToLower(strings.TrimSpace(genre))

	// Music genres
	musicGenres := []string{
		"rock", "pop", "jazz", "classical", "hip-hop", "hip hop", "country",
		"electronic", "blues", "reggae", "folk", "metal", "punk", "r&b",
		"soul", "funk", "dance", "techno", "house", "ambient", "indie",
		"alternative", "grunge", "ska", "latin", "world", "gospel",
	}

	// News/Talk genres
	newsGenres := []string{
		"news", "talk", "politics", "current affairs", "public radio",
		"discussion", "interview", "call-in", "spoken word", "commentary",
		"analysis", "reporting", "journalism", "public affairs",
	}

	// Sports genres
	sportsGenres := []string{
		"sports", "football", "basketball", "baseball", "soccer", "hockey",
		"tennis", "golf", "racing", "motorsports", "athletics", "cricket",
		"rugby", "boxing", "mma", "sports talk", "sports news",
	}

	for _, music := range musicGenres {
		if strings.Contains(genre, music) {
			return ContentMusic
		}
	}

	for _, news := range newsGenres {
		if strings.Contains(genre, news) {
			return ContentNews
		}
	}

	for _, sport := range sportsGenres {
		if strings.Contains(genre, sport) {
			return ContentSports
		}
	}

	// Check for talk radio patterns
	if strings.Contains(genre, "talk") && !strings.Contains(genre, "sports") {
		return ContentTalk
	}

	return ContentUnknown
}

// inferFromStation infers content type from station name and URL patterns
func inferFromStation(station, url string) ContentType {
	station = strings.ToLower(strings.TrimSpace(station))
	url = strings.ToLower(url)

	// News indicators
	newsIndicators := []string{
		"news", "npr", "bbc", "cnn", "cbc", "abc news", "nbc news",
		"fox news", "public radio", "current affairs", "talk radio",
	}

	// Sports indicators
	sportsIndicators := []string{
		"sports", "espn", "fox sports", "sports radio", "the fan",
		"sport", "athletic", "game", "stadium",
	}

	// Music indicators
	musicIndicators := []string{
		"fm", "music", "hits", "rock", "pop", "jazz", "country",
		"classic", "radio", "mix", "beat", "sound", "groove",
	}

	// Check station name and URL
	combined := station + " " + url

	for _, indicator := range newsIndicators {
		if strings.Contains(combined, indicator) {
			return ContentNews
		}
	}

	for _, indicator := range sportsIndicators {
		if strings.Contains(combined, indicator) {
			return ContentSports
		}
	}

	for _, indicator := range musicIndicators {
		if strings.Contains(combined, indicator) {
			return ContentMusic
		}
	}

	// Special cases for talk radio
	if strings.Contains(combined, "talk") && !strings.Contains(combined, "sports") {
		return ContentTalk
	}

	return ContentUnknown
}
