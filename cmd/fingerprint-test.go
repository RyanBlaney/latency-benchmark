package cmd

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/extractors"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/transcode"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

var (
	ftVerbose         bool
	ftSegmentDuration time.Duration
	ftTimeout         time.Duration
	ftContentType     string
	ftMinConfidence   float64
	ftMaxOffsetSec    float64
)

var ftCmd = &cobra.Command{
	Use:   "fingerprint-test [url1] [url2]",
	Short: "Test aligned fingerprint comparison",
	Long:  `Test fingerprint generation, alignment detection, and similarity comparison between two audio sources.`,
	Args:  cobra.ExactArgs(2),
	RunE:  runFingerprintTest,
}

func init() {
	rootCmd.AddCommand(ftCmd)

	ftCmd.Flags().BoolVarP(&ftVerbose, "verbose", "v", false, "verbose output")
	ftCmd.Flags().DurationVarP(&ftSegmentDuration, "segment-duration", "t", time.Second*240, "audio segment duration")
	ftCmd.Flags().DurationVarP(&ftTimeout, "timeout", "T", time.Second*240, "timeout for operations")
	ftCmd.Flags().StringVar(&ftContentType, "content-type", "news", "content type (music, news, talk, sports)")
	ftCmd.Flags().Float64Var(&ftMinConfidence, "min-confidence", 0.25, "minimum confidence for valid alignment")
	ftCmd.Flags().Float64Var(&ftMaxOffsetSec, "max-offset", 190, "maximum time offset for alignment (seconds)")
}

type AudioStream struct {
	ID              string
	URL             string
	AudioData       *common.AudioData
	Fingerprint     *fingerprint.AudioFingerprint
	LoadTime        time.Duration
	FingerprintTime time.Duration
	Error           error
}

type ComparisonResult struct {
	Stream1           *AudioStream
	Stream2           *AudioStream
	AlignmentFeatures *extractors.AlignmentFeatures
	SimilarityResult  *fingerprint.SimilarityResult
	AlignmentTime     time.Duration
	ComparisonTime    time.Duration
}

func runFingerprintTest(cmd *cobra.Command, args []string) error {
	url1, url2 := args[0], args[1]

	fmt.Printf("🎵 INTEGRATED FINGERPRINT TEST\n")
	fmt.Printf("==============================\n")
	fmt.Printf("Stream 1: %s\n", url1)
	fmt.Printf("Stream 2: %s\n", url2)
	fmt.Printf("Content Type: %s\n", ftContentType)
	fmt.Printf("Duration: %.1fs\n", ftSegmentDuration.Seconds())
	fmt.Printf("Max Offset: %.1fs\n", ftMaxOffsetSec)
	fmt.Printf("\n")

	ctx, cancel := context.WithTimeout(context.Background(), ftTimeout)
	defer cancel()

	timer := NewPerformanceTimer()
	timer.StartEvent("total")

	// Step 1: Load and fingerprint both streams
	fmt.Printf("📥 Loading and Fingerprinting Streams\n")
	fmt.Printf("=====================================\n")

	stream1 := loadAudioStream(ctx, "stream_1", url1)
	stream2 := loadAudioStream(ctx, "stream_2", url2)

	if stream1.Error != nil {
		return fmt.Errorf("failed to process stream 1: %v", stream1.Error)
	}
	if stream2.Error != nil {
		return fmt.Errorf("failed to process stream 2: %v", stream2.Error)
	}

	fmt.Printf("✅ %s: %.1fs audio downloaded in %dms\n",
		stream1.URL, stream1.AudioData.Duration.Seconds(), stream1.LoadTime.Milliseconds())
	fmt.Printf("✅ %s: %.1fs audio downloaded in %dms\n",
		stream2.URL, stream2.AudioData.Duration.Seconds(), stream2.LoadTime.Milliseconds())
	fmt.Printf("\n")

	// Step 2: Perform alignment detection
	fmt.Printf("⏰ Detecting Temporal Alignment\n")
	fmt.Printf("===============================\n")

	timer.StartEvent("alignment")
	alignmentFeatures, err := detectAlignment(stream1, stream2)
	timer.EndEvent("alignment")

	if err != nil {
		fmt.Printf("⚠️  Alignment detection failed: %v\n", err)
		alignmentFeatures = nil
	} else {
		isValid := alignmentFeatures.OffsetConfidence >= ftMinConfidence
		status := map[bool]string{true: "✅", false: "❌"}[isValid]

		fmt.Printf("%s Alignment detected: offset=%.2fs, confidence=%.3f, method=%s\n",
			status, alignmentFeatures.TemporalOffset, alignmentFeatures.OffsetConfidence, alignmentFeatures.Method)

		if ftVerbose && isValid {
			fmt.Printf("   Similarity: %.3f, Quality: %.3f\n",
				alignmentFeatures.OverallSimilarity, alignmentFeatures.AlignmentQuality)
		}
	}
	fmt.Printf("\n")

	// Step 3: Truncate audio outside of aligned bounds
	alignedAudio1, alignedAudio2, err := truncateToAlignedSegments(
		stream1.AudioData, stream2.AudioData, alignmentFeatures)
	if err != nil {
		return fmt.Errorf("failed to truncate aligned segments: %v", err)
	}

	// Step 4: Fingerprinting
	fmt.Printf("Fingerprint Generation\n")
	fmt.Printf("=========================\n")

	timer.StartEvent("fingerprinting")

	stream1.Fingerprint, err = generateFingerprint(alignedAudio1, stream1.URL)
	if err != nil {
		return fmt.Errorf("failed to generate fingerprint for %s: %v", stream1.ID, err)
	}

	stream2.Fingerprint, err = generateFingerprint(alignedAudio2, stream2.URL)
	if err != nil {
		return fmt.Errorf("failed to generate fingerprint for %s: %v", stream2.ID, err)
	}

	timer.EndEvent("fingerprinting")

	// Step 5: Compare fingerprints
	fmt.Printf("🔍 Fingerprint Comparison\n")
	fmt.Printf("=========================\n")

	timer.StartEvent("comparison")

	// Create comparator
	comparisonConfig := fingerprint.ContentOptimizedComparisonConfig(parseContentType(ftContentType))
	comparator := fingerprint.NewFingerprintComparator(comparisonConfig)

	// Compare with alignment (if available)
	var resultWithAlign *fingerprint.SimilarityResult
	if alignmentFeatures != nil && alignmentFeatures.OffsetConfidence >= ftMinConfidence {
		// alignmentConfig := config.AlignmentConfigForContent(parseContentType(ftContentType))
		resultWithAlign, err = comparator.Compare(
			stream1.Fingerprint, stream2.Fingerprint)
		if err != nil {
			fmt.Printf("⚠️  Comparison with alignment failed: %v\n", err)
			resultWithAlign = nil
		}
	}

	timer.EndEvent("comparison")

	if resultWithAlign != nil {
		fmt.Printf("With Alignment:    similarity=%.3f, confidence=%.3f (offset=%.2fs)\n",
			resultWithAlign.OverallSimilarity, resultWithAlign.Confidence, resultWithAlign.TemporalOffset)
	} else {
		fmt.Printf("With Alignment:    skipped (no valid alignment)\n")
	}
	fmt.Printf("\n")

	if ftVerbose {
		fmt.Printf("📊 Detailed Analysis\n")
		fmt.Printf("====================\n")
		printDetailedResults(resultWithAlign, stream1, stream2, alignmentFeatures)
		fmt.Printf("\n")
	}

	// Step 6: Final summary
	timer.EndEvent("total")

	fmt.Printf("⏱️  Performance Summary\n")
	fmt.Printf("=======================\n")
	fmt.Printf("Total Time:     %.0fms\n", timer.GetDuration("total").Seconds()*1000)
	fmt.Printf("Loading:        %.0fms\n", (stream1.LoadTime+stream2.LoadTime).Seconds()*1000)
	fmt.Printf("Fingerprinting: %.0fms\n", (stream1.FingerprintTime+stream2.FingerprintTime).Seconds()*1000)
	fmt.Printf("Alignment:      %.0fms\n", timer.GetDuration("alignment").Seconds()*1000)
	fmt.Printf("Comparison:     %.0fms\n", timer.GetDuration("comparison").Seconds()*1000)
	fmt.Printf("\n")

	fmt.Printf("🎯 Verdict\n")
	fmt.Printf("==========\n")

	bestSimilarity := resultWithAlign.OverallSimilarity

	if bestSimilarity > 0.8 {
		fmt.Printf("✅ STRONG MATCH: Streams are very likely from the same source (%.1f%% similar)\n", bestSimilarity*100)
	} else if bestSimilarity > 0.6 {
		fmt.Printf("⚠️  PARTIAL MATCH: Streams may be related (%.1f%% similar)\n", bestSimilarity*100)
	} else {
		fmt.Printf("❌ NO MATCH: Streams appear to be different sources (%.1f%% similar)\n", bestSimilarity*100)
	}

	return nil
}

func loadAudio(ctx context.Context, input string) (*common.AudioData, error) {
	if isLocalFile(input) {
		return loadLocalFile(input)
	} else {
		return loadStreamURL(ctx, input)
	}
}

func loadAudioStream(ctx context.Context, id, url string) *AudioStream {
	stream := &AudioStream{ID: id, URL: url}

	// Load audio
	loadStart := time.Now()
	audioData, err := loadAudio(ctx, url)
	if err != nil {
		stream.Error = fmt.Errorf("failed to load audio: %w", err)
		return stream
	}
	stream.AudioData = audioData
	stream.LoadTime = time.Since(loadStart)

	return stream

}

func loadLocalFile(filePath string) (*common.AudioData, error) {
	cleanPath := strings.TrimPrefix(filePath, "file://")

	fileData, err := os.ReadFile(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	decoder := transcode.NewNormalizingDecoder(ftContentType)
	anyData, err := decoder.DecodeBytes(fileData)
	if err != nil {
		return nil, fmt.Errorf("failed to decode audio: %w", err)
	}

	if commonAudio, ok := anyData.(*common.AudioData); ok {
		// Truncate if needed
		if ftSegmentDuration > 0 {
			maxSamples := int(ftSegmentDuration.Seconds() * float64(commonAudio.SampleRate))
			if len(commonAudio.PCM) > maxSamples {
				commonAudio.PCM = commonAudio.PCM[:maxSamples]
			}
		}
		commonAudio.Duration = time.Duration(float64(len(commonAudio.PCM))/float64(commonAudio.SampleRate)) * time.Second
		return commonAudio, nil
	}

	audioData := common.ConvertToAudioData(anyData)
	if audioData == nil {
		return nil, fmt.Errorf("decoder returned unexpected type: %T", anyData)
	}
	return audioData, nil
}

func loadStreamURL(ctx context.Context, url string) (*common.AudioData, error) {
	managerConfig := &stream.ManagerConfig{
		StreamTimeout:        ftTimeout,
		OverallTimeout:       ftTimeout + (10 * time.Second),
		MaxConcurrentStreams: 1,
		ResultBufferSize:     1,
	}
	manager := stream.NewManagerWithConfig(managerConfig)

	results, err := manager.ExtractAudioSequential(ctx, []string{url}, ftSegmentDuration)
	if err != nil {
		return nil, err
	}

	if len(results.Results) == 0 || results.Results[0].Error != nil {
		if len(results.Results) > 0 {
			return nil, results.Results[0].Error
		}
		return nil, fmt.Errorf("no results from stream extraction")
	}

	return results.Results[0].AudioData, nil
}

func generateFingerprint(audioData *common.AudioData, url string) (*fingerprint.AudioFingerprint, error) {
	// Create fingerprint generator with EXACT working parameters
	fingerprintConfig := fingerprint.ContentOptimizedFingerprintConfig(config.ContentType(ftContentType))

	// FIX: Use the EXACT parameters that worked before
	fingerprintConfig.WindowSize = 1024 // Was 2048
	fingerprintConfig.HopSize = 256     // Was 512

	// FIX: Ensure FeatureConfig matches the working configuration
	fingerprintConfig.FeatureConfig.WindowSize = 1024 // Match windowSize
	fingerprintConfig.FeatureConfig.HopSize = 256     // Match hopSize
	fingerprintConfig.FeatureConfig.SampleRate = audioData.SampleRate

	// Speech-optimized settings for news content (match working config)
	if ftContentType == "news" || ftContentType == "talk" {
		fingerprintConfig.FeatureConfig.EnableSpeechFeatures = true
		fingerprintConfig.FeatureConfig.EnableHarmonicFeatures = false
		fingerprintConfig.FeatureConfig.EnableChroma = false
		fingerprintConfig.FeatureConfig.MFCCCoefficients = 12
		fingerprintConfig.FeatureConfig.FreqRange = [2]float64{80.0, 8000.0}
	}

	fingerprintGenerator := fingerprint.NewFingerprintGenerator(fingerprintConfig)

	// Convert to transcode.AudioData
	transcodeAudioData := &transcode.AudioData{
		PCM:        audioData.PCM,
		SampleRate: audioData.SampleRate,
		Channels:   audioData.Channels,
		Metadata: &transcode.StreamMetadata{
			URL:         url,
			Type:        "file",
			ContentType: ftContentType,
			SampleRate:  audioData.SampleRate,
			Channels:    audioData.Channels,
			Timestamp:   time.Now(),
		},
	}

	// Generate fingerprint
	return fingerprintGenerator.GenerateFingerprint(transcodeAudioData)
}

func detectAlignment(stream1, stream2 *AudioStream) (*extractors.AlignmentFeatures, error) {
	contentType := parseContentType(ftContentType)
	featureConfig := createFeatureConfig(contentType, stream1.AudioData.SampleRate)
	alignmentConfig := config.AlignmentConfigForContent(contentType)

	featureExtractor, err := extractors.NewFeatureExtractorFactory().CreateExtractor(contentType, *featureConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create feature extractor: %v", err)
	}

	analyzer := analyzers.NewSpectralAnalyzer(featureConfig.SampleRate)
	analyzers.DefaultWindowConfig()

	spectrogram1, err := analyzer.ComputeSTFTWithWindow(stream1.AudioData.PCM, featureConfig.WindowSize, featureConfig.HopSize, analyzers.WindowHann)
	if err != nil {
		return nil, fmt.Errorf("failed to compute STFT for %s: %v", stream1.ID, err)
	}

	spectrogram2, err := analyzer.ComputeSTFTWithWindow(stream2.AudioData.PCM, featureConfig.WindowSize, featureConfig.HopSize, analyzers.WindowHann)
	if err != nil {
		return nil, fmt.Errorf("failed to compute STFT for %s: %v", stream1.ID, err)
	}

	stream1Features, err := featureExtractor.ExtractFeatures(spectrogram1, stream1.AudioData.PCM, featureConfig.SampleRate)
	if err != nil {
		return nil, fmt.Errorf("failed to extract features for %s: %v", stream1.ID, err)
	}

	stream2Features, err := featureExtractor.ExtractFeatures(spectrogram2, stream2.AudioData.PCM, featureConfig.SampleRate)
	if err != nil {
		return nil, fmt.Errorf("failed to extract features for %s: %v", stream2.ID, err)
	}

	alignmentExtractor := extractors.NewAlignmentExtractorWithMaxLag(
		featureConfig, alignmentConfig, ftMaxOffsetSec)

	return alignmentExtractor.ExtractAlignmentFeatures(
		stream1Features, stream2Features,
		stream1.AudioData.PCM, stream2.AudioData.PCM,
		stream1.AudioData.SampleRate)
}

func printDetailedResults(resultWithAlign *fingerprint.SimilarityResult,
	stream1, stream2 *AudioStream, alignmentFeatures *extractors.AlignmentFeatures) {

	fmt.Printf("Stream Details:\n")
	fmt.Printf("  %s: %.1fs, %dHz, %d channels\n",
		stream1.ID, stream1.AudioData.Duration.Seconds(),
		stream1.AudioData.SampleRate, stream1.AudioData.Channels)
	fmt.Printf("  %s: %.1fs, %dHz, %d channels\n",
		stream2.ID, stream2.AudioData.Duration.Seconds(),
		stream2.AudioData.SampleRate, stream2.AudioData.Channels)
	fmt.Printf("\n")

	if resultWithAlign != nil {
		fmt.Printf("Feature Breakdown (with alignment):\n")
		for feature, distance := range resultWithAlign.FeatureDistances {
			similarity := 1.0 - distance
			fmt.Printf("  %s: %.3f\n", feature, similarity)
		}
		fmt.Printf("  Hash similarity: %.3f\n", resultWithAlign.HashSimilarity)
		fmt.Printf("\n")
	}

	if alignmentFeatures != nil {
		fmt.Printf("Alignment Details:\n")
		fmt.Printf("  Method: %s\n", alignmentFeatures.Method)
		fmt.Printf("  Offset: %.3fs\n", alignmentFeatures.TemporalOffset)
		fmt.Printf("  Confidence: %.3f\n", alignmentFeatures.OffsetConfidence)
		fmt.Printf("  Quality: %.3f\n", alignmentFeatures.AlignmentQuality)
		fmt.Printf("  Time stretch: %.3f\n", alignmentFeatures.TimeStretch)
	}
}

// Helper functions
func parseContentType(contentTypeStr string) config.ContentType {
	switch strings.ToLower(contentTypeStr) {
	case "music":
		return config.ContentMusic
	case "news":
		return config.ContentNews
	case "talk":
		return config.ContentTalk
	case "sports":
		return config.ContentSports
	case "mixed":
		return config.ContentMixed
	default:
		return config.ContentNews // Default for this test
	}
}

// TODO: feature config by content type
func createFeatureConfig(contentType config.ContentType, sampleRate int) *config.FeatureConfig {
	featureConfig := &config.FeatureConfig{
		WindowSize: 1024,
		HopSize:    256,
		SampleRate: sampleRate,
		FreqRange:  [2]float64{80.0, 8000.0},

		EnableChroma:           false,
		EnableMFCC:             true,
		EnableSpectralContrast: false,
		EnableTemporalFeatures: true,
		EnableSpeechFeatures:   true,
		EnableHarmonicFeatures: false,

		MFCCCoefficients: 12,
		ChromaBins:       12,

		SimilarityWeights: map[string]float64{
			"mfcc":     0.50,
			"spectral": 0.25,
			"temporal": 0.15,
			"speech":   0.10,
		},
		MatchThreshold: 0.70,
	}

	return featureConfig
}

func isLocalFile(input string) bool {
	return strings.HasPrefix(input, "file://") ||
		strings.HasPrefix(input, "/") ||
		strings.HasPrefix(input, "./") ||
		strings.HasPrefix(input, "../") ||
		(!strings.HasPrefix(input, "http://") && !strings.HasPrefix(input, "https://"))
}

func truncateToAlignedSegments(audio1, audio2 *common.AudioData, alignment *extractors.AlignmentFeatures) (*common.AudioData, *common.AudioData, error) {
	offsetSeconds := alignment.TemporalOffset
	sampleRate := float64(audio1.SampleRate)

	// Convert to samples with proper rounding
	offsetSamples := int(math.Round(math.Abs(offsetSeconds) * sampleRate))

	fmt.Printf("DEBUG: Offset=%.2fs, OffsetSamples=%d\n", offsetSeconds, offsetSamples)
	fmt.Printf("DEBUG: Audio1 len=%d samples (%.1fs), Audio2 len=%d samples (%.1fs)\n",
		len(audio1.PCM), float64(len(audio1.PCM))/sampleRate,
		len(audio2.PCM), float64(len(audio2.PCM))/sampleRate)

	var start1, start2, commonLength int

	if offsetSeconds > 0 {
		// Stream 2 is ahead: skip beginning of stream 2, keep beginning of stream 1
		start1 = 0
		start2 = offsetSamples

		if start2 >= len(audio2.PCM) {
			return nil, nil, fmt.Errorf("offset too large: need to skip %d samples but audio2 only has %d", start2, len(audio2.PCM))
		}

		// Calculate how much audio remains after skipping
		remaining1 := len(audio1.PCM) - start1 // All of audio1
		remaining2 := len(audio2.PCM) - start2 // Audio2 after skipping
		commonLength = min(remaining1, remaining2)

	} else if offsetSeconds < 0 {
		// Stream 1 is ahead: skip beginning of stream 1, keep beginning of stream 2
		start1 = offsetSamples
		start2 = 0

		if start1 >= len(audio1.PCM) {
			return nil, nil, fmt.Errorf("offset too large: need to skip %d samples but audio1 only has %d", start1, len(audio1.PCM))
		}

		remaining1 := len(audio1.PCM) - start1 // Audio1 after skipping
		remaining2 := len(audio2.PCM) - start2 // All of audio2
		commonLength = min(remaining1, remaining2)

	} else {
		// No offset
		start1, start2 = 0, 0
		commonLength = min(len(audio1.PCM), len(audio2.PCM))
	}

	if commonLength <= 0 {
		return nil, nil, fmt.Errorf("no overlapping audio after alignment")
	}

	// Add some padding to ensure we get the best aligned portion
	// Skip a bit more at the beginning and end to avoid edge effects
	paddingSamples := int(0.5 * sampleRate) // 0.5 second padding
	if commonLength > 2*paddingSamples {
		start1 += paddingSamples
		start2 += paddingSamples
		commonLength -= 2 * paddingSamples
	}

	fmt.Printf("DEBUG: After alignment - start1=%d, start2=%d, commonLength=%d (%.1fs)\n",
		start1, start2, commonLength, float64(commonLength)/sampleRate)

	// Create aligned audio segments
	aligned1 := &common.AudioData{
		PCM:        audio1.PCM[start1 : start1+commonLength],
		SampleRate: audio1.SampleRate,
		Channels:   audio1.Channels,
		Duration:   time.Duration(float64(commonLength) / sampleRate * float64(time.Second)),
	}

	aligned2 := &common.AudioData{
		PCM:        audio2.PCM[start2 : start2+commonLength],
		SampleRate: audio2.SampleRate,
		Channels:   audio2.Channels,
		Duration:   time.Duration(float64(commonLength) / sampleRate * float64(time.Second)),
	}

	return aligned1, aligned2, nil
}
