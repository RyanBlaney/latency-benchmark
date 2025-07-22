package cmd

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/config"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/analyzers"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/extractors"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/transcode"
)

var (
	ftVerbose         bool
	ftDebug           bool
	ftSegmentDuration time.Duration
	ftTimeout         time.Duration
	ftParallel        bool
	ftMaxConcurrent   int

	// Feature flags for testing components
	ftTestExtraction bool
	ftTestAlignment  bool
	ftTestComparison bool

	// Feature extraction parameters
	ftFeatureRate   float64
	ftMaxOffsetSec  float64
	ftMinConfidence float64
	ftContentType   string
)

var ftCmd = &cobra.Command{
	Use:   "fingerprint-test [url1] [url2] [url3...]",
	Short: "Test the complete fingerprint pipeline",
	Long:  `Test feature extraction, temporal alignment, and stream comparison for audio fingerprinting.`,
	Args:  cobra.MinimumNArgs(2),
	RunE:  runFingerprintTest,
}

func init() {
	rootCmd.AddCommand(ftCmd)

	// Basic flags
	ftCmd.Flags().BoolVarP(&ftVerbose, "verbose", "v", false,
		"verbose output (overrides global verbose)")
	ftCmd.Flags().BoolVarP(&ftDebug, "debug", "d", false,
		"debug logging mode")
	ftCmd.Flags().DurationVarP(&ftSegmentDuration, "segment-duration", "t", time.Second*30,
		"the downloaded length of each stream")
	ftCmd.Flags().DurationVarP(&ftTimeout, "timeout", "T", time.Second*120,
		"timeout for stream operations")
	ftCmd.Flags().BoolVarP(&ftParallel, "parallel", "p", true,
		"use parallel extraction (false for sequential)")
	ftCmd.Flags().IntVarP(&ftMaxConcurrent, "max-concurrent", "c", 5,
		"maximum concurrent streams for parallel mode")

	// Component test flags
	ftCmd.Flags().BoolVar(&ftTestExtraction, "test-extraction", true,
		"test feature extraction")
	ftCmd.Flags().BoolVar(&ftTestAlignment, "test-alignment", true,
		"test temporal alignment")
	ftCmd.Flags().BoolVar(&ftTestComparison, "test-comparison", true,
		"test fingerprint comparison")

	// Feature extraction parameters
	ftCmd.Flags().Float64Var(&ftFeatureRate, "feature-rate", 10.0,
		"feature extraction rate (Hz)")
	ftCmd.Flags().Float64Var(&ftMaxOffsetSec, "max-offset", 30.0,
		"maximum time offset for alignment (seconds)")
	ftCmd.Flags().Float64Var(&ftMinConfidence, "min-confidence", 0.6,
		"minimum confidence for valid alignment")
	ftCmd.Flags().StringVar(&ftContentType, "content-type", "music",
		"content type for feature extraction (music, news, talk, sports, mixed, general)")
}

// StreamData holds all data for a single stream
type StreamData struct {
	StreamID   string
	URL        string
	PCM        []float64
	SampleRate int
	Channels   int
	Duration   time.Duration
	Features   *extractors.ExtractedFeatures
	Error      error
}

// FingerprintTestResults holds all test results using new API
type FingerprintTestResults struct {
	Streams           []*StreamData
	AlignmentResults  map[string]*extractors.AlignmentFeatures
	ComparisonResults map[string]float64

	ExtractionTime time.Duration
	AlignmentTime  time.Duration
	ComparisonTime time.Duration
	TotalTime      time.Duration
}

func runFingerprintTest(cmd *cobra.Command, args []string) error {
	urls := args

	fmt.Printf("🎵 FINGERPRINT PIPELINE TEST\n")
	fmt.Printf("============================\n\n")

	// Configuration summary
	fmt.Printf("⚙️  Configuration:\n")
	fmt.Printf("   Mode: %s\n", map[bool]string{true: "Parallel", false: "Sequential"}[ftParallel])
	fmt.Printf("   Inputs: %d\n", len(urls))
	fmt.Printf("   Duration: %.1f seconds\n", ftSegmentDuration.Seconds())
	fmt.Printf("   Content Type: %s\n", ftContentType)
	fmt.Printf("   Feature Rate: %.1f Hz\n", ftFeatureRate)
	fmt.Printf("   Max Offset: %.1f seconds\n", ftMaxOffsetSec)

	// Check input types
	localFiles := 0
	streamUrls := 0
	for _, input := range urls {
		if isLocalFile(input) {
			localFiles++
		} else {
			streamUrls++
		}
	}

	fmt.Printf("   Input Types: %d local files, %d stream URLs\n", localFiles, streamUrls)

	fmt.Printf("\n🧪 Test Components:\n")
	fmt.Printf("   Feature Extraction: %s\n", enabledStatus(ftTestExtraction))
	fmt.Printf("   Temporal Alignment: %s\n", enabledStatus(ftTestAlignment))
	fmt.Printf("   Stream Comparison: %s\n", enabledStatus(ftTestComparison))
	fmt.Printf("\n")

	// Create context with overall timeout
	ctx, cancel := context.WithTimeout(context.Background(), ftTimeout)
	defer cancel()

	timer := NewPerformanceTimer()
	timer.StartEvent("overall")

	// Initialize test results
	testResults := &FingerprintTestResults{
		AlignmentResults:  make(map[string]*extractors.AlignmentFeatures),
		ComparisonResults: make(map[string]float64),
	}

	// PHASE 1: Audio Extraction & Feature Extraction (Combined)
	fmt.Printf("🎵 PHASE 1: Audio & Feature Extraction\n")
	fmt.Printf("======================================\n")

	timer.StartEvent("extraction")

	err := runCombinedExtraction(ctx, urls, testResults)
	if err != nil {
		return fmt.Errorf("❌ Combined extraction failed: %v", err)
	}

	timer.EndEvent("extraction")
	testResults.ExtractionTime = timer.GetDuration("extraction")

	successfulStreams := countSuccessfulStreams(testResults.Streams)
	fmt.Printf("✅ Combined extraction completed in %.2f seconds\n", testResults.ExtractionTime.Seconds())
	fmt.Printf("   Successful: %d/%d streams\n", successfulStreams, len(urls))

	if successfulStreams != len(urls) {
		fmt.Printf("   ⚠️  Failed streams:\n")
		for _, stream := range testResults.Streams {
			if stream.Error != nil {
				fmt.Printf("     - %s: %v\n", stream.URL, stream.Error)
			}
		}
	}
	fmt.Printf("\n")

	// Check if we have enough successful streams
	if successfulStreams < 2 {
		return fmt.Errorf("❌ Need at least 2 successful streams for alignment, got %d", successfulStreams)
	}

	// PHASE 2: Temporal Alignment
	if ftTestAlignment {
		fmt.Printf("⏰ PHASE 2: Temporal Alignment\n")
		fmt.Printf("==============================\n")

		timer.StartEvent("alignment")

		err = runTemporalAlignment(testResults)
		if err != nil {
			return fmt.Errorf("❌ Temporal alignment failed: %v", err)
		}

		timer.EndEvent("alignment")
		testResults.AlignmentTime = timer.GetDuration("alignment")

		fmt.Printf("✅ Temporal alignment completed in %.2f seconds\n", testResults.AlignmentTime.Seconds())
		fmt.Printf("   Analyzed %d stream pairs\n", len(testResults.AlignmentResults))
		fmt.Printf("\n")
	} else {
		fmt.Printf("⏭️  PHASE 2: Temporal Alignment SKIPPED\n\n")
	}

	// PHASE 3: Stream Comparison
	if ftTestComparison {
		fmt.Printf("🔍 PHASE 3: Stream Comparison\n")
		fmt.Printf("=============================\n")

		timer.StartEvent("comparison")

		err = runStreamComparison(testResults)
		if err != nil {
			return fmt.Errorf("❌ Stream comparison failed: %v", err)
		}

		timer.EndEvent("comparison")
		testResults.ComparisonTime = timer.GetDuration("comparison")

		fmt.Printf("✅ Stream comparison completed in %.2f seconds\n", testResults.ComparisonTime.Seconds())
		fmt.Printf("   Compared %d stream pairs\n", len(testResults.ComparisonResults))
		fmt.Printf("\n")
	} else {
		fmt.Printf("⏭️  PHASE 3: Stream Comparison SKIPPED\n\n")
	}

	// Final Results Summary
	timer.EndEvent("overall")
	testResults.TotalTime = timer.GetDuration("overall")

	fmt.Printf("📊 FINAL RESULTS\n")
	fmt.Printf("================\n")

	printFinalResults(testResults)

	return nil
}

func runCombinedExtraction(ctx context.Context, urls []string, results *FingerprintTestResults) error {
	contentType := parseContentTypeFlag(ftContentType)
	factory := extractors.NewFeatureExtractorFactory()

	// Create content-specific feature extractor
	featureConfig := createFeatureConfig(contentType, 44100) // Default sample rate, will be updated

	// Process each input
	results.Streams = make([]*StreamData, len(urls))

	for i, url := range urls {
		streamID := fmt.Sprintf("stream_%d", i+1)
		fmt.Printf("   Processing %s (%s)...\n", streamID, url)

		stream := &StreamData{
			StreamID: streamID,
			URL:      url,
		}
		results.Streams[i] = stream

		// Load audio data
		audioData, err := loadAudioData(ctx, url)
		if err != nil {
			stream.Error = fmt.Errorf("failed to load audio: %w", err)
			fmt.Printf("   ❌ %s: %v\n", streamID, err)
			continue
		}

		// Store audio data
		stream.PCM = audioData.PCM
		stream.SampleRate = audioData.SampleRate
		stream.Channels = audioData.Channels
		stream.Duration = time.Duration(float64(len(audioData.PCM))/float64(audioData.SampleRate)) * time.Second

		// Update feature config with actual sample rate
		featureConfig.SampleRate = audioData.SampleRate
		extractor, err := factory.CreateExtractor(contentType, *featureConfig)
		if err != nil {
			stream.Error = fmt.Errorf("failed to create extractor: %w", err)
			continue
		}

		// Create spectral analyzer
		analyzer := analyzers.NewSpectralAnalyzer(audioData.SampleRate)

		// Compute spectrogram
		spectrogram, err := analyzer.ComputeSTFTWithWindow(
			audioData.PCM,
			featureConfig.WindowSize,
			featureConfig.HopSize,
			analyzers.WindowHann,
		)
		if err != nil {
			stream.Error = fmt.Errorf("failed to compute spectrogram: %w", err)
			continue
		}

		// Extract features
		features, err := extractor.ExtractFeatures(spectrogram, audioData.PCM, audioData.SampleRate)
		if err != nil {
			stream.Error = fmt.Errorf("failed to extract features: %w", err)
			continue
		}

		stream.Features = features

		if ftVerbose {
			logStreamSummary(stream, extractor)
		} else {
			fmt.Printf("   ✅ %s: %.1fs, %dHz, features extracted\n",
				streamID, stream.Duration.Seconds(), stream.SampleRate)
		}
	}

	return nil
}

func runTemporalAlignment(results *FingerprintTestResults) error {
	fmt.Printf("⚖️  Aligning temporal features between streams...\n")

	successfulStreams := getSuccessfulStreams(results.Streams)
	if len(successfulStreams) < 2 {
		return fmt.Errorf("need at least 2 successful streams for alignment")
	}

	// Create alignment extractor
	contentType := parseContentTypeFlag(ftContentType)
	featureConfig := createFeatureConfig(contentType, 44100)
	alignmentConfig := config.AlignmentConfigForContent(contentType)
	alignmentExtractor := extractors.NewAlignmentExtractorWithMaxLag(featureConfig, alignmentConfig, ftMaxOffsetSec)

	// Compare each pair of streams
	for i := range len(successfulStreams) {
		for j := i + 1; j < len(successfulStreams); j++ {
			stream1 := successfulStreams[i]
			stream2 := successfulStreams[j]
			pairID := fmt.Sprintf("%s_vs_%s", stream1.StreamID, stream2.StreamID)

			fmt.Printf("   Aligning %s <-> %s...\n", stream1.StreamID, stream2.StreamID)

			// Perform alignment using new API
			alignmentFeatures, err := alignmentExtractor.ExtractAlignmentFeatures(
				stream1.Features, stream2.Features,
				stream1.PCM, stream2.PCM,
				stream1.SampleRate)

			if err != nil {
				fmt.Printf("   ⚠️  %s: alignment failed - %v\n", pairID, err)
				continue
			}

			results.AlignmentResults[pairID] = alignmentFeatures

			// Determine alignment validity
			isValid := alignmentFeatures.OffsetConfidence >= ftMinConfidence &&
				alignmentFeatures.OverallSimilarity > 0.3 &&
				math.Abs(alignmentFeatures.TemporalOffset) <= ftMaxOffsetSec

			status := map[bool]string{true: "✅", false: "❌"}[isValid]

			fmt.Printf("   %s %s: offset=%.2fs, confidence=%.3f, similarity=%.3f, method=%s\n",
				status, pairID, alignmentFeatures.TemporalOffset,
				alignmentFeatures.OffsetConfidence, alignmentFeatures.OverallSimilarity,
				alignmentFeatures.Method)

			if ftVerbose && isValid {
				logAlignmentDetails(pairID, alignmentFeatures)
			}
		}
	}

	return nil
}

func runStreamComparison(results *FingerprintTestResults) error {
	fmt.Printf("🎯 Comparing stream fingerprints...\n")

	successfulStreams := getSuccessfulStreams(results.Streams)

	// Compare each pair of streams
	for i := range len(successfulStreams) {
		for j := i + 1; j < len(successfulStreams); j++ {
			stream1 := successfulStreams[i]
			stream2 := successfulStreams[j]
			pairID := fmt.Sprintf("%s_vs_%s", stream1.StreamID, stream2.StreamID)

			fmt.Printf("   Comparing %s <-> %s...\n", stream1.StreamID, stream2.StreamID)

			// Calculate similarity score
			similarity := calculateStreamSimilarity(stream1.Features, stream2.Features)
			results.ComparisonResults[pairID] = similarity

			// Determine match status
			var status, confidence string
			if similarity > 0.8 {
				status = "✅ Match"
				confidence = "High"
			} else if similarity > 0.6 {
				status = "⚠️  Similar"
				confidence = "Medium"
			} else {
				status = "❌ Different"
				confidence = "Low"
			}

			fmt.Printf("   %s %s: similarity=%.3f (%s confidence)\n",
				status, pairID, similarity, confidence)

			if ftVerbose {
				logDetailedComparison(pairID, stream1.Features, stream2.Features, similarity)
			}
		}
	}

	return nil
}

func printFinalResults(results *FingerprintTestResults) {
	fmt.Printf("⏱️  Performance Summary:\n")
	fmt.Printf("   Combined Extraction: %.2f seconds\n", results.ExtractionTime.Seconds())
	if results.AlignmentTime > 0 {
		fmt.Printf("   Temporal Alignment: %.2f seconds\n", results.AlignmentTime.Seconds())
	}
	if results.ComparisonTime > 0 {
		fmt.Printf("   Stream Comparison: %.2f seconds\n", results.ComparisonTime.Seconds())
	}
	fmt.Printf("   Total Pipeline: %.2f seconds\n", results.TotalTime.Seconds())
	fmt.Printf("\n")

	// Alignment summary
	if len(results.AlignmentResults) > 0 {
		validAlignments := 0
		avgConfidence := 0.0
		avgOffset := 0.0

		for _, alignment := range results.AlignmentResults {
			isValid := alignment.OffsetConfidence >= ftMinConfidence &&
				alignment.OverallSimilarity > 0.3 &&
				math.Abs(alignment.TemporalOffset) <= ftMaxOffsetSec

			if isValid {
				validAlignments++
				avgConfidence += alignment.OffsetConfidence
				avgOffset += math.Abs(alignment.TemporalOffset)
			}
		}

		if validAlignments > 0 {
			avgConfidence /= float64(validAlignments)
			avgOffset /= float64(validAlignments)
		}

		fmt.Printf("🎯 Alignment Summary:\n")
		fmt.Printf("   Valid Alignments: %d/%d\n", validAlignments, len(results.AlignmentResults))
		if validAlignments > 0 {
			fmt.Printf("   Average Confidence: %.3f\n", avgConfidence)
			fmt.Printf("   Average Offset: %.2f seconds\n", avgOffset)
		}
		fmt.Printf("\n")
	}

	// Comparison summary
	if len(results.ComparisonResults) > 0 {
		matches := 0
		similar := 0
		avgSimilarity := 0.0

		for _, similarity := range results.ComparisonResults {
			avgSimilarity += similarity
			if similarity > 0.8 {
				matches++
			} else if similarity > 0.6 {
				similar++
			}
		}

		avgSimilarity /= float64(len(results.ComparisonResults))

		fmt.Printf("🔍 Comparison Summary:\n")
		fmt.Printf("   Strong Matches: %d/%d\n", matches, len(results.ComparisonResults))
		fmt.Printf("   Partial Matches: %d/%d\n", similar, len(results.ComparisonResults))
		fmt.Printf("   Average Similarity: %.3f\n", avgSimilarity)
		fmt.Printf("\n")
	}

	// Recommendations
	printRecommendations(results)
}

// Helper functions

func loadAudioData(ctx context.Context, input string) (*common.AudioData, error) {
	if isLocalFile(input) {
		return loadLocalFile(input)
	} else {
		return loadStreamURL(ctx, input)
	}
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
		return commonAudio, nil
	}

	// Convert if needed
	audioData := common.ConvertToAudioData(anyData)
	if audioData == nil {
		return nil, fmt.Errorf("decoder returned unexpected type: %T", anyData)
	}

	return audioData, nil
}

func loadStreamURL(ctx context.Context, url string) (*common.AudioData, error) {
	// Create a simple stream manager for single URL
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

func countSuccessfulStreams(streams []*StreamData) int {
	count := 0
	for _, stream := range streams {
		if stream.Error == nil && stream.Features != nil {
			count++
		}
	}
	return count
}

func getSuccessfulStreams(streams []*StreamData) []*StreamData {
	var successful []*StreamData
	for _, stream := range streams {
		if stream.Error == nil && stream.Features != nil {
			successful = append(successful, stream)
		}
	}
	return successful
}

func logStreamSummary(stream *StreamData, extractor extractors.FeatureExtractor) {
	fmt.Printf("      %s features extracted:\n", extractor.GetName())
	fmt.Printf("        Content Type: %s\n", extractor.GetContentType())
	fmt.Printf("        Duration: %.1fs, Sample Rate: %dHz\n", stream.Duration.Seconds(), stream.SampleRate)

	features := stream.Features
	if features.SpectralFeatures != nil {
		fmt.Printf("        Spectral: %d frames\n", len(features.SpectralFeatures.SpectralCentroid))
	}
	if features.TemporalFeatures != nil {
		fmt.Printf("        Temporal: RMS=%d, Dynamics=%.2f\n",
			len(features.TemporalFeatures.RMSEnergy), features.TemporalFeatures.DynamicRange)
	}
	if len(features.MFCC) > 0 {
		fmt.Printf("        MFCC: %dx%d coefficients\n", len(features.MFCC), len(features.MFCC[0]))
	}
}

func logAlignmentDetails(pairID string, features *extractors.AlignmentFeatures) {
	fmt.Printf("      Detailed alignment for %s:\n", pairID)
	fmt.Printf("        Method: %s, Quality: %.3f\n", features.Method, features.AlignmentQuality)
	fmt.Printf("        Time stretch factor: %.3f\n", features.TimeStretch)

	if features.Consistency != nil {
		fmt.Printf("        Consistency: %.3f (stddev: %.3f)\n",
			features.Consistency.Consistency, features.Consistency.StdDevOffset)
	}
}

func logDetailedComparison(pairID string, features1, features2 *extractors.ExtractedFeatures, similarity float64) {
	fmt.Printf("      Detailed comparison for %s:\n", pairID)
	fmt.Printf("        Spectral similarity: %.3f\n", similarity)

	if features1.EnergyFeatures != nil && features2.EnergyFeatures != nil {
		energyDiff := math.Abs(features1.EnergyFeatures.EnergyVariance - features2.EnergyFeatures.EnergyVariance)
		fmt.Printf("        Energy variance diff: %.3f\n", energyDiff)
	}

	if features1.TemporalFeatures != nil && features2.TemporalFeatures != nil {
		dynamicDiff := math.Abs(features1.TemporalFeatures.DynamicRange - features2.TemporalFeatures.DynamicRange)
		fmt.Printf("        Dynamic range diff: %.3f\n", dynamicDiff)
	}
}

func printRecommendations(results *FingerprintTestResults) {
	fmt.Printf("💡 Recommendations:\n")

	failedStreams := len(results.Streams) - countSuccessfulStreams(results.Streams)
	if failedStreams > 0 {
		fmt.Printf("   • Fix %d failed stream(s) for better analysis\n", failedStreams)
	}

	validAlignments := 0
	for _, alignment := range results.AlignmentResults {
		isValid := alignment.OffsetConfidence >= ftMinConfidence &&
			alignment.OverallSimilarity > 0.3 &&
			math.Abs(alignment.TemporalOffset) <= ftMaxOffsetSec
		if isValid {
			validAlignments++
		}
	}

	if validAlignments == 0 && len(results.AlignmentResults) > 0 {
		fmt.Printf("   • No valid alignments found - check if streams are from same source\n")
	} else if validAlignments > 0 {
		fmt.Printf("   • %d valid alignment(s) found - streams appear synchronized\n", validAlignments)
	}

	strongMatches := 0
	for _, similarity := range results.ComparisonResults {
		if similarity > 0.8 {
			strongMatches++
		}
	}

	if strongMatches > 0 {
		fmt.Printf("   • %d strong match(es) detected - streams likely from same source!\n", strongMatches)
	} else if len(results.ComparisonResults) > 0 {
		fmt.Printf("   • No strong matches - streams may be from different sources\n")
	}
}

// Shared helper functions from original file
func enabledStatus(enabled bool) string {
	if enabled {
		return "✅ Enabled"
	}
	return "⏭️  Disabled"
}

func calculateStreamSimilarity(features1, features2 *extractors.ExtractedFeatures) float64 {
	if features1.SpectralFeatures == nil || features2.SpectralFeatures == nil {
		return 0.0
	}

	centroid1 := features1.SpectralFeatures.SpectralCentroid
	centroid2 := features2.SpectralFeatures.SpectralCentroid

	minLen := len(centroid1)
	minLen = min(minLen, len(centroid2))

	if minLen == 0 {
		return 0.0
	}

	var sum1, sum2, sum1Sq, sum2Sq, sumProduct float64

	for i := 0; i < minLen; i++ {
		sum1 += centroid1[i]
		sum2 += centroid2[i]
		sum1Sq += centroid1[i] * centroid1[i]
		sum2Sq += centroid2[i] * centroid2[i]
		sumProduct += centroid1[i] * centroid2[i]
	}

	n := float64(minLen)
	numerator := n*sumProduct - sum1*sum2
	denominator := (n*sum1Sq - sum1*sum1) * (n*sum2Sq - sum2*sum2)

	if denominator <= 0 {
		return 0.0
	}

	correlation := numerator / math.Sqrt(denominator)
	return (correlation + 1.0) / 2.0
}

func createFeatureConfig(contentType config.ContentType, sampleRate int) *config.FeatureConfig {
	featureConfig := &config.FeatureConfig{
		WindowSize: 2048,
		HopSize:    512,
		SampleRate: sampleRate,
		FreqRange:  [2]float64{20.0, float64(sampleRate / 2)},

		EnableChroma:           false,
		EnableMFCC:             true,
		EnableSpectralContrast: false,
		EnableTemporalFeatures: true,
		EnableSpeechFeatures:   false,
		EnableHarmonicFeatures: false,

		MFCCCoefficients: 13,
		ChromaBins:       12,
		ContrastBands:    6,

		SimilarityWeights: map[string]float64{
			"spectral": 1.0,
			"temporal": 0.8,
			"energy":   0.6,
		},
		MatchThreshold: 0.75,
	}

	// Content-specific optimizations
	switch contentType {
	case config.ContentMusic:
		featureConfig.EnableChroma = true
		featureConfig.EnableHarmonicFeatures = true
		featureConfig.EnableSpectralContrast = true
		featureConfig.MatchThreshold = 0.80
	case config.ContentNews, config.ContentTalk:
		featureConfig.WindowSize = 1024
		featureConfig.HopSize = 256
		featureConfig.EnableSpeechFeatures = true
		featureConfig.MFCCCoefficients = 12
		featureConfig.FreqRange = [2]float64{80.0, 8000.0}
		featureConfig.MatchThreshold = 0.70
	case config.ContentSports:
		featureConfig.EnableTemporalFeatures = true
		featureConfig.EnableSpectralContrast = true
		featureConfig.WindowSize = 1024
		featureConfig.HopSize = 256
		featureConfig.MatchThreshold = 0.75
	case config.ContentMixed:
		featureConfig.EnableChroma = true
		featureConfig.EnableMFCC = true
		featureConfig.EnableSpectralContrast = true
		featureConfig.EnableTemporalFeatures = true
		featureConfig.EnableSpeechFeatures = true
		featureConfig.EnableHarmonicFeatures = true
		featureConfig.MatchThreshold = 0.72
	}

	return featureConfig
}

func parseContentTypeFlag(contentTypeStr string) config.ContentType {
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
	case "general", "unknown":
		return config.ContentUnknown
	default:
		fmt.Printf("   ⚠️  Unknown content type '%s', using music\n", contentTypeStr)
		return config.ContentMusic
	}
}

func isLocalFile(input string) bool {
	return strings.HasPrefix(input, "file://") ||
		strings.HasPrefix(input, "/") ||
		strings.HasPrefix(input, "./") ||
		strings.HasPrefix(input, "../") ||
		(!strings.HasPrefix(input, "http://") && !strings.HasPrefix(input, "https://"))
}
