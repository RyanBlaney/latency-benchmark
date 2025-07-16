package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	// "github.com/spf13/viper"
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

type FingerprintTestResults struct {
	AudioResults      *stream.ParallelExtractionResult
	FeatureResults    map[string]*extractors.ExtractedFeatures
	AlignmentResults  map[string]*extractors.AlignmentResult
	ComparisonResults map[string]float64

	ExtractionTime time.Duration
	AlignmentTime  time.Duration
	ComparisonTime time.Duration
	TotalTime      time.Duration
}

/* func runFingerprintTest(cmd *cobra.Command, args []string) error {
	urls := args
	verbose := ftVerbose || viper.GetBool("verbose")

	fmt.Printf("🎵 FINGERPRINT PIPELINE TEST\n")
	fmt.Printf("============================\n\n")

	// Configuration summary
	fmt.Printf("⚙️  Configuration:\n")
	fmt.Printf("   Mode: %s\n", map[bool]string{true: "Parallel", false: "Sequential"}[ftParallel])
	fmt.Printf("   Streams: %d\n", len(urls))
	fmt.Printf("   Duration: %.1f seconds\n", ftSegmentDuration.Seconds())
	fmt.Printf("   Content Type: %s\n", ftContentType)
	fmt.Printf("   Feature Rate: %.1f Hz\n", ftFeatureRate)
	fmt.Printf("   Max Offset: %.1f seconds\n", ftMaxOffsetSec)

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

	var err error

	// Load configuration
	// fmt.Printf("⚙️  Loading configuration...\n")
	// appConfig, err := configs.LoadConfig()
	// if err != nil {
	// return fmt.Errorf("❌ Failed to load config: %v", err)
	// }
	// fmt.Printf("✅ Configuration loaded\n\n")

	// Create Stream Manager
	fmt.Printf("🏗️  Setting up Stream Manager...\n")
	managerConfig := &stream.ManagerConfig{
		StreamTimeout:        ftTimeout,
		OverallTimeout:       ftTimeout + (10 * time.Second),
		MaxConcurrentStreams: ftMaxConcurrent,
		ResultBufferSize:     len(urls),
	}
	manager := stream.NewManagerWithConfig(managerConfig)
	fmt.Printf("✅ Stream Manager configured\n\n")

	// Initialize test results
	testResults := &FingerprintTestResults{
		FeatureResults:    make(map[string]*extractors.ExtractedFeatures),
		AlignmentResults:  make(map[string]*extractors.AlignmentResult),
		ComparisonResults: make(map[string]float64),
	}

	// PHASE 1: Audio Extraction
	fmt.Printf("🎵 PHASE 1: Audio Extraction\n")
	fmt.Printf("=============================\n")

	timer.StartEvent("audio_extraction")
	fmt.Printf("📡 Extracting audio from %d streams...\n", len(urls))

	var audioResults *stream.ParallelExtractionResult
	if ftParallel {
		audioResults, err = manager.ExtractAudioParallel(ctx, urls, ftSegmentDuration)
	} else {
		audioResults, err = manager.ExtractAudioSequential(ctx, urls, ftSegmentDuration)
	}

	timer.EndEvent("audio_extraction")
	testResults.ExtractionTime = timer.GetDuration("audio_extraction")
	testResults.AudioResults = audioResults

	if err != nil {
		return fmt.Errorf("❌ Audio extraction failed: %v", err)
	}

	fmt.Printf("✅ Audio extraction completed in %.2f seconds\n", testResults.ExtractionTime.Seconds())
	fmt.Printf("   Successful: %d/%d streams\n", audioResults.SuccessfulStreams, len(urls))

	if audioResults.FailedStreams > 0 {
		fmt.Printf("   ⚠️  Failed: %d streams\n", audioResults.FailedStreams)
	}
	fmt.Printf("\n")

	// Check if we have enough successful streams
	if audioResults.SuccessfulStreams < 2 {
		return fmt.Errorf("❌ Need at least 2 successful streams for fingerprinting, got %d", audioResults.SuccessfulStreams)
	}

	// PHASE 2: Feature Extraction
	if ftTestExtraction {
		fmt.Printf("🧬 PHASE 2: Feature Extraction\n")
		fmt.Printf("==============================\n")

		timer.StartEvent("feature_extraction")

		err = runFeatureExtraction(testResults, verbose)
		if err != nil {
			return fmt.Errorf("❌ Feature extraction failed: %v", err)
		}

		timer.EndEvent("feature_extraction")
		testResults.ExtractionTime = timer.GetDuration("feature_extraction")

		fmt.Printf("✅ Feature extraction completed in %.2f seconds\n", testResults.ExtractionTime.Seconds())
		fmt.Printf("   Extracted features from %d streams\n", len(testResults.FeatureResults))
		fmt.Printf("\n")
	} else {
		fmt.Printf("⏭️  PHASE 2: Feature Extraction SKIPPED\n\n")
	}

	// PHASE 3: Temporal Alignment
	if ftTestAlignment && ftTestExtraction {
		fmt.Printf("⏰ PHASE 3: Temporal Alignment\n")
		fmt.Printf("==============================\n")

		timer.StartEvent("alignment")

		err = runTemporalAlignment(testResults, verbose)
		if err != nil {
			return fmt.Errorf("❌ Temporal alignment failed: %v", err)
		}

		timer.EndEvent("alignment")
		testResults.AlignmentTime = timer.GetDuration("alignment")

		fmt.Printf("✅ Temporal alignment completed in %.2f seconds\n", testResults.AlignmentTime.Seconds())
		fmt.Printf("   Analyzed %d stream pairs\n", len(testResults.AlignmentResults))
		fmt.Printf("\n")
	} else {
		fmt.Printf("⏭️  PHASE 3: Temporal Alignment SKIPPED\n\n")
	}

	// PHASE 4: Stream Comparison
	if ftTestComparison && ftTestExtraction {
		fmt.Printf("🔍 PHASE 4: Stream Comparison\n")
		fmt.Printf("=============================\n")

		timer.StartEvent("comparison")

		err = runStreamComparison(testResults, verbose)
		if err != nil {
			return fmt.Errorf("❌ Stream comparison failed: %v", err)
		}

		timer.EndEvent("comparison")
		testResults.ComparisonTime = timer.GetDuration("comparison")

		fmt.Printf("✅ Stream comparison completed in %.2f seconds\n", testResults.ComparisonTime.Seconds())
		fmt.Printf("   Compared %d stream pairs\n", len(testResults.ComparisonResults))
		fmt.Printf("\n")
	} else {
		fmt.Printf("⏭️  PHASE 4: Stream Comparison SKIPPED\n\n")
	}

	// Final Results Summary
	timer.EndEvent("overall")
	testResults.TotalTime = timer.GetDuration("overall")

	fmt.Printf("📊 FINAL RESULTS\n")
	fmt.Printf("================\n")

	printFinalResults(testResults, verbose)

	return nil
} */

func runFingerprintTest(cmd *cobra.Command, args []string) error {
	urls := args
	// verbose := ftVerbose || viper.GetBool("verbose")

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

	var err error

	// Create Stream Manager (only needed for live streams)
	var manager *stream.Manager
	if streamUrls > 0 {
		fmt.Printf("🏗️  Setting up Stream Manager...\n")
		managerConfig := &stream.ManagerConfig{
			StreamTimeout:        ftTimeout,
			OverallTimeout:       ftTimeout + (10 * time.Second),
			MaxConcurrentStreams: ftMaxConcurrent,
			ResultBufferSize:     len(urls),
		}
		manager = stream.NewManagerWithConfig(managerConfig)
		fmt.Printf("✅ Stream Manager configured\n\n")
	}

	// Initialize test results
	testResults := &FingerprintTestResults{
		FeatureResults:    make(map[string]*extractors.ExtractedFeatures),
		AlignmentResults:  make(map[string]*extractors.AlignmentResult),
		ComparisonResults: make(map[string]float64),
	}

	// PHASE 1: Audio Extraction
	fmt.Printf("🎵 PHASE 1: Audio Extraction\n")
	fmt.Printf("=============================\n")

	timer.StartEvent("audio_extraction")

	var audioResults *stream.ParallelExtractionResult

	if localFiles == len(urls) {
		// All inputs are local files
		fmt.Printf("📁 Extracting audio from %d local files...\n", len(urls))
		audioResults, err = extractAudioFromLocalFiles(ctx, urls, ftSegmentDuration)
	} else if streamUrls == len(urls) {
		// All inputs are streams
		fmt.Printf("📡 Extracting audio from %d streams...\n", len(urls))
		if ftParallel {
			audioResults, err = manager.ExtractAudioParallel(ctx, urls, ftSegmentDuration)
		} else {
			audioResults, err = manager.ExtractAudioSequential(ctx, urls, ftSegmentDuration)
		}
	} else {
		// Mixed inputs - handle separately
		fmt.Printf("📋 Extracting audio from %d mixed inputs...\n", len(urls))
		audioResults, err = extractMixedInputs(ctx, urls, ftSegmentDuration, manager)
	}

	timer.EndEvent("audio_extraction")
	testResults.ExtractionTime = timer.GetDuration("audio_extraction")
	testResults.AudioResults = audioResults

	if err != nil {
		return fmt.Errorf("❌ Audio extraction failed: %v", err)
	}

	fmt.Printf("✅ Audio extraction completed in %.2f seconds\n", testResults.ExtractionTime.Seconds())
	fmt.Printf("   Successful: %d/%d streams\n", audioResults.SuccessfulStreams, len(urls))

	if audioResults.FailedStreams > 0 {
		fmt.Printf("   ⚠️  Failed: %d streams\n", audioResults.FailedStreams)
		// Show which ones failed
		for i, result := range audioResults.Results {
			if result.Error != nil {
				fmt.Printf("     - %s: %v\n", urls[i], result.Error)
			}
		}
	}
	fmt.Printf("\n")

	// Check if we have enough successful streams
	if audioResults.SuccessfulStreams < 2 {
		return fmt.Errorf("❌ Need at least 2 successful streams for fingerprinting, got %d", audioResults.SuccessfulStreams)
	}

	// PHASE 2: Feature Extraction
	if ftTestExtraction {
		fmt.Printf("🧬 PHASE 2: Feature Extraction\n")
		fmt.Printf("==============================\n")

		timer.StartEvent("feature_extraction")

		err = runFeatureExtraction(testResults, verbose)
		if err != nil {
			return fmt.Errorf("❌ Feature extraction failed: %v", err)
		}

		timer.EndEvent("feature_extraction")
		testResults.ExtractionTime = timer.GetDuration("feature_extraction")

		fmt.Printf("✅ Feature extraction completed in %.2f seconds\n", testResults.ExtractionTime.Seconds())
		fmt.Printf("   Extracted features from %d streams\n", len(testResults.FeatureResults))
		fmt.Printf("\n")
	} else {
		fmt.Printf("⏭️  PHASE 2: Feature Extraction SKIPPED\n\n")
	}

	// PHASE 3: Temporal Alignment
	if ftTestAlignment && ftTestExtraction {
		fmt.Printf("⏰ PHASE 3: Temporal Alignment\n")
		fmt.Printf("==============================\n")

		timer.StartEvent("alignment")

		err = runTemporalAlignment(testResults, verbose)
		if err != nil {
			return fmt.Errorf("❌ Temporal alignment failed: %v", err)
		}

		timer.EndEvent("alignment")
		testResults.AlignmentTime = timer.GetDuration("alignment")

		fmt.Printf("✅ Temporal alignment completed in %.2f seconds\n", testResults.AlignmentTime.Seconds())
		fmt.Printf("   Analyzed %d stream pairs\n", len(testResults.AlignmentResults))
		fmt.Printf("\n")
	} else {
		fmt.Printf("⏭️  PHASE 3: Temporal Alignment SKIPPED\n\n")
	}

	// PHASE 4: Stream Comparison
	if ftTestComparison && ftTestExtraction {
		fmt.Printf("🔍 PHASE 4: Stream Comparison\n")
		fmt.Printf("=============================\n")

		timer.StartEvent("comparison")

		err = runStreamComparison(testResults, verbose)
		if err != nil {
			return fmt.Errorf("❌ Stream comparison failed: %v", err)
		}

		timer.EndEvent("comparison")
		testResults.ComparisonTime = timer.GetDuration("comparison")

		fmt.Printf("✅ Stream comparison completed in %.2f seconds\n", testResults.ComparisonTime.Seconds())
		fmt.Printf("   Compared %d stream pairs\n", len(testResults.ComparisonResults))
		fmt.Printf("\n")
	} else {
		fmt.Printf("⏭️  PHASE 4: Stream Comparison SKIPPED\n\n")
	}

	// Final Results Summary
	timer.EndEvent("overall")
	testResults.TotalTime = timer.GetDuration("overall")

	fmt.Printf("📊 FINAL RESULTS\n")
	fmt.Printf("================\n")

	printFinalResults(testResults, verbose)

	return nil
}

// Handle mixed local files and stream URLs
func extractMixedInputs(ctx context.Context, inputs []string, segmentDuration time.Duration, manager *stream.Manager) (*stream.ParallelExtractionResult, error) {
	results := make([]*stream.AudioExtractionResult, len(inputs))
	successfulStreams := 0
	failedStreams := 0

	for i, input := range inputs {
		if isLocalFile(input) {
			// Handle as local file
			cleanPath := strings.TrimPrefix(input, "file://")
			fileData, err := os.ReadFile(cleanPath)
			if err != nil {
				results[i] = &stream.AudioExtractionResult{
					URL:   input,
					Error: fmt.Errorf("failed to read file: %w", err),
				}
				failedStreams++
				continue
			}

			decoder := transcode.NewNormalizingDecoder(ftContentType)

			anyData, err := decoder.DecodeBytes(fileData)
			if err != nil {
				results[i] = &stream.AudioExtractionResult{
					URL:   input,
					Error: fmt.Errorf("failed to decode audio: %w", err),
				}
				failedStreams++
				continue
			}

			var audioData *common.AudioData
			if commonAudio, ok := anyData.(*common.AudioData); ok {
				audioData = commonAudio
			} else {
				// Use reflection to convert unknown AudioData type to common.AudioData
				audioData = common.ConvertToAudioData(anyData)
				if audioData == nil {
					return nil, fmt.Errorf("decoder returned unexpected type: %T", anyData)
				}
			}

			// Truncate if needed
			if segmentDuration > 0 {
				maxSamples := int(segmentDuration.Seconds() * float64(audioData.SampleRate))
				if len(audioData.PCM) > maxSamples {
					audioData.PCM = audioData.PCM[:maxSamples]
				}
			}

			results[i] = &stream.AudioExtractionResult{
				URL:       input,
				AudioData: audioData,
				Error:     nil,
			}
			successfulStreams++
		} else {
			// Handle as stream URL
			streamResults, err := manager.ExtractAudioSequential(ctx, []string{input}, segmentDuration)
			if err != nil || len(streamResults.Results) == 0 {
				results[i] = &stream.AudioExtractionResult{
					URL:   input,
					Error: fmt.Errorf("failed to extract from stream: %w", err),
				}
				failedStreams++
			} else {
				results[i] = streamResults.Results[0]
				if results[i].Error == nil {
					successfulStreams++
				} else {
					failedStreams++
				}
			}
		}
	}

	return &stream.ParallelExtractionResult{
		Results:           results,
		SuccessfulStreams: successfulStreams,
		FailedStreams:     failedStreams,
	}, nil
}

func runFeatureExtraction(results *FingerprintTestResults, verbose bool) error {
	fmt.Printf("🔬 Extracting features from audio streams...\n")

	successfulStreams := getSuccessfulStreams(results.AudioResults)

	// Parse content type
	contentType := parseContentTypeFlag(ftContentType)
	fmt.Printf("   Using content type: %s\n", contentType)

	// Create feature extractor factory
	factory := extractors.NewFeatureExtractorFactory()

	for i, result := range successfulStreams {
		streamID := fmt.Sprintf("stream_%d", i+1)
		fmt.Printf("   Processing %s (%s)...\n", streamID, result.URL)

		// Create content-optimized feature config
		featureConfig := createFeatureConfig(contentType, result.AudioData.SampleRate)

		// Create content-specific feature extractor
		extractor, err := factory.CreateExtractor(contentType, *featureConfig)
		if err != nil {
			return fmt.Errorf("failed to create extractor for %s: %w", streamID, err)
		}

		// Create spectral analyzer for this stream
		analyzer := analyzers.NewSpectralAnalyzer(result.AudioData.SampleRate)

		windowSize := featureConfig.WindowSize
		hopSize := featureConfig.HopSize
		windowType := analyzers.WindowHann

		// Compute spectrogram
		spectrogram, err := analyzer.ComputeSTFTWithWindow(
			result.AudioData.PCM,
			windowSize,
			hopSize,
			windowType,
		)
		if err != nil {
			return fmt.Errorf("failed to compute spectrogram for %s: %w", streamID, err)
		}

		// Extract features using content-specific extractor
		features, err := extractor.ExtractFeatures(spectrogram, result.AudioData.PCM, result.AudioData.SampleRate)
		if err != nil {
			return fmt.Errorf("failed to extract features from %s: %w", streamID, err)
		}

		results.FeatureResults[streamID] = features

		// Log feature summary
		if verbose {
			logFeatureSummary(streamID, features, extractor, featureConfig)
		} else {
			fmt.Printf("   ✅ %s: %s extractor, features extracted\n",
				streamID, extractor.GetName())
		}
	}

	return nil
}

func runTemporalAlignment(results *FingerprintTestResults, verbose bool) error {
	fmt.Printf("⚖️  Aligning temporal features between streams...\n")

	streamIDs := getStreamIDs(results.FeatureResults)

	// Compare each pair of streams
	for i := range len(streamIDs) {
		for j := i + 1; j < len(streamIDs); j++ {
			stream1ID := streamIDs[i]
			stream2ID := streamIDs[j]
			pairID := fmt.Sprintf("%s_vs_%s", stream1ID, stream2ID)

			fmt.Printf("   Aligning %s <-> %s...\n", stream1ID, stream2ID)

			features1 := results.FeatureResults[stream1ID]
			features2 := results.FeatureResults[stream2ID]

			// Perform alignment
			alignment, err := extractors.AlignStreams(features1, features2, ftMaxOffsetSec, ftFeatureRate)
			if err != nil {
				fmt.Printf("   ⚠️  %s: alignment failed - %v\n", pairID, err)
				continue
			}

			results.AlignmentResults[pairID] = alignment

			// Log alignment result
			status := "❌"
			if alignment.IsValid {
				status = "✅"
			}

			fmt.Printf("   %s %s: offset=%.2fs, confidence=%.3f, valid=%t\n",
				status, pairID, alignment.OffsetSeconds, alignment.Confidence, alignment.IsValid)

			if verbose && alignment.IsValid {
				fmt.Printf("      Offset range: ±%.1fs, minimum confidence: %.2f\n",
					ftMaxOffsetSec, ftMinConfidence)
			}
		}
	}

	return nil
}

func runStreamComparison(results *FingerprintTestResults, verbose bool) error {
	fmt.Printf("🎯 Comparing stream fingerprints...\n")

	streamIDs := getStreamIDs(results.FeatureResults)

	// Compare each pair of streams
	for i := 0; i < len(streamIDs); i++ {
		for j := i + 1; j < len(streamIDs); j++ {
			stream1ID := streamIDs[i]
			stream2ID := streamIDs[j]
			pairID := fmt.Sprintf("%s_vs_%s", stream1ID, stream2ID)

			fmt.Printf("   Comparing %s <-> %s...\n", stream1ID, stream2ID)

			features1 := results.FeatureResults[stream1ID]
			features2 := results.FeatureResults[stream2ID]

			// Calculate similarity score
			similarity := calculateStreamSimilarity(features1, features2)
			results.ComparisonResults[pairID] = similarity

			// Determine match status
			status := "❌ Different"
			confidence := "Low"
			if similarity > 0.8 {
				status = "✅ Match"
				confidence = "High"
			} else if similarity > 0.6 {
				status = "⚠️  Similar"
				confidence = "Medium"
			}

			fmt.Printf("   %s %s: similarity=%.3f (%s confidence)\n",
				status, pairID, similarity, confidence)

			if verbose {
				logDetailedComparison(pairID, features1, features2, similarity)
			}
		}
	}

	return nil
}

func printFinalResults(results *FingerprintTestResults, verbose bool) {
	fmt.Printf("⏱️  Performance Summary:\n")
	fmt.Printf("   Audio Extraction: %.2f seconds\n", results.ExtractionTime.Seconds())
	if results.ExtractionTime > 0 {
		fmt.Printf("   Feature Extraction: %.2f seconds\n", results.ExtractionTime.Seconds())
	}
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
			if alignment.IsValid {
				validAlignments++
				avgConfidence += alignment.Confidence
				avgOffset += abs(alignment.OffsetSeconds)
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
	fmt.Printf("💡 Recommendations:\n")

	if results.AudioResults.FailedStreams > 0 {
		fmt.Printf("   • Fix %d failed stream(s) for better analysis\n", results.AudioResults.FailedStreams)
	}

	validAlignments := countValidAlignments(results.AlignmentResults)
	if validAlignments == 0 && len(results.AlignmentResults) > 0 {
		fmt.Printf("   • No valid alignments found - check if streams are from same source\n")
	} else if validAlignments > 0 {
		fmt.Printf("   • %d valid alignment(s) found - streams appear synchronized\n", validAlignments)
	}

	strongMatches := countStrongMatches(results.ComparisonResults)
	if strongMatches > 0 {
		fmt.Printf("   • %d strong match(es) detected - streams likely from same source!\n", strongMatches)
	} else if len(results.ComparisonResults) > 0 {
		fmt.Printf("   • No strong matches - streams may be from different sources\n")
	}

	if verbose && len(results.FeatureResults) > 0 {
		fmt.Printf("\n📋 Detailed Feature Data:\n")
		for streamID, features := range results.FeatureResults {
			fmt.Printf("   %s: spectral=%d, temporal=%d, energy=%d, MFCC=%dx%d\n",
				streamID,
				len(features.SpectralFeatures.SpectralCentroid),
				len(features.TemporalFeatures.RMSEnergy),
				len(features.EnergyFeatures.ShortTimeEnergy),
				len(features.MFCC), len(features.MFCC[0]))
		}
	}
}

// Helper functions

func enabledStatus(enabled bool) string {
	if enabled {
		return "✅ Enabled"
	}
	return "⏭️  Disabled"
}

func getSuccessfulStreams(results *stream.ParallelExtractionResult) []*stream.AudioExtractionResult {
	var successful []*stream.AudioExtractionResult
	for _, result := range results.Results {
		if result.Error == nil && result.AudioData != nil {
			successful = append(successful, result)
		}
	}
	return successful
}

func getStreamIDs(features map[string]*extractors.ExtractedFeatures) []string {
	var ids []string
	for id := range features {
		ids = append(ids, id)
	}
	return ids
}

func logFeatureSummary(streamID string, features *extractors.ExtractedFeatures, extractor extractors.FeatureExtractor, featureConfig *config.FeatureConfig) {
	fmt.Printf("      %s features extracted:\n", extractor.GetName())
	fmt.Printf("        Content Type: %s\n", extractor.GetContentType())
	fmt.Printf("        Window Size: %d, Hop Size: %d\n", featureConfig.WindowSize, featureConfig.HopSize)

	if features.SpectralFeatures != nil {
		fmt.Printf("        Spectral: %d frames\n", len(features.SpectralFeatures.SpectralCentroid))
	}
	if features.TemporalFeatures != nil {
		fmt.Printf("        Temporal: RMS=%d, Dynamics=%.2f\n",
			len(features.TemporalFeatures.RMSEnergy), features.TemporalFeatures.DynamicRange)
	}
	if features.EnergyFeatures != nil {
		fmt.Printf("        Energy: %d frames, variance=%.2f\n",
			len(features.EnergyFeatures.ShortTimeEnergy), features.EnergyFeatures.EnergyVariance)
	}
	if features.MFCC != nil && len(features.MFCC) > 0 {
		fmt.Printf("        MFCC: %dx%d coefficients\n", len(features.MFCC), len(features.MFCC[0]))
	}
	if features.ChromaFeatures != nil && len(features.ChromaFeatures) > 0 {
		fmt.Printf("        Chroma: %dx%d features\n", len(features.ChromaFeatures), len(features.ChromaFeatures[0]))
	}

	// Show enabled features
	fmt.Printf("        Enabled Features: ")
	if featureConfig.EnableMFCC {
		fmt.Printf("MFCC ")
	}
	if featureConfig.EnableChroma {
		fmt.Printf("Chroma ")
	}
	if featureConfig.EnableSpectralContrast {
		fmt.Printf("SpectralContrast ")
	}
	if featureConfig.EnableTemporalFeatures {
		fmt.Printf("Temporal ")
	}
	if featureConfig.EnableSpeechFeatures {
		fmt.Printf("Speech ")
	}
	if featureConfig.EnableHarmonicFeatures {
		fmt.Printf("Harmonic ")
	}
	fmt.Printf("\n")

	// Show feature weights
	weights := extractor.GetFeatureWeights()
	fmt.Printf("        Feature Weights: ")
	for feature, weight := range weights {
		fmt.Printf("%s=%.2f ", feature, weight)
	}
	fmt.Printf("\n")
}

func calculateStreamSimilarity(features1, features2 *extractors.ExtractedFeatures) float64 {
	// Simple similarity calculation based on spectral centroid correlation
	if features1.SpectralFeatures == nil || features2.SpectralFeatures == nil {
		return 0.0
	}

	centroid1 := features1.SpectralFeatures.SpectralCentroid
	centroid2 := features2.SpectralFeatures.SpectralCentroid

	// Use shorter length
	minLen := len(centroid1)
	if len(centroid2) < minLen {
		minLen = len(centroid2)
	}

	if minLen == 0 {
		return 0.0
	}

	// Calculate correlation
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

	correlation := numerator / (denominator * 0.5) // Simplified sqrt

	// Convert to similarity (0-1 range)
	return (correlation + 1.0) / 2.0
}

func logDetailedComparison(pairID string, features1, features2 *extractors.ExtractedFeatures, similarity float64) {
	fmt.Printf("      Detailed comparison for %s:\n", pairID)
	fmt.Printf("        Spectral similarity: %.3f\n", similarity)

	if features1.EnergyFeatures != nil && features2.EnergyFeatures != nil {
		energyDiff := abs(features1.EnergyFeatures.EnergyVariance - features2.EnergyFeatures.EnergyVariance)
		fmt.Printf("        Energy variance diff: %.3f\n", energyDiff)
	}

	if features1.TemporalFeatures != nil && features2.TemporalFeatures != nil {
		dynamicDiff := abs(features1.TemporalFeatures.DynamicRange - features2.TemporalFeatures.DynamicRange)
		fmt.Printf("        Dynamic range diff: %.3f\n", dynamicDiff)
	}
}

func countValidAlignments(alignments map[string]*extractors.AlignmentResult) int {
	count := 0
	for _, alignment := range alignments {
		if alignment.IsValid {
			count++
		}
	}
	return count
}

func countStrongMatches(comparisons map[string]float64) int {
	count := 0
	for _, similarity := range comparisons {
		if similarity > 0.8 {
			count++
		}
	}
	return count
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// createFeatureConfig creates content-optimized feature configuration
func createFeatureConfig(contentType config.ContentType, sampleRate int) *config.FeatureConfig {
	// Base configuration
	featureConfig := &config.FeatureConfig{
		WindowSize: 2048,
		HopSize:    512,
		FreqRange:  [2]float64{20.0, float64(sampleRate / 2)}, // Full frequency range

		// Default feature enables
		EnableChroma:           false,
		EnableMFCC:             true,
		EnableSpectralContrast: false,
		EnableTemporalFeatures: true,
		EnableSpeechFeatures:   false,
		EnableHarmonicFeatures: false,

		// Default parameters
		MFCCCoefficients: 13,
		ChromaBins:       12,
		ContrastBands:    6,

		// Default weights (will be overridden by extractor)
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
		featureConfig.MFCCCoefficients = 13
		featureConfig.ChromaBins = 12
		featureConfig.MatchThreshold = 0.80
		featureConfig.SimilarityWeights = map[string]float64{
			"spectral": 1.0,
			"chroma":   0.9,
			"harmonic": 0.8,
			"temporal": 0.7,
		}

	case config.ContentNews, config.ContentTalk:
		featureConfig.WindowSize = 1024 // Shorter window for speech
		featureConfig.HopSize = 256
		featureConfig.EnableSpeechFeatures = true
		featureConfig.EnableMFCC = true
		featureConfig.MFCCCoefficients = 12                // Fewer coefficients for speech
		featureConfig.FreqRange = [2]float64{80.0, 8000.0} // Speech frequency range
		featureConfig.MatchThreshold = 0.70
		featureConfig.SimilarityWeights = map[string]float64{
			"spectral": 1.0,
			"mfcc":     0.9,
			"speech":   0.8,
			"temporal": 0.6,
		}

	case config.ContentSports:
		featureConfig.EnableTemporalFeatures = true
		featureConfig.EnableSpectralContrast = true
		featureConfig.WindowSize = 1024 // Shorter for dynamic content
		featureConfig.HopSize = 256
		featureConfig.MatchThreshold = 0.75
		featureConfig.SimilarityWeights = map[string]float64{
			"spectral": 1.0,
			"temporal": 1.0, // High weight on temporal for sports
			"energy":   0.9,
			"contrast": 0.7,
		}

	case config.ContentMixed:
		// Enable everything for mixed content
		featureConfig.EnableChroma = true
		featureConfig.EnableMFCC = true
		featureConfig.EnableSpectralContrast = true
		featureConfig.EnableTemporalFeatures = true
		featureConfig.EnableSpeechFeatures = true
		featureConfig.EnableHarmonicFeatures = true
		featureConfig.MatchThreshold = 0.72
		featureConfig.SimilarityWeights = map[string]float64{
			"spectral": 1.0,
			"mfcc":     0.8,
			"temporal": 0.8,
			"chroma":   0.6,
			"harmonic": 0.6,
		}

	default: // ContentUnknown
		// Conservative approach - enable basic features
		featureConfig.EnableMFCC = true
		featureConfig.EnableTemporalFeatures = true
		featureConfig.MatchThreshold = 0.75
	}

	return featureConfig
}

// parseContentTypeFlag converts string flag to ContentType
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

func extractAudioFromLocalFiles(ctx context.Context, filePaths []string, segmentDuration time.Duration) (*stream.ParallelExtractionResult, error) {
	results := make([]*stream.AudioExtractionResult, len(filePaths))
	successfulStreams := 0
	failedStreams := 0

	for i, filePath := range filePaths {
		// Remove file:// prefix if present
		cleanPath := strings.TrimPrefix(filePath, "file://")

		fmt.Printf("   Reading file: %s\n", cleanPath)

		// Read the file
		fileData, err := os.ReadFile(cleanPath)
		if err != nil {
			results[i] = &stream.AudioExtractionResult{
				URL:   filePath,
				Error: fmt.Errorf("failed to read file: %w", err),
			}
			failedStreams++
			continue
		}

		// Decode the audio using your transcode package
		decoder := transcode.NewNormalizingDecoder(ftContentType)

		anyData, err := decoder.DecodeBytes(fileData)
		if err != nil {
			results[i] = &stream.AudioExtractionResult{
				URL:   filePath,
				Error: fmt.Errorf("failed to decode audio: %w", err),
			}
			failedStreams++
			continue
		}

		var audioData *common.AudioData
		if commonAudio, ok := anyData.(*common.AudioData); ok {
			audioData = commonAudio
		} else {
			// Use reflection to convert unknown AudioData type to common.AudioData
			audioData = common.ConvertToAudioData(anyData)
			if audioData == nil {
				return nil, fmt.Errorf("decoder returned unexpected type: %T", anyData)
			}
		}

		// Truncate to segment duration if needed
		if segmentDuration > 0 {
			maxSamples := int(segmentDuration.Seconds() * float64(audioData.SampleRate))
			if len(audioData.PCM) > maxSamples {
				audioData.PCM = audioData.PCM[:maxSamples]
			}
		}

		results[i] = &stream.AudioExtractionResult{
			URL:       filePath,
			AudioData: audioData,
			Error:     nil,
		}
		successfulStreams++

		fmt.Printf("   ✅ Loaded: %s (%.1fs, %dHz, %d samples)\n",
			cleanPath,
			float64(len(audioData.PCM))/float64(audioData.SampleRate),
			audioData.SampleRate,
			len(audioData.PCM))
	}

	return &stream.ParallelExtractionResult{
		Results:           results,
		SuccessfulStreams: successfulStreams,
		FailedStreams:     failedStreams,
	}, nil
}

// Check if input is a local file path
func isLocalFile(input string) bool {
	// Check for file:// prefix or local file patterns
	return strings.HasPrefix(input, "file://") ||
		strings.HasPrefix(input, "/") ||
		strings.HasPrefix(input, "./") ||
		strings.HasPrefix(input, "../") ||
		(!strings.HasPrefix(input, "http://") && !strings.HasPrefix(input, "https://"))
}
