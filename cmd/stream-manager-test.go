package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/tunein/cdn-benchmark-cli/configs"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
)

var (
	smtVerbose         bool
	smtDebug           bool
	smtSegmentDuration time.Duration
	smtTimeout         time.Duration
	smtParallel        bool
	smtMaxConcurrent   int
)

var smtCmd = &cobra.Command{
	Use:   "stream-manager-test [url1] [url2] [url3...]",
	Short: "Test the Stream Manager for parallel/sequential audio extraction",
	Long: `Test the Stream Manager functionality for extracting audio from multiple 
	streams either in parallel or sequentially. This tests the manager's ability 
	to handle different stream types simultaneously and provides timing analysis.`,
	Args: cobra.MinimumNArgs(2),
	RunE: runStreamManagerTest,
}

func init() {
	rootCmd.AddCommand(smtCmd)

	smtCmd.Flags().BoolVarP(&smtVerbose, "verbose", "v", false,
		"verbose output (overrides global verbose)")
	smtCmd.Flags().BoolVarP(&smtDebug, "debug", "d", false,
		"debug logging mode")
	smtCmd.Flags().DurationVarP(&smtSegmentDuration, "segment-duration", "t", time.Second*10,
		"the downloaded length of each stream")
	smtCmd.Flags().DurationVarP(&smtTimeout, "timeout", "T", time.Second*60,
		"timeout for stream operations")
	smtCmd.Flags().BoolVarP(&smtParallel, "parallel", "p", true,
		"use parallel extraction (false for sequential)")
	smtCmd.Flags().IntVarP(&smtMaxConcurrent, "max-concurrent", "c", 5,
		"maximum concurrent streams for parallel mode")
}

func runStreamManagerTest(cmd *cobra.Command, args []string) error {
	urls := args
	verbose := smtVerbose || viper.GetBool("verbose")

	fmt.Printf("Stream Manager Test\n")
	fmt.Printf("===================\n\n")

	if smtParallel {
		fmt.Printf("Mode: Parallel Extraction\n")
	} else {
		fmt.Printf("Mode: Sequential Extraction\n")
	}
	fmt.Printf("Streams: %d\n", len(urls))
	fmt.Printf("Target Duration: %.1f seconds\n", smtSegmentDuration.Seconds())
	fmt.Printf("Timeout: %.1f seconds\n", smtTimeout.Seconds())
	if smtParallel {
		fmt.Printf("Max Concurrent: %d\n", smtMaxConcurrent)
	}
	fmt.Printf("\n")

	// Create context with overall timeout
	ctx, cancel := context.WithTimeout(context.Background(), smtTimeout)
	defer cancel()

	timer := NewPerformanceTimer()
	timer.StartEvent("overall")

	// Load configuration
	timer.StartEvent("config_loading")
	fmt.Printf("⚙️  Loading configuration...\n")
	appConfig, err := configs.LoadConfig()
	if err != nil {
		return fmt.Errorf("❌ Failed to load config: %v", err)
	}
	fmt.Printf("✅ Configuration loaded\n\n")
	timer.EndEvent("config_loading")

	// Create Stream Manager
	timer.StartEvent("manager_setup")
	fmt.Printf("🏗️  Setting up Stream Manager...\n")

	managerConfig := &stream.ManagerConfig{
		StreamTimeout:        smtTimeout,
		OverallTimeout:       smtTimeout + (10 * time.Second), // Extra buffer
		MaxConcurrentStreams: smtMaxConcurrent,
		ResultBufferSize:     len(urls),
	}

	manager := stream.NewManagerWithConfig(managerConfig)
	fmt.Printf("✅ Stream Manager configured\n")
	if verbose {
		fmt.Printf("   Stream Timeout: %.1fs\n", managerConfig.StreamTimeout.Seconds())
		fmt.Printf("   Overall Timeout: %.1fs\n", managerConfig.OverallTimeout.Seconds())
		fmt.Printf("   Max Concurrent: %d\n", managerConfig.MaxConcurrentStreams)
	}
	fmt.Printf("\n")
	timer.EndEvent("manager_setup")

	// Print stream information
	fmt.Printf("📡 Streams to Process:\n")
	for i, url := range urls {
		fmt.Printf("   %d. %s\n", i+1, url)
	}
	fmt.Printf("\n")

	// Extract audio using Stream Manager
	timer.StartEvent("audio_extraction")
	fmt.Printf("🎵 Starting audio extraction...\n")
	fmt.Printf("   Method: %s\n", map[bool]string{true: "Parallel", false: "Sequential"}[smtParallel])
	fmt.Printf("   Target Duration: %.1f seconds per stream\n", smtSegmentDuration.Seconds())
	fmt.Printf("\n")

	var results *stream.ParallelExtractionResult
	if smtParallel {
		results, err = manager.ExtractAudioParallel(ctx, urls, smtSegmentDuration)
	} else {
		results, err = manager.ExtractAudioSequential(ctx, urls, smtSegmentDuration)
	}

	timer.EndEvent("audio_extraction")

	if err != nil {
		return fmt.Errorf("❌ Audio extraction failed: %v", err)
	}

	fmt.Printf("✅ Audio extraction completed!\n\n")

	// Analyze and display results
	timer.StartEvent("results_analysis")
	fmt.Printf("📊 Results Analysis\n")
	fmt.Printf("==================\n\n")

	fmt.Printf("⏱️  Overall Performance:\n")
	fmt.Printf("   Total Time: %.2f seconds\n", results.TotalDuration.Seconds())
	fmt.Printf("   Successful Streams: %d/%d\n", results.SuccessfulStreams, len(urls))
	fmt.Printf("   Failed Streams: %d/%d\n", results.FailedStreams, len(urls))

	if smtParallel && len(urls) > 1 {
		fmt.Printf("   Max Time Difference: %d ms\n", results.MaxTimeDiff.Milliseconds())

		// Calculate efficiency for parallel mode
		if results.TotalDuration.Seconds() > 0 {
			efficiency := (smtSegmentDuration.Seconds() * float64(results.SuccessfulStreams)) / results.TotalDuration.Seconds()
			fmt.Printf("   Parallel Efficiency: %.2fx\n", efficiency)
		}
	}
	fmt.Printf("\n")

	// Individual stream results
	fmt.Printf("🔍 Individual Stream Results:\n")
	for i, result := range results.Results {
		fmt.Printf("   Stream %d: %s\n", i+1, result.URL)
		fmt.Printf("     Status: %s\n", map[bool]string{true: "✅ Success", false: "❌ Failed"}[result.Error == nil])
		fmt.Printf("     Type: %s\n", result.StreamType)
		fmt.Printf("     Extraction Time: %.2f seconds\n", result.Duration.Seconds())

		if result.Error != nil {
			fmt.Printf("     Error: %v\n", result.Error)
		} else if result.AudioData != nil {
			fmt.Printf("     Audio Duration: %.2f seconds\n", result.AudioData.Duration.Seconds())
			fmt.Printf("     Sample Rate: %d Hz\n", result.AudioData.SampleRate)
			fmt.Printf("     Channels: %d\n", result.AudioData.Channels)
			fmt.Printf("     Samples: %d\n", len(result.AudioData.PCM))

			// Calculate efficiency for this stream
			if result.Duration.Seconds() > 0 {
				streamEfficiency := result.AudioData.Duration.Seconds() / result.Duration.Seconds()
				fmt.Printf("     Stream Efficiency: %.2fx\n", streamEfficiency)
			}
		}

		if result.Metadata != nil {
			fmt.Printf("     Bitrate: %d kbps\n", result.Metadata.Bitrate)
			if result.Metadata.Station != "" {
				fmt.Printf("     Station: %s\n", result.Metadata.Station)
			}
		}
		fmt.Printf("\n")
	}

	// Validation test
	fmt.Printf("🔍 Validation Tests:\n")

	// Test basic validation
	err = manager.ValidateExtractionResults(results, 1, 5000) // At least 1 success, max 5s time diff
	if err != nil {
		fmt.Printf("   ❌ Basic Validation: %v\n", err)
	} else {
		fmt.Printf("   ✅ Basic Validation: Passed\n")
	}

	// Test fingerprinting validation (stricter)
	if results.SuccessfulStreams >= 2 {
		err = manager.ValidateExtractionResults(results, 2, 1000) // At least 2 successes, max 1s time diff
		if err != nil {
			fmt.Printf("   ❌ Fingerprinting Validation: %v\n", err)
		} else {
			fmt.Printf("   ✅ Fingerprinting Validation: Passed\n")
		}
	} else {
		fmt.Printf("   ⚠️  Fingerprinting Validation: Skipped (need 2+ successful streams)\n")
	}

	timer.EndEvent("results_analysis")

	// Performance summary
	timer.EndEvent("overall")
	fmt.Printf("\n")
	fmt.Printf("⏱️  Performance Summary:\n")
	fmt.Printf("   Config Loading: %.0f ms\n", timer.EndEvent("config_loading").Seconds()*1000)
	fmt.Printf("   Manager Setup: %.0f ms\n", timer.EndEvent("manager_setup").Seconds()*1000)
	fmt.Printf("   Audio Extraction: %.2f seconds\n", timer.EndEvent("audio_extraction").Seconds())
	fmt.Printf("   Results Analysis: %.0f ms\n", timer.EndEvent("results_analysis").Seconds()*1000)
	fmt.Printf("   Total Time: %.2f seconds\n", timer.EndEvent("overall").Seconds())

	// Export detailed results if verbose
	if verbose {
		fmt.Printf("\n")
		fmt.Printf("📋 Detailed Results (JSON):\n")
		fmt.Printf("==========================\n")

		detailedResults, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			fmt.Printf("❌ Failed to marshal results: %v\n", err)
		} else {
			fmt.Printf("%s\n", string(detailedResults))
		}
	}

	// Recommendations
	fmt.Printf("\n")
	fmt.Printf("💡 Recommendations:\n")

	if results.FailedStreams > 0 {
		fmt.Printf("   • %d stream(s) failed - check URLs and network connectivity\n", results.FailedStreams)
	}

	if smtParallel && results.MaxTimeDiff.Milliseconds() > 1000 {
		fmt.Printf("   • High timing variation (%.1fs) - consider sequential mode for precise synchronization\n",
			results.MaxTimeDiff.Seconds())
	}

	if results.SuccessfulStreams >= 2 {
		fmt.Printf("   • Results look good for fingerprinting analysis!\n")
	}

	// Performance analysis
	if smtParallel && len(urls) > 1 {
		theoreticalTime := smtSegmentDuration.Seconds()
		actualTime := results.TotalDuration.Seconds()
		if actualTime > theoreticalTime*1.5 {
			fmt.Printf("   • Parallel extraction took %.1fx longer than expected - some streams may be slow\n",
				actualTime/theoreticalTime)
		}
	}

	if verbose {
		fmt.Printf("\n📊 App Config: %+v\n", appConfig)
	}

	return nil
}
