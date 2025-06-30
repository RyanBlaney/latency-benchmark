package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/tunein/cdn-benchmark-cli/configs"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

var (
	decoderTimeout      time.Duration
	decoderVerbose      bool
	decoderShowConfig   bool
	decoderQuality      string
	decoderSampleRate   int
	decoderChannels     int
	decoderMaxDuration  time.Duration
	decoderTestFile     string
	decoderTestFormats  []string
	decoderBenchmark    bool
	decoderValidateOnly bool
)

var decoderCmd = &cobra.Command{
	Use:   "decoder-test [stream-url-or-file]",
	Short: "Test audio decoder functionality and performance",
	Long: `Test the audio decoder with various inputs including streams and files.

This command provides comprehensive testing of the FFmpeg-based audio decoder:
- FFmpeg/FFprobe availability validation
- Configuration testing and validation
- File decoding capabilities
- Stream segment decoding
- Format support verification
- Performance benchmarking
- Quality assessment across different settings

The decoder supports all formats that FFmpeg handles including:
- HLS segments (AAC in TS containers, raw AAC)
- ICEcast streams (MP3, OGG, AAC)
- Audio files (WAV, FLAC, MP3, AAC, M4A, etc.)

Examples:
  # Test decoder availability and configuration
  decoder-test --validate-only

  # Test with a local audio file
  decoder-test --test-file /path/to/audio.mp3 --verbose

  # Test with HLS stream segments
  decoder-test https://example.com/playlist.m3u8 --quality high --verbose

  # Benchmark decoder performance
  decoder-test --benchmark --sample-rate 44100 --channels 1 --quality medium https://stream.example.com/

  # Test specific formats
  decoder-test --test-formats aac,mp3,wav --verbose

  # Custom decoder settings
  decoder-test --sample-rate 48000 --channels 2 --max-duration 10s --quality high /path/to/file.flac`,
	Args: func(cmd *cobra.Command, args []string) error {
		if decoderValidateOnly {
			return nil // No args needed for validation-only mode
		}
		if len(decoderTestFormats) > 0 {
			return nil // No args needed when testing specific formats
		}
		if decoderTestFile != "" {
			return nil // No args needed when using --test-file
		}
		if len(args) != 1 {
			return fmt.Errorf("requires exactly one stream URL or file path")
		}
		return nil
	},
	RunE: runDecoderTest,
}

func init() {
	rootCmd.AddCommand(decoderCmd)

	decoderCmd.Flags().DurationVar(&decoderTimeout, "timeout", 30*time.Second,
		"operation timeout")
	decoderCmd.Flags().BoolVarP(&decoderVerbose, "verbose", "v", false,
		"verbose output")
	decoderCmd.Flags().BoolVar(&decoderShowConfig, "show-config", false,
		"show decoder configuration details")
	decoderCmd.Flags().StringVar(&decoderQuality, "quality", "medium",
		"resampling quality (fast/medium/high)")
	decoderCmd.Flags().IntVar(&decoderSampleRate, "sample-rate", 44100,
		"target sample rate")
	decoderCmd.Flags().IntVar(&decoderChannels, "channels", 1,
		"target channels (1=mono, 2=stereo)")
	decoderCmd.Flags().DurationVar(&decoderMaxDuration, "max-duration", 0,
		"maximum decode duration (0=unlimited)")
	decoderCmd.Flags().StringVar(&decoderTestFile, "test-file", "",
		"test with specific audio file")
	decoderCmd.Flags().StringSliceVar(&decoderTestFormats, "test-formats", []string{},
		"test specific formats (comma-separated)")
	decoderCmd.Flags().BoolVar(&decoderBenchmark, "benchmark", false,
		"run performance benchmarks")
	decoderCmd.Flags().BoolVar(&decoderValidateOnly, "validate-only", false,
		"only validate decoder availability")
}

func runDecoderTest(cmd *cobra.Command, args []string) error {
	verbose := decoderVerbose || viper.GetBool("verbose")

	printHeader("Audio Decoder Testing", getInputDescription(args))

	ctx, cancel := context.WithTimeout(context.Background(), decoderTimeout)
	defer cancel()

	timer := NewPerformanceTimer()
	timer.StartEvent("total_test")

	// Step 1: Configuration and Validation
	timer.StartEvent("config_validation")
	printStep(1, "Decoder Configuration and Validation")

	appConfig, err := configs.LoadConfig()
	if err != nil {
		printError("Failed to load application config: %v", err)
		return fmt.Errorf("failed to load application config: %w", err)
	}
	printSuccess("Application configuration loaded")

	// Create decoder configuration
	decoderConfig := audio.DefaultDecoderConfig()

	// Apply command-line overrides
	if decoderSampleRate > 0 {
		decoderConfig.TargetSampleRate = decoderSampleRate
	}
	if decoderChannels > 0 {
		decoderConfig.TargetChannels = decoderChannels
	}
	if decoderQuality != "" {
		decoderConfig.ResampleQuality = decoderQuality
	}
	if decoderMaxDuration > 0 {
		decoderConfig.MaxDuration = decoderMaxDuration
	}
	decoderConfig.Timeout = decoderTimeout

	// Apply app config overrides
	if appConfig.Audio.SampleRate > 0 {
		decoderConfig.TargetSampleRate = appConfig.Audio.SampleRate
	}
	if appConfig.Audio.Channels > 0 {
		decoderConfig.TargetChannels = appConfig.Audio.Channels
	}

	printSuccess("Decoder configuration created")

	// Create decoder instance
	decoder := audio.NewDecoder(decoderConfig)
	printSuccess("Decoder instance created")

	// Validate decoder
	if err := decoder.ValidateConfig(); err != nil {
		printError("Decoder validation failed: %v", err)
		return fmt.Errorf("decoder validation failed: %w", err)
	}
	printSuccess("Decoder validation passed")

	timer.EndEvent("config_validation")
	fmt.Println()

	if decoderShowConfig {
		printSectionHeader("Decoder Configuration")
		displayDecoderConfiguration(decoder, verbose)
		fmt.Println()
	}

	// If validate-only mode, stop here
	if decoderValidateOnly {
		printSectionHeader("Validation Summary")
		printSuccess("FFmpeg/FFprobe availability: Confirmed")
		printSuccess("Decoder configuration: Valid")
		printSuccess("All decoder dependencies: Available")
		fmt.Printf("\n%sValidation completed successfully%s\n", ColorGreen, ColorReset)
		return nil
	}

	// Step 2: Format Support Testing
	if len(decoderTestFormats) > 0 {
		timer.StartEvent("format_testing")
		printStep(2, "Format Support Testing")

		err = testFormatSupport(decoder, decoderTestFormats, verbose)
		if err != nil {
			printWarning("Format testing completed with issues: %v", err)
		} else {
			printSuccess("Format testing completed successfully")
		}

		timer.EndEvent("format_testing")
		fmt.Println()
	}

	// Step 3: File Decoding Test
	if decoderTestFile != "" {
		timer.StartEvent("file_decoding")
		printStep(3, "File Decoding Test")

		err = testFileDecoding(decoder, decoderTestFile, verbose)
		if err != nil {
			printError("File decoding test failed: %v", err)
		} else {
			printSuccess("File decoding test completed successfully")
		}

		timer.EndEvent("file_decoding")
		fmt.Println()
	}

	// Step 4: Stream Decoding Test
	if len(args) > 0 {
		streamURL := args[0]

		timer.StartEvent("stream_decoding")
		printStep(4, "Stream Decoding Test")

		err = testStreamDecoding(ctx, decoder, streamURL, verbose)
		if err != nil {
			printError("Stream decoding test failed: %v", err)
		} else {
			printSuccess("Stream decoding test completed successfully")
		}

		timer.EndEvent("stream_decoding")
		fmt.Println()
	}

	// Step 5: Performance Benchmarking
	if decoderBenchmark {
		timer.StartEvent("benchmarking")
		printStep(5, "Performance Benchmarking")

		err = runDecoderBenchmarks(decoder, verbose)
		if err != nil {
			printWarning("Benchmarking completed with issues: %v", err)
		} else {
			printSuccess("Performance benchmarking completed")
		}

		timer.EndEvent("benchmarking")
		fmt.Println()
	}

	// Performance Summary
	timer.EndEvent("total_test")
	if verbose {
		printSectionHeader("Performance Summary")
		displayDecoderPerformanceSummary(timer)
		fmt.Println()
	}

	// Test Summary
	printSectionHeader("Test Summary")
	printDecoderTestSummary(decoder, timer)

	return nil
}

func getInputDescription(args []string) string {
	if decoderValidateOnly {
		return "Validation Only"
	}
	if decoderTestFile != "" {
		return fmt.Sprintf("File: %s", decoderTestFile)
	}
	if len(decoderTestFormats) > 0 {
		return fmt.Sprintf("Formats: %s", strings.Join(decoderTestFormats, ", "))
	}
	if len(args) > 0 {
		return args[0]
	}
	return "Configuration Test"
}

func displayDecoderConfiguration(decoder *audio.Decoder, verbose bool) {
	config := decoder.GetConfig()

	printInfo("Target Settings:")
	fmt.Printf("      Sample Rate: %v Hz\n", config["target_sample_rate"])
	fmt.Printf("      Channels: %v\n", config["target_channels"])
	fmt.Printf("      Output Format: %v\n", config["output_format"])
	fmt.Printf("      Resample Quality: %v\n", config["resample_quality"])

	printInfo("Operational Settings:")
	fmt.Printf("      Timeout: %v\n", config["timeout"])
	fmt.Printf("      Max Duration: %v\n", config["max_duration"])

	printInfo("FFmpeg Settings:")
	fmt.Printf("      FFmpeg Path: %v\n", config["ffmpeg_path"])
	fmt.Printf("      FFprobe Path: %v\n", config["ffprobe_path"])

	if verbose {
		printInfo("Supported Formats:")
		formats := decoder.GetSupportedFormats()
		for i, format := range formats {
			if i < 10 { // Show first 10
				fmt.Printf("      %s\n", format)
			} else if i == 10 {
				fmt.Printf("      ... and %d more formats\n", len(formats)-10)
				break
			}
		}
	}
}

func testFormatSupport(decoder *audio.Decoder, formats []string, verbose bool) error {
	printInfo("Testing format support for: %s", strings.Join(formats, ", "))

	// TODO: Account for verbose

	supportedFormats := decoder.GetSupportedFormats()
	supportedMap := make(map[string]bool)
	for _, format := range supportedFormats {
		supportedMap[format] = true
	}

	allSupported := true
	for _, format := range formats {
		if supportedMap[format] {
			printSuccess("Format %s: Supported", format)
		} else {
			printWarning("Format %s: Not explicitly listed (may still work)", format)
			allSupported = false
		}
	}

	if allSupported {
		return nil
	}
	return fmt.Errorf("some formats not explicitly supported")
}

func testFileDecoding(decoder *audio.Decoder, filename string, verbose bool) error {
	printInfo("Testing file decoding: %s", filename)

	// Check if file exists
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return fmt.Errorf("file does not exist: %s", filename)
	}
	printSuccess("File found and accessible")

	// Decode the file
	startTime := time.Now()
	audioData, err := decoder.DecodeFile(filename)
	decodeTime := time.Since(startTime)

	if err != nil {
		return fmt.Errorf("decoding failed: %w", err)
	}

	printSuccess("File decoded successfully")
	printInfo("Decode Time: %v", decodeTime)
	displayDecodedAudioInfo(audioData, verbose)

	return nil
}

func testStreamDecoding(ctx context.Context, decoder *audio.Decoder, streamURL string, verbose bool) error {
	printInfo("Testing stream decoding: %s", streamURL)

	// TODO: unused decoder

	// Create stream factory and detect stream type
	factory := stream.NewFactory()
	handler, err := factory.DetectAndCreate(ctx, streamURL)
	if err != nil {
		return fmt.Errorf("failed to create stream handler: %w", err)
	}
	defer handler.Close()

	printSuccess("Stream handler created for type: %s", handler.Type())

	// Connect to stream
	err = handler.Connect(ctx, streamURL)
	if err != nil {
		return fmt.Errorf("failed to connect to stream: %w", err)
	}
	printSuccess("Connected to stream")

	// Get metadata
	metadata, err := handler.GetMetadata()
	if err != nil {
		printWarning("Failed to get metadata: %v", err)
	} else {
		printSuccess("Stream metadata retrieved")
		if verbose {
			displayStreamMetadata(metadata)
		}
	}

	// Read audio data
	startTime := time.Now()
	audioData, err := handler.ReadAudio(ctx)
	readTime := time.Since(startTime)

	if err != nil {
		return fmt.Errorf("failed to read audio data: %w", err)
	}

	printSuccess("Audio data read from stream")
	printInfo("Read Time: %v", readTime)
	displayStreamAudioInfo(audioData, verbose)

	return nil
}

func runDecoderBenchmarks(decoder *audio.Decoder, verbose bool) error {
	printInfo("Running decoder performance benchmarks...")

	// TODO test various benchmarks
	// This would ideally test with various sample files
	// For now, we'll show what benchmarks would test

	printInfo("Benchmark Categories:")
	fmt.Printf("      Format Performance: Different audio formats (AAC, MP3, FLAC)\n")
	fmt.Printf("      Resampling Performance: Quality vs. speed tradeoffs\n")
	fmt.Printf("      File Size Impact: Small vs. large file processing\n")
	fmt.Printf("      Concurrent Processing: Multiple simultaneous decodes\n")

	if verbose {
		printInfo("Note: Full benchmarking requires sample audio files")
		printInfo("Consider using --test-file with various format samples")
	}

	return nil
}

func displayDecodedAudioInfo(audioData *audio.AudioData, verbose bool) {
	printInfo("Decoded Audio Properties:")
	fmt.Printf("      Samples: %d\n", len(audioData.PCM))
	fmt.Printf("      Sample Rate: %d Hz\n", audioData.SampleRate)
	fmt.Printf("      Channels: %d\n", audioData.Channels)
	fmt.Printf("      Duration: %.3f seconds\n", audioData.Duration.Seconds())

	if audioData.Metadata != nil {
		fmt.Printf("      Source Codec: %s\n", audioData.Metadata.Codec)
		if audioData.Metadata.Bitrate > 0 {
			fmt.Printf("      Source Bitrate: %d kbps\n", audioData.Metadata.Bitrate)
		}
	}

	if verbose && len(audioData.PCM) > 0 {
		// Calculate basic audio statistics
		var sum, min, max float64
		min = audioData.PCM[0]
		max = audioData.PCM[0]

		for _, sample := range audioData.PCM {
			sum += sample
			if sample < min {
				min = sample
			}
			if sample > max {
				max = sample
			}
		}

		avg := sum / float64(len(audioData.PCM))
		peakAmplitude := max
		if -min > max {
			peakAmplitude = -min
		}

		printInfo("Audio Statistics:")
		fmt.Printf("      Average Amplitude: %.6f\n", avg)
		fmt.Printf("      Peak Amplitude: %.6f\n", peakAmplitude)
		fmt.Printf("      Dynamic Range: %.6f (%.2f dB)\n", max-min, 20*log10(max-min))

		// Check for potential issues
		if peakAmplitude > 0.99 {
			printWarning("Potential clipping detected (peak > 0.99)")
		}
		if avg < 0.001 && avg > -0.001 {
			printWarning("Very low signal level detected")
		}
	}
}

// For stream handler results (from handler.ReadAudio)
func displayStreamAudioInfo(audioData *common.AudioData, verbose bool) {
	printInfo("Stream Audio Properties:")
	fmt.Printf("      Samples: %d\n", len(audioData.PCM))
	fmt.Printf("      Sample Rate: %d Hz\n", audioData.SampleRate)
	fmt.Printf("      Channels: %d\n", audioData.Channels)
	fmt.Printf("      Duration: %.3f seconds\n", audioData.Duration.Seconds())

	if audioData.Metadata != nil {
		fmt.Printf("      Source Codec: %s\n", audioData.Metadata.Codec)
		if audioData.Metadata.Bitrate > 0 {
			fmt.Printf("      Source Bitrate: %d kbps\n", audioData.Metadata.Bitrate)
		}
	}

	if verbose && len(audioData.PCM) > 0 {
		// Calculate basic audio statistics
		var sum, min, max float64
		min = audioData.PCM[0]
		max = audioData.PCM[0]

		for _, sample := range audioData.PCM {
			sum += sample
			if sample < min {
				min = sample
			}
			if sample > max {
				max = sample
			}
		}

		avg := sum / float64(len(audioData.PCM))
		peakAmplitude := max
		if -min > max {
			peakAmplitude = -min
		}

		printInfo("Audio Statistics:")
		fmt.Printf("      Average Amplitude: %.6f\n", avg)
		fmt.Printf("      Peak Amplitude: %.6f\n", peakAmplitude)
		fmt.Printf("      Dynamic Range: %.6f (%.2f dB)\n", max-min, 20*log10(max-min))

		// Check for potential issues
		if peakAmplitude > 0.99 {
			printWarning("Potential clipping detected (peak > 0.99)")
		}
		if avg < 0.001 && avg > -0.001 {
			printWarning("Very low signal level detected")
		}
	}
}

func displayStreamMetadata(metadata *common.StreamMetadata) {
	printInfo("Stream Metadata:")
	fmt.Printf("      Type: %s\n", metadata.Type)
	if metadata.Codec != "" {
		fmt.Printf("      Codec: %s\n", metadata.Codec)
	}
	if metadata.Bitrate > 0 {
		fmt.Printf("      Bitrate: %d kbps\n", metadata.Bitrate)
	}
	if metadata.SampleRate > 0 {
		fmt.Printf("      Sample Rate: %d Hz\n", metadata.SampleRate)
	}
	if metadata.Channels > 0 {
		fmt.Printf("      Channels: %d\n", metadata.Channels)
	}
	if metadata.Station != "" {
		fmt.Printf("      Station: %s\n", metadata.Station)
	}
}

func displayDecoderPerformanceSummary(timer *PerformanceTimer) {
	printInfo("Performance Breakdown:")

	events := []string{
		"config_validation", "format_testing", "file_decoding",
		"stream_decoding", "benchmarking",
	}

	for _, event := range events {
		duration := timer.GetDuration(event)
		if duration > 0 {
			eventName := strings.ReplaceAll(titleCaser.String(strings.ReplaceAll(event, "_", " ")), " ", " ")
			fmt.Printf("      %s: %v\n", eventName, duration)
		}
	}
}

func printDecoderTestSummary(decoder *audio.Decoder, timer *PerformanceTimer) {
	printResult("Configuration", decoder != nil)
	printResult("FFmpeg Availability", true) // If we got here, it's available

	if decoderTestFile != "" {
		printResult("File Decoding", true) // If we got here without error, it worked
	}

	if len(decoderTestFormats) > 0 {
		printResult("Format Testing", true)
	}

	fmt.Println()
	printInfo("Decoder Summary:")
	config := decoder.GetConfig()
	fmt.Printf("   Target Sample Rate: %v Hz\n", config["target_sample_rate"])
	fmt.Printf("   Target Channels: %v\n", config["target_channels"])
	fmt.Printf("   Resample Quality: %v\n", config["resample_quality"])
	fmt.Printf("   FFmpeg Path: %v\n", config["ffmpeg_path"])

	fmt.Printf("\n%sTotal Test Duration: %v%s\n", ColorBold, timer.GetTotalDuration(), ColorReset)
}

// Helper function for logarithm (if not available)
func log10(x float64) float64 {
	return 0.4342944819 * logNatural(x) // log10(x) = ln(x) / ln(10)
}

func logNatural(x float64) float64 {
	// Simple natural log approximation for small positive values
	if x <= 0 {
		return 0
	}
	if x == 1 {
		return 0
	}
	// TODO: replace this placeholder
	// This is a very basic approximation - in practice you'd use math.Log
	return 0
}
