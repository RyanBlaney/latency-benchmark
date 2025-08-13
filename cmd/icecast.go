package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/RyanBlaney/latency-benchmark/configs"
	"github.com/RyanBlaney/latency-benchmark/pkg/stream/common"
	"github.com/RyanBlaney/latency-benchmark/pkg/stream/icecast"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	icecastTimeout             time.Duration
	icecastVerbose             bool
	icecastValidateStream      bool
	icecastShowMetadata        bool
	icecastShowConfig          bool
	icecastTestAudio           bool
	icecastShowICYData         bool
	icecastSampleDuration      time.Duration
	icecastMaxReadAttempts     int
	icecastCustomConfig        string
	icecastShowHealthCheck     bool
	icecastShowQualityAnalysis bool
	icecastFollowMetadata      bool
	icecastDetectionTimeout    time.Duration
)

var icecastCmd = &cobra.Command{
	Use:   "icecast [url]",
	Short: "ICEcast-specific testing and analysis",
	Long: `Perform comprehensive ICEcast testing including stream validation,
metadata extraction, ICY data parsing, audio quality assessment, 
connection health monitoring, and real-time metadata following.

This command tests the complete ICEcast detection and processing pipeline:
- Configuration loading and validation
- URL pattern detection
- HTTP header analysis
- ICY metadata extraction
- Stream validation
- Audio downloading and quality assessment
- Connection health monitoring
- Real-time metadata updates

Examples:
  # Test basic ICEcast detection and metadata
  icecast http://stream.example.com:8000/stream

  # Full comprehensive testing with audio
  icecast --validate --quality-analysis --health-check --test-audio --verbose http://radio.example.com:8000/listen

  # Show detailed metadata and configuration
  icecast --show-metadata --show-config --show-icy-data --verbose http://stream.example.com/radio.mp3

  # Follow live metadata updates
  icecast --follow-metadata --timeout 60s http://live.example.com:8000/stream

  # Test with custom settings
  icecast --sample-duration 45s --max-read-attempts 5 --detection-timeout 10s http://stream.example.com/audio`,
	Args: cobra.ExactArgs(1),
	RunE: runICEcastTest,
}

func init() {
	rootCmd.AddCommand(icecastCmd)

	icecastCmd.Flags().DurationVar(&icecastTimeout, "timeout", 30*time.Second,
		"operation timeout")
	icecastCmd.Flags().BoolVarP(&icecastVerbose, "verbose", "v", false,
		"verbose output")
	icecastCmd.Flags().BoolVar(&icecastValidateStream, "validate", false,
		"perform comprehensive stream validation")
	icecastCmd.Flags().BoolVar(&icecastShowMetadata, "show-metadata", false,
		"show detailed stream metadata")
	icecastCmd.Flags().BoolVar(&icecastShowConfig, "show-config", false,
		"show effective configuration")
	icecastCmd.Flags().BoolVar(&icecastTestAudio, "test-audio", false,
		"download and test audio data")
	icecastCmd.Flags().BoolVar(&icecastShowICYData, "show-icy-data", false,
		"show detailed ICY metadata")
	icecastCmd.Flags().DurationVar(&icecastSampleDuration, "sample-duration", 30*time.Second,
		"audio sample duration")
	icecastCmd.Flags().IntVar(&icecastMaxReadAttempts, "max-read-attempts", 10,
		"maximum read attempts for audio data")
	icecastCmd.Flags().StringVar(&icecastCustomConfig, "config", "",
		"path to custom ICEcast configuration file")
	icecastCmd.Flags().BoolVar(&icecastShowHealthCheck, "health-check", false,
		"show stream health assessment")
	icecastCmd.Flags().BoolVar(&icecastShowQualityAnalysis, "quality-analysis", false,
		"show quality assessment and scoring")
	icecastCmd.Flags().BoolVar(&icecastFollowMetadata, "follow-metadata", false,
		"follow live metadata updates")
	icecastCmd.Flags().DurationVar(&icecastDetectionTimeout, "detection-timeout", 5*time.Second,
		"stream detection timeout")
}

func runICEcastTest(cmd *cobra.Command, args []string) error {
	url := args[0]
	verbose := icecastVerbose || viper.GetBool("verbose")

	printHeader("ICEcast Stream Testing", url)

	ctx, cancel := context.WithTimeout(context.Background(), icecastTimeout)
	defer cancel()

	timer := NewPerformanceTimer()
	timer.StartEvent("total_test")

	// Step 1: Configuration Loading
	timer.StartEvent("config_loading")
	printStep(1, "Configuration Loading")

	appConfig, err := configs.LoadConfig()
	if err != nil {
		printError("Failed to load application config: %v", err)
		return fmt.Errorf("failed to load application config: %w", err)
	}
	printSuccess("Application configuration loaded")

	icecastConfig := appConfig.ToICEcastConfig()
	if icecastConfig == nil {
		icecastConfig = icecast.DefaultConfig()
		printWarning("Using default ICEcast configuration")
	} else {
		printSuccess("ICEcast configuration created from app config")
	}

	// Apply custom settings from flags
	if icecastSampleDuration != 30*time.Second {
		icecastConfig.Audio.SampleDuration = icecastSampleDuration
	}
	if icecastMaxReadAttempts != 10 {
		icecastConfig.Audio.MaxReadAttempts = icecastMaxReadAttempts
	}
	if icecastDetectionTimeout != 5*time.Second {
		icecastConfig.Detection.TimeoutSeconds = int(icecastDetectionTimeout.Seconds())
	}

	if err := icecastConfig.Validate(); err != nil {
		printError("Configuration validation failed: %v", err)
		return fmt.Errorf("configuration validation failed: %w", err)
	}
	printSuccess("Configuration validation passed")

	timer.EndEvent("config_loading")
	fmt.Println()

	if icecastShowConfig {
		printSectionHeader("Configuration Details")
		displayICEcastConfiguration(icecastConfig, verbose)
		fmt.Println()
	}

	// Step 2: URL Pattern Detection
	timer.StartEvent("url_detection")
	printStep(2, "URL Pattern Detection")

	detector := icecast.NewDetectorWithConfig(icecastConfig.Detection)
	urlDetection := detector.DetectFromURL(url)

	if urlDetection == common.StreamTypeICEcast {
		printSuccess("URL pattern indicates ICEcast stream")
	} else {
		printWarning("URL pattern does not match ICEcast")
	}
	printInfo("Detection result: %s", urlDetection)

	timer.EndEvent("url_detection")
	fmt.Println()

	// Step 3: HTTP Header Detection
	timer.StartEvent("header_detection")
	printStep(3, "HTTP Header Detection")

	headerDetection := detector.DetectFromHeaders(ctx, url, icecastConfig.HTTP)

	if headerDetection == common.StreamTypeICEcast {
		printSuccess("HTTP headers indicate ICEcast stream")
	} else {
		printWarning("HTTP headers do not indicate ICEcast")
	}
	printInfo("Detection result: %s", headerDetection)

	timer.EndEvent("header_detection")
	fmt.Println()

	// Step 4: Stream Probing
	timer.StartEvent("stream_probing")
	printStep(4, "Stream Probing")

	metadata, err := icecast.ProbeStreamWithConfig(ctx, url, icecastConfig)
	if err != nil {
		printError("Stream probing failed: %v", err)
		return fmt.Errorf("stream probing failed: %w", err)
	}

	printSuccess("Stream probing completed")
	printInfo("Stream Type: %s", metadata.Type)
	printInfo("Codec: %s", metadata.Codec)
	if metadata.Bitrate > 0 {
		printInfo("Bitrate: %d kbps", metadata.Bitrate)
	}
	if metadata.Station != "" {
		printInfo("Station: %s", metadata.Station)
	}

	timer.EndEvent("stream_probing")
	fmt.Println()

	// Step 5: Handler Integration Test
	timer.StartEvent("handler_integration")
	printStep(5, "Handler Integration Test")

	handler := icecast.NewHandlerWithConfig(icecastConfig)
	defer handler.Close()

	canHandle := handler.CanHandle(ctx, url)
	if canHandle {
		printSuccess("Handler can process this stream")
	} else {
		printError("Handler cannot process this stream")
		return fmt.Errorf("handler rejected stream")
	}

	err = handler.Connect(ctx, url)
	if err != nil {
		printError("Connection failed: %v", err)
		return fmt.Errorf("connection failed: %w", err)
	}
	printSuccess("Successfully connected to stream")

	handlerMetadata, err := handler.GetMetadata()
	if err != nil {
		printWarning("Metadata extraction failed: %v", err)
	} else {
		printSuccess("Metadata extracted successfully")
		if verbose {
			displayMetadata(handlerMetadata, verbose)
		}
	}

	timer.EndEvent("handler_integration")
	fmt.Println()

	// Step 6: ICY Metadata Analysis
	if icecastShowICYData && handler.HasICYMetadata() {
		timer.StartEvent("icy_analysis")
		printStep(6, "ICY Metadata Analysis")

		printSuccess("ICY metadata supported")
		printInfo("Metadata Interval: %d bytes", handler.GetICYMetadataInterval())

		currentTitle := handler.GetCurrentICYTitle()
		if currentTitle != "" {
			printInfo("Current Title: %s", currentTitle)

			artist, title := icecast.ParseICYTitle(currentTitle)
			if artist != "" && title != "" {
				printInfo("Parsed Artist: %s", artist)
				printInfo("Parsed Title: %s", title)
			}
		} else {
			printInfo("No current ICY title available")
		}

		timer.EndEvent("icy_analysis")
		fmt.Println()
	}

	// Step 7: Audio Testing
	var audioData *common.AudioData
	if icecastTestAudio {
		timer.StartEvent("audio_processing")
		printStep(7, "Audio Download and Processing")

		audioData, err = handler.ReadAudio(ctx)
		if err != nil {
			printWarning("Audio download failed: %v", err)
		} else {
			printSuccess("Audio data downloaded successfully")
			displayAudioData(audioData, verbose)
		}

		timer.EndEvent("audio_processing")
		fmt.Println()
	}

	// Step 8: Stream Validation
	var validationPassed bool
	if icecastValidateStream {
		timer.StartEvent("stream_validation")
		printStep(8, "Stream Validation")

		validator := icecast.NewValidatorWithConfig(icecastConfig)

		if err := validator.ValidateURL(ctx, url); err != nil {
			printWarning("URL validation failed: %v", err)
		} else {
			printSuccess("URL validation passed")
		}

		if err := validator.ValidateStream(ctx, handler); err != nil {
			printWarning("Stream validation failed: %v", err)
		} else {
			printSuccess("Stream validation passed")
			validationPassed = true
		}

		if audioData != nil {
			if err := validator.ValidateAudio(audioData); err != nil {
				printWarning("Audio validation failed: %v", err)
			} else {
				printSuccess("Audio validation passed")
			}
		}

		timer.EndEvent("stream_validation")
		fmt.Println()
	}

	// Step 9: Health Check
	if icecastShowHealthCheck {
		timer.StartEvent("health_check")
		printStep(9, "Health Assessment")

		stats := handler.GetStats()
		responseTime := timer.GetDuration("stream_probing")

		printInfo("Connection Time: %v", stats.ConnectionTime)
		printInfo("Response Time: %v", responseTime)
		printInfo("Bytes Received: %d", stats.BytesReceived)
		printInfo("Average Bitrate: %.1f kbps", stats.AverageBitrate)
		printInfo("Buffer Health: %.1f%%", stats.BufferHealth*100)

		if stats.BufferHealth > 0.8 {
			printSuccess("Stream health: Excellent")
		} else if stats.BufferHealth > 0.5 {
			printWarning("Stream health: Good")
		} else {
			printWarning("Stream health: Poor")
		}

		timer.EndEvent("health_check")
		fmt.Println()
	}

	// Step 10: Quality Analysis
	if icecastShowQualityAnalysis && handlerMetadata != nil {
		timer.StartEvent("quality_analysis")
		printStep(10, "Quality Assessment")

		qualityScore := calculateQualityScore(handlerMetadata, audioData)
		printInfo("Quality Score: %.1f/100", qualityScore)

		if qualityScore >= 90 {
			printSuccess("Quality Rating: Excellent")
		} else if qualityScore >= 70 {
			printSuccess("Quality Rating: Good")
		} else if qualityScore >= 50 {
			printWarning("Quality Rating: Fair")
		} else {
			printWarning("Quality Rating: Poor")
		}

		timer.EndEvent("quality_analysis")
		fmt.Println()
	}

	// Step 11: Follow Metadata Updates
	if icecastFollowMetadata && handler.HasICYMetadata() {
		printStep(11, "Following Live Metadata")
		err = followMetadataUpdates(ctx, handler, verbose)
		if err != nil {
			printWarning("Metadata following failed: %v", err)
		}
		fmt.Println()
	}

	// Performance Summary
	timer.EndEvent("total_test")
	if verbose {
		printSectionHeader("Performance Summary")
		displayPerformanceSummary(timer)
		fmt.Println()
	}

	// Test Summary
	printSectionHeader("Test Summary")
	printTestSummary(icecastConfig, urlDetection, headerDetection, metadata, handlerMetadata,
		canHandle, audioData, validationPassed, timer)

	return nil
}

func printHeader(title, url string) {
	fmt.Printf("%s%s%s%s: %s%s%s\n", ColorBold, ColorBlue, title, ColorReset, ColorCyan, url, ColorReset)
	fmt.Printf("%s%s%s\n\n", ColorBlue, strings.Repeat("═", 80), ColorReset)
}

func printStep(num int, title string) {
	fmt.Printf("%s%s%d%s %s%s%s\n", ColorBold, ColorPurple, num, ColorReset, ColorWhite, title, ColorReset)
}

func printSectionHeader(title string) {
	fmt.Printf("%s%s%s%s\n", ColorBold, ColorBlue, title, ColorReset)
}

func printSuccess(format string, args ...any) {
	fmt.Printf("   %s✓%s %s\n", ColorGreen, ColorReset, fmt.Sprintf(format, args...))
}

func printWarning(format string, args ...any) {
	fmt.Printf("   %s⚠%s %s\n", ColorYellow, ColorReset, fmt.Sprintf(format, args...))
}

func printError(format string, args ...any) {
	fmt.Printf("   %s✗%s %s\n", ColorRed, ColorReset, fmt.Sprintf(format, args...))
}

func printInfo(format string, args ...any) {
	fmt.Printf("   %s•%s %s\n", ColorCyan, ColorReset, fmt.Sprintf(format, args...))
}

func displayICEcastConfiguration(config *icecast.Config, verbose bool) {
	printInfo("HTTP Settings:")
	fmt.Printf("      User Agent: %s\n", config.HTTP.UserAgent)
	fmt.Printf("      Connection Timeout: %v\n", config.HTTP.ConnectionTimeout)
	fmt.Printf("      Read Timeout: %v\n", config.HTTP.ReadTimeout)
	fmt.Printf("      Max Redirects: %d\n", config.HTTP.MaxRedirects)
	fmt.Printf("      Request ICY Meta: %t\n", config.HTTP.RequestICYMeta)

	printInfo("Audio Settings:")
	fmt.Printf("      Buffer Size: %d bytes\n", config.Audio.BufferSize)
	fmt.Printf("      Sample Duration: %v\n", config.Audio.SampleDuration)
	fmt.Printf("      Max Read Attempts: %d\n", config.Audio.MaxReadAttempts)
	fmt.Printf("      Handle ICY Meta: %t\n", config.Audio.HandleICYMeta)

	if verbose {
		printInfo("Detection Settings:")
		fmt.Printf("      Timeout: %d seconds\n", config.Detection.TimeoutSeconds)
		fmt.Printf("      URL Patterns: %d configured\n", len(config.Detection.URLPatterns))
		fmt.Printf("      Content Types: %d configured\n", len(config.Detection.ContentTypes))
		fmt.Printf("      Common Ports: %v\n", config.Detection.CommonPorts)
	}
}

func calculateQualityScore(metadata *common.StreamMetadata, audioData *common.AudioData) float64 {
	score := 50.0 // Base score

	// Bitrate scoring
	if metadata.Bitrate >= 320 {
		score += 20
	} else if metadata.Bitrate >= 192 {
		score += 15
	} else if metadata.Bitrate >= 128 {
		score += 10
	} else if metadata.Bitrate >= 96 {
		score += 5
	}

	// Sample rate scoring
	if metadata.SampleRate >= 48000 {
		score += 15
	} else if metadata.SampleRate >= 44100 {
		score += 10
	} else if metadata.SampleRate >= 22050 {
		score += 5
	}

	// Codec scoring
	switch metadata.Codec {
	case "flac":
		score += 15
	case "aac", "ogg", "opus":
		score += 10
	case "mp3":
		score += 5
	}

	// Audio data scoring
	if audioData != nil && len(audioData.PCM) > 0 {
		score += 10 // Bonus for having audio data
	}

	if score > 100 {
		score = 100
	}

	return score
}

func followMetadataUpdates(ctx context.Context, handler *icecast.Handler, verbose bool) error {
	printInfo("Following live metadata updates for %v...", icecastTimeout)

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	lastTitle := ""

	for {
		select {
		case <-ctx.Done():
			printInfo("Timeout reached")
			return nil
		case <-ticker.C:
			currentTitle := handler.GetCurrentICYTitle()
			if currentTitle != "" && currentTitle != lastTitle {
				printSuccess("Metadata Update: %s", currentTitle)

				if verbose {
					artist, title := icecast.ParseICYTitle(currentTitle)
					if artist != "" && title != "" {
						fmt.Printf("      Artist: %s\n", artist)
						fmt.Printf("      Title: %s\n", title)
					}
				}

				lastTitle = currentTitle
			}
		}
	}
}

func printTestSummary(config *icecast.Config, urlDetection, headerDetection common.StreamType,
	metadata, handlerMetadata *common.StreamMetadata, canHandle bool, audioData *common.AudioData,
	validationPassed bool, timer *PerformanceTimer) {

	printResult("Configuration", config != nil)
	printResult("URL Detection", urlDetection == common.StreamTypeICEcast)
	printResult("Header Detection", headerDetection == common.StreamTypeICEcast)
	printResult("Stream Probing", metadata != nil)
	printResult("Handler Integration", canHandle && handlerMetadata != nil)

	if icecastTestAudio {
		printResult("Audio Processing", audioData != nil)
	}

	if icecastValidateStream {
		printResult("Stream Validation", validationPassed)
	}

	fmt.Println()
	printInfo("Stream Classification:")
	if handlerMetadata != nil {
		fmt.Printf("   Type: ICEcast Stream\n")
		fmt.Printf("   Codec: %s\n", handlerMetadata.Codec)
		if handlerMetadata.Bitrate > 0 {
			fmt.Printf("   Bitrate: %d kbps\n", handlerMetadata.Bitrate)
		}
		if handlerMetadata.Station != "" {
			fmt.Printf("   Station: %s\n", handlerMetadata.Station)
		}
		if handlerMetadata.Genre != "" {
			fmt.Printf("   Genre: %s\n", handlerMetadata.Genre)
		}
	}

	fmt.Printf("\n%sTotal Test Duration: %v%s\n", ColorBold, timer.GetTotalDuration(), ColorReset)
}

func printResult(name string, success bool) {
	if success {
		fmt.Printf("%-20s %s✓ PASS%s\n", name+":", ColorGreen, ColorReset)
	} else {
		fmt.Printf("%-20s %s✗ FAIL%s\n", name+":", ColorRed, ColorReset)
	}
}

func main() {
	if err := icecastCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
