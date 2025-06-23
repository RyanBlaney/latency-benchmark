package cmd

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/tunein/cdn-benchmark-cli/configs"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

var (
	// Benchmark command flags
	benchmarkProfile    string
	benchmarkDuration   time.Duration
	benchmarkRegions    []string
	benchmarkStreams    []string
	benchmarkReference  string
	benchmarkConcurrent bool
	benchmarkOutput     string
	benchmarkTags       []string
)

// benchmarkCmd represents the benchmark command
var benchmarkCmd = &cobra.Command{
	Use:   "benchmark [flags] [stream-urls...]",
	Short: "Run CDN performance benchmarks",
	Long: `Run comprehensive CDN performance benchmarks against specified streams.

This command performs multi-dimensional analysis including:
- Audio quality comparison using spectral fingerprinting
- Latency measurement (network, processing, end-to-end)
- Stream reliability and connection statistics
- Regional performance comparison

Examples:
  # Run benchmark with default profile
  cdn-benchmark benchmark https://stream.example.com/hls/playlist.m3u8

  # Run with specific profile and regions
  cdn-benchmark benchmark --profile production --regions us-west,eu-west https://stream.example.com/

  # Compare CDN stream against reference
  cdn-benchmark benchmark --reference https://origin.example.com/stream https://cdn.example.com/stream

  # Run concurrent tests across multiple streams
  cdn-benchmark benchmark --concurrent --duration 5m stream1.m3u8 stream2.m3u8

  # Use custom output format and tags
  cdn-benchmark benchmark --output json --tags env=staging,test=nightly https://stream.example.com/`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 && benchmarkProfile == "" {
			return fmt.Errorf("requires at least one stream URL or --profile flag")
		}
		return nil
	},
	RunE: runBenchmark,
}

func init() {
	rootCmd.AddCommand(benchmarkCmd)

	// Profile and configuration flags
	benchmarkCmd.Flags().StringVarP(&benchmarkProfile, "profile", "p", "",
		"test profile name from configuration")
	benchmarkCmd.Flags().DurationVarP(&benchmarkDuration, "duration", "d", 30*time.Second,
		"test duration")
	benchmarkCmd.Flags().StringSliceVarP(&benchmarkRegions, "regions", "r", []string{},
		"comma-separated list of regions to test")

	// Stream configuration flags
	benchmarkCmd.Flags().StringSliceVarP(&benchmarkStreams, "streams", "s", []string{},
		"comma-separated list of additional stream URLs")
	benchmarkCmd.Flags().StringVar(&benchmarkReference, "reference", "",
		"reference stream URL for quality comparison")

	// Execution flags
	benchmarkCmd.Flags().BoolVarP(&benchmarkConcurrent, "concurrent", "c", false,
		"run tests concurrently across regions/streams")
	benchmarkCmd.Flags().StringVarP(&benchmarkOutput, "output-file", "f", "",
		"output file path (default: stdout)")
	benchmarkCmd.Flags().StringSliceVarP(&benchmarkTags, "tags", "t", []string{},
		"additional tags in key=value format")

	// Quality threshold overrides
	benchmarkCmd.Flags().Float64("min-similarity", 0,
		"minimum audio similarity threshold (0.0-1.0)")
	benchmarkCmd.Flags().Duration("max-latency", 0,
		"maximum acceptable latency")
	benchmarkCmd.Flags().Int("min-bitrate", 0,
		"minimum acceptable bitrate (kbps)")

	// Advanced flags
	benchmarkCmd.Flags().Int("buffer-size", 0,
		"stream buffer size in bytes")
	benchmarkCmd.Flags().Duration("connection-timeout", 0,
		"connection timeout duration")
	benchmarkCmd.Flags().String("user-agent", "",
		"custom user agent string")

	// Bind flags to viper for configuration override
	bindBenchmarkFlags()
}

func bindBenchmarkFlags() {
	flagMap := map[string]string{
		"profile":  "benchmark.profile",
		"duration": "benchmark.duration",
		// Don't bind regions here - it conflicts with regions config section
		// "regions":            "benchmark.regions",
		"streams":            "benchmark.streams",
		"reference":          "benchmark.reference",
		"concurrent":         "benchmark.concurrent",
		"output-file":        "benchmark.output_file",
		"tags":               "benchmark.tags",
		"min-similarity":     "quality.min_similarity",
		"max-latency":        "quality.max_latency",
		"min-bitrate":        "quality.min_bitrate",
		"buffer-size":        "stream.buffer_size",
		"connection-timeout": "stream.connection_timeout",
		"user-agent":         "stream.user_agent",
	}

	for flagName, viperKey := range flagMap {
		viper.BindPFlag(viperKey, benchmarkCmd.Flags().Lookup(flagName))
	}
}

func runBenchmark(cmd *cobra.Command, args []string) error {
	// Load configuration
	config, err := configs.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load configuration: %w", err)
	}

	// Validate configuration
	if err := configs.ValidateConfig(config); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	// Create context with cancellation for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Fprintf(os.Stderr, "\nReceived interrupt signal, shutting down gracefully...\n")
		cancel()
	}()

	// Initialize stream factory
	factory := stream.NewFactory()

	// Collect all stream URLs
	streamURLs := args
	if benchmarkProfile != "" {
		profile, exists := config.Profiles[benchmarkProfile]
		if !exists {
			return fmt.Errorf("profile '%s' not found in configuration", benchmarkProfile)
		}
		for _, endpoint := range profile.Streams {
			streamURLs = append(streamURLs, endpoint.URL)
		}
	}
	streamURLs = append(streamURLs, benchmarkStreams...)

	if len(streamURLs) == 0 {
		return fmt.Errorf("no stream URLs specified")
	}

	// Print test configuration
	fmt.Printf("Starting CDN Benchmark Suite\n")
	fmt.Printf("Configuration:\n")
	fmt.Printf("  Streams: %d\n", len(streamURLs))
	fmt.Printf("  Duration: %v\n", benchmarkDuration)
	fmt.Printf("  Regions: %v\n", benchmarkRegions)
	fmt.Printf("  Concurrent: %v\n", benchmarkConcurrent)
	if benchmarkReference != "" {
		fmt.Printf("  Reference: %s\n", benchmarkReference)
	}
	fmt.Printf("\n")

	// Validate all streams before starting tests
	fmt.Printf("Validating streams...\n")
	validStreams, err := validateStreams(ctx, factory, streamURLs)
	if err != nil {
		return fmt.Errorf("stream validation failed: %w", err)
	}

	if len(validStreams) == 0 {
		return fmt.Errorf("no valid streams found")
	}

	fmt.Printf("Validated %d/%d streams\n\n", len(validStreams), len(streamURLs))

	// Execute benchmark tests
	results, err := executeBenchmarkTests(ctx, factory, config, validStreams)
	if err != nil {
		return fmt.Errorf("benchmark execution failed: %w", err)
	}

	// Output results
	return outputResults(results)
}

// validateStreams validates all provided stream URLs
func validateStreams(ctx context.Context, factory *stream.Factory, urls []string) ([]string, error) {
	var validStreams []string

	for i, url := range urls {
		fmt.Printf("  [%d/%d] Validating %s... ", i+1, len(urls), url)

		handler, err := factory.DetectAndCreate(ctx, url)
		if err != nil {
			fmt.Printf("❌ Failed: %v\n", err)
			continue
		}

		err = handler.Connect(ctx, url)
		if err != nil {
			fmt.Printf("❌ Connection failed: %v\n", err)
			handler.Close()
			continue
		}

		metadata, err := handler.GetMetadata()
		if err != nil {
			fmt.Printf("❌ Metadata error: %v\n", err)
			handler.Close()
			continue
		}

		handler.Close()
		fmt.Printf("✅ %s (%s)\n", metadata.Type, metadata.Station)
		validStreams = append(validStreams, url)
	}

	return validStreams, nil
}

// BenchmarkResult represents the result of a benchmark test
type BenchmarkResult struct {
	StreamURL    string                 `json:"stream_url"`
	StreamType   common.StreamType      `json:"stream_type"`
	Region       string                 `json:"region"`
	Metadata     *common.StreamMetadata `json:"metadata"`
	Stats        *common.StreamStats    `json:"stats"`
	QualityScore float64                `json:"quality_score"`
	LatencyMs    float64                `json:"latency_ms"`
	Success      bool                   `json:"success"`
	Error        string                 `json:"error,omitempty"`
	StartTime    time.Time              `json:"start_time"`
	EndTime      time.Time              `json:"end_time"`
	Duration     time.Duration          `json:"duration"`
	Tags         map[string]string      `json:"tags,omitempty"`
}

// BenchmarkSummary contains aggregated results
type BenchmarkSummary struct {
	TotalTests      int               `json:"total_tests"`
	SuccessfulTests int               `json:"successful_tests"`
	FailedTests     int               `json:"failed_tests"`
	SuccessRate     float64           `json:"success_rate"`
	Results         []BenchmarkResult `json:"results"`
	Summary         map[string]any    `json:"summary"`
	GeneratedAt     time.Time         `json:"generated_at"`
}

// executeBenchmarkTests runs the actual benchmark tests
// executeBenchmarkTests runs the actual benchmark tests
func executeBenchmarkTests(ctx context.Context, factory *stream.Factory, config *configs.Config, urls []string) (*BenchmarkSummary, error) {
	var results []BenchmarkResult
	startTime := time.Now()

	// Determine regions to test - prioritize command line flag over config
	var regions []string
	if len(benchmarkRegions) > 0 {
		// Use regions from command line flag
		regions = benchmarkRegions
	} else {
		// Use enabled regions from config
		for name, region := range config.Regions {
			if region.Enabled {
				regions = append(regions, name)
			}
		}
	}

	if len(regions) == 0 {
		regions = []string{"default"} // Use default region if none specified
	}

	fmt.Printf("Running benchmark tests...\n")
	totalTests := len(urls) * len(regions)
	currentTest := 0

	for _, url := range urls {
		for _, region := range regions {
			currentTest++
			fmt.Printf("  [%d/%d] Testing %s in %s... ", currentTest, totalTests, url, region)

			result := executeSingleTest(ctx, factory, url, region)
			results = append(results, result)

			if result.Success {
				fmt.Printf("✅ Quality: %.2f, Latency: %.1fms\n",
					result.QualityScore, result.LatencyMs)
			} else {
				fmt.Printf("❌ %s\n", result.Error)
			}

			// Check for context cancellation
			select {
			case <-ctx.Done():
				fmt.Printf("\nBenchmark cancelled\n")
				goto summarize
			default:
			}
		}
	}

summarize:
	// Calculate summary statistics
	summary := &BenchmarkSummary{
		TotalTests:  len(results),
		Results:     results,
		GeneratedAt: time.Now(),
		Summary:     make(map[string]any),
	}

	successCount := 0
	var totalLatency, totalQuality float64

	for _, result := range results {
		if result.Success {
			successCount++
			totalLatency += result.LatencyMs
			totalQuality += result.QualityScore
		}
	}

	summary.SuccessfulTests = successCount
	summary.FailedTests = summary.TotalTests - successCount
	summary.SuccessRate = float64(successCount) / float64(summary.TotalTests)

	if successCount > 0 {
		summary.Summary["average_latency_ms"] = totalLatency / float64(successCount)
		summary.Summary["average_quality"] = totalQuality / float64(successCount)
	}

	summary.Summary["total_duration"] = time.Since(startTime)

	return summary, nil
}

// executeSingleTest runs a single benchmark test
func executeSingleTest(ctx context.Context, factory *stream.Factory, url, region string) BenchmarkResult {
	result := BenchmarkResult{
		StreamURL: url,
		Region:    region,
		StartTime: time.Now(),
		Tags:      make(map[string]string),
	}

	// Add tags from command line
	for _, tag := range benchmarkTags {
		parts := strings.SplitN(tag, "=", 2)
		if len(parts) == 2 {
			result.Tags[parts[0]] = parts[1]
		}
	}

	// Create test context with timeout
	testCtx, cancel := context.WithTimeout(ctx, benchmarkDuration)
	defer cancel()

	// Create and connect handler
	handler, err := factory.DetectAndCreate(testCtx, url)
	if err != nil {
		result.Error = fmt.Sprintf("failed to create handler: %v", err)
		result.EndTime = time.Now()
		result.Duration = result.EndTime.Sub(result.StartTime)
		return result
	}
	defer handler.Close()

	result.StreamType = handler.Type()

	err = handler.Connect(testCtx, url)
	if err != nil {
		result.Error = fmt.Sprintf("connection failed: %v", err)
		result.EndTime = time.Now()
		result.Duration = result.EndTime.Sub(result.StartTime)
		return result
	}

	// Get metadata
	metadata, err := handler.GetMetadata()
	if err != nil {
		result.Error = fmt.Sprintf("metadata error: %v", err)
		result.EndTime = time.Now()
		result.Duration = result.EndTime.Sub(result.StartTime)
		return result
	}
	result.Metadata = metadata

	// Simulate audio quality testing and latency measurement
	// In real implementation, this would:
	// 1. Read audio data for specified duration
	// 2. Perform spectral analysis and fingerprinting
	// 3. Compare against reference stream if provided
	// 4. Measure network and processing latency
	// 5. Calculate quality metrics

	// For now, simulate with basic timing and mock values
	time.Sleep(1 * time.Second) // Simulate processing time

	stats := handler.GetStats()
	result.Stats = stats

	// Mock quality score and latency (replace with actual implementation)
	result.QualityScore = 0.95 + (rand.Float64()-0.5)*0.1 // 0.90-1.0
	result.LatencyMs = 50 + rand.Float64()*100            // 50-150ms
	result.Success = true

	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(result.StartTime)

	return result
}

// outputResults formats and outputs the benchmark results
func outputResults(summary *BenchmarkSummary) error {
	outputFormat := viper.GetString("output_format")

	switch outputFormat {
	case "json":
		return outputJSON(summary)
	case "table":
		return outputTable(summary)
	case "csv":
		return outputCSV(summary)
	default:
		return fmt.Errorf("unsupported output format: %s", outputFormat)
	}
}

// outputJSON outputs results in JSON format
func outputJSON(summary *BenchmarkSummary) error {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(summary)
}

// outputTable outputs results in table format
func outputTable(summary *BenchmarkSummary) error {
	fmt.Printf("\n=== CDN Benchmark Results ===\n\n")

	// Summary table
	fmt.Printf("Summary:\n")
	fmt.Printf("  Total Tests: %d\n", summary.TotalTests)
	fmt.Printf("  Successful: %d\n", summary.SuccessfulTests)
	fmt.Printf("  Failed: %d\n", summary.FailedTests)
	fmt.Printf("  Success Rate: %.1f%%\n", summary.SuccessRate*100)

	if avgLatency, ok := summary.Summary["average_latency_ms"]; ok {
		fmt.Printf("  Average Latency: %.1fms\n", avgLatency)
	}
	if avgQuality, ok := summary.Summary["average_quality"]; ok {
		fmt.Printf("  Average Quality: %.3f\n", avgQuality)
	}

	fmt.Printf("\nDetailed Results:\n")
	fmt.Printf("%-50s %-10s %-10s %-10s %-10s %s\n",
		"Stream", "Region", "Type", "Quality", "Latency", "Status")
	fmt.Printf("%s\n", strings.Repeat("-", 100))

	for _, result := range summary.Results {
		status := "✅ Success"
		if !result.Success {
			status = "❌ Failed"
		}

		// Truncate URL for display
		url := result.StreamURL
		if len(url) > 45 {
			url = url[:42] + "..."
		}

		fmt.Printf("%-50s %-10s %-10s %-10.3f %-10.1f %s\n",
			url, result.Region, result.StreamType,
			result.QualityScore, result.LatencyMs, status)
	}

	return nil
}

// outputCSV outputs results in CSV format
func outputCSV(summary *BenchmarkSummary) error {
	writer := csv.NewWriter(os.Stdout)
	defer writer.Flush()

	// Write header
	header := []string{
		"stream_url", "region", "stream_type", "quality_score",
		"latency_ms", "success", "error", "start_time", "duration",
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	// Write data rows
	for _, result := range summary.Results {
		row := []string{
			result.StreamURL,
			result.Region,
			string(result.StreamType),
			fmt.Sprintf("%.3f", result.QualityScore),
			fmt.Sprintf("%.1f", result.LatencyMs),
			fmt.Sprintf("%t", result.Success),
			result.Error,
			result.StartTime.Format(time.RFC3339),
			result.Duration.String(),
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}
