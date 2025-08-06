package app

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/configs"
	"github.com/tunein/cdn-benchmark-cli/internal/benchmark"
	"github.com/tunein/cdn-benchmark-cli/internal/latency"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
	"github.com/tunein/cdn-benchmark-cli/pkg/output"
)

// Context holds the application context and configuration
type Context struct {
	// CLI arguments
	ConfigFile          string // Application configuration file (optional)
	BroadcastConfigFile string // Broadcast groups configuration file (required)
	OutputFile          string
	OutputFormat        string
	Timeout             time.Duration
	SegmentDuration     time.Duration
	MaxConcurrent       int
	Verbose             bool
	Quiet               bool
	DetailedAnalysis    bool
	SkipFingerprint     bool

	// Runtime context
	Logger          logging.Logger
	Config          *BenchmarkConfig
	BroadcastConfig *BroadcastConfig
}

// BenchmarkApp handles the benchmark application lifecycle
type BenchmarkApp struct {
	ctx             *Context
	config          *BenchmarkConfig
	broadcastConfig *BroadcastConfig
	logger          logging.Logger
}

// NewBenchmarkApp creates a new benchmark application
func NewBenchmarkApp(ctx *Context) (*BenchmarkApp, error) {
	// Set up logging
	logger := setupLogging(ctx)
	ctx.Logger = logger

	// Load configuration
	config, broadcastConfig, err := loadAndMergeConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to load configuration: %w", err)
	}
	ctx.Config = config
	ctx.BroadcastConfig = broadcastConfig

	logger.Debug("Benchmark application initialized", logging.Fields{
		"app_config_file":       ctx.ConfigFile,
		"broadcast_config_file": ctx.BroadcastConfigFile,
		"output_format":         ctx.OutputFormat,
		"timeout":               ctx.Timeout.Seconds(),
		"broadcast_groups":      len(broadcastConfig.GetEnabledBroadcastGroups()),
	})

	return &BenchmarkApp{
		ctx:             ctx,
		config:          config,
		broadcastConfig: broadcastConfig,
		logger:          logger,
	}, nil
}

// Run executes the benchmark
func (app *BenchmarkApp) Run(ctx context.Context) error {
	app.logger.Debug("Starting CDN benchmark execution", logging.Fields{
		"enabled_groups": len(app.broadcastConfig.GetEnabledBroadcastGroups()),
	})

	// Create and run benchmark orchestrator
	orchestrator, err := benchmark.NewOrchestrator(app.config, app.broadcastConfig, app.logger)
	if err != nil {
		return fmt.Errorf("failed to create benchmark orchestrator: %w", err)
	}

	summary, err := orchestrator.RunBenchmarkByIndex(ctx, 0) // DEPLOYMENT: The index will be the job's assigned index
	if err != nil {
		return fmt.Errorf("benchmark execution failed: %w", err)
	}

	// Generate detailed analytics if requested
	var performanceMetrics *benchmark.PerformanceMetrics
	var qualityMetrics *benchmark.QualityMetrics
	var reliabilityMetrics *benchmark.ReliabilityMetrics

	if app.ctx.DetailedAnalysis {
		app.logger.Debug("Generating detailed analytics")
		metricsCalculator := benchmark.NewMetricsCalculator(app.logger)
		performanceMetrics = metricsCalculator.CalculatePerformanceMetrics(summary)
		qualityMetrics = metricsCalculator.CalculateQualityMetrics(summary)
		reliabilityMetrics = metricsCalculator.CalculateReliabilityMetrics(summary)
	}

	// Output results
	if err := app.outputResults(summary, performanceMetrics, qualityMetrics, reliabilityMetrics); err != nil {
		return fmt.Errorf("failed to output results: %w", err)
	}

	// Return error if all broadcasts failed
	if summary.FailedBroadcasts > 0 && summary.SuccessfulBroadcasts == 0 {
		return fmt.Errorf("all benchmark measurements failed")
	}

	return nil
}

// setupLogging configures logging based on context
func setupLogging(ctx *Context) logging.Logger {
	// TODO: Create logger with appropriate level based on ctx.Verbose/ctx.Quiet
	// For now, return default logger
	return logging.NewDefaultLogger()
}

// loadAndMergeConfig loads configuration from files and merges with CLI flags
func loadAndMergeConfig(ctx *Context) (*BenchmarkConfig, *BroadcastConfig, error) {
	// Load base configuration
	baseConfig, err := configs.LoadConfig()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load base configuration: %w", err)
	}

	fmt.Printf("Base Config")
	fmt.Printf("Stream.ReadTimeout: %d", baseConfig.Stream.ReadTimeout)
	fmt.Printf("HLS.HTTP.ReadTimeout: %d", baseConfig.HLS.HTTP.ReadTimeout)
	fmt.Printf("HLS.Audio.SampleDuration: %d", baseConfig.HLS.Audio.SampleDuration)
	fmt.Printf("ICEcast.HTTP.ReadTimeout: %d", baseConfig.ICEcast.HTTP.ReadTimeout)
	fmt.Printf("ICEcast.Audio.SampleDuration: %d", baseConfig.ICEcast.Audio.SampleDuration)
	fmt.Printf("Quality.MaxLatency: %d", baseConfig.Quality.MaxLatency)

	// Load benchmark-specific configuration from file
	var benchmarkConfig *BenchmarkConfig
	if ctx.ConfigFile != "" {
		benchmarkConfig, err = loadBenchmarkConfigFromFile(ctx.ConfigFile)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to load benchmark configuration: %w", err)
		}
	}

	fmt.Printf("Benchmark Config")
	fmt.Printf("Benchmark.OperationTimeout: %d", benchmarkConfig.Benchmark.OperationTimeout)
	fmt.Printf("Benchmark.AudioSegmentDuration: %d", benchmarkConfig.Benchmark.AudioSegmentDuration)
	fmt.Printf("Benchmark.MaxAlignmentOffset: %f", benchmarkConfig.Benchmark.MaxAlignmentOffset)

	// Load broadcast groups configuration from file (required)
	if ctx.BroadcastConfigFile == "" {
		return nil, nil, fmt.Errorf("broadcast configuration file is required")
	}

	broadcastConfig, err := loadBroadcastConfigFromFile(ctx.BroadcastConfigFile)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load broadcast configuration: %w", err)
	}

	// Merge configurations
	mergedConfig := mergeBenchmarkConfig(baseConfig, benchmarkConfig, ctx)

	// Validate final configurations
	if err := mergedConfig.Validate(); err != nil {
		return nil, nil, fmt.Errorf("invalid benchmark configuration: %w", err)
	}

	if err := broadcastConfig.Validate(); err != nil {
		return nil, nil, fmt.Errorf("invalid broadcast configuration: %w", err)
	}

	return mergedConfig, broadcastConfig, nil
}

// outputResults handles all result output
func (app *BenchmarkApp) outputResults(summary *latency.BenchmarkSummary, performance *benchmark.PerformanceMetrics, quality *benchmark.QualityMetrics, reliability *benchmark.ReliabilityMetrics) error {
	// Create clean output structure (exclude raw data)
	outputData := map[string]any{
		"benchmark_summary": cleanBenchmarkSummary(summary),
		"timestamp":         time.Now(),
		"configuration": map[string]any{
			"segment_duration":  app.ctx.SegmentDuration.Seconds(),
			"timeout":           app.ctx.Timeout.Seconds(),
			"detailed_analysis": app.ctx.DetailedAnalysis,
			"skip_fingerprint":  app.ctx.SkipFingerprint,
		},
	}

	// Add detailed metrics if available (but clean them)
	if performance != nil || app.config.Verbose {
		outputData["performance_metrics"] = performance
	}
	if quality != nil || app.config.Verbose {
		outputData["quality_metrics"] = quality
	}
	if reliability != nil || app.config.Verbose {
		outputData["reliability_metrics"] = reliability
	}

	// Create formatter
	var formatter output.Formatter
	switch app.ctx.OutputFormat {
	case "json":
		formatter = &output.JSONFormatter{}
	case "yaml":
		formatter = &output.YAMLFormatter{}
	case "csv":
		formatter = &output.CSVFormatter{}
	case "table":
		formatter = &output.TableFormatter{}
	default:
		formatter = &output.JSONFormatter{}
	}

	// Format data
	formattedData, err := formatter.Format(outputData, true)
	if err != nil {
		// If JSON formatting fails due to infinite values, try to sanitize the data
		if strings.Contains(err.Error(), "unsupported value") {
			sanitizedData := sanitizeForJSON(outputData)
			formattedData, err = formatter.Format(sanitizedData, true)
		}
		if err != nil {
			return fmt.Errorf("failed to format output data: %w", err)
		}
	}

	// Write to file or stdout
	if app.ctx.OutputFile != "" {
		return app.writeToFile(formattedData)
	}

	_, err = os.Stdout.Write(formattedData)
	return err
}

// cleanBenchmarkSummary removes raw data from the benchmark summary
func cleanBenchmarkSummary(summary *latency.BenchmarkSummary) map[string]any {
	cleanSummary := map[string]any{
		"start_time":              summary.StartTime,
		"end_time":                summary.EndTime,
		"total_duration":          summary.TotalDuration.Seconds(),
		"successful_broadcasts":   summary.SuccessfulBroadcasts,
		"failed_broadcasts":       summary.FailedBroadcasts,
		"overall_health_score":    summary.OverallHealthScore,
		"average_latency_metrics": summary.AverageLatencyMetrics,
		"broadcast_measurements":  make(map[string]any),
	}

	// Clean broadcast measurements (remove raw audio and fingerprint data)
	for name, broadcast := range summary.BroadcastMeasurements {
		cleanBroadcast := map[string]any{
			"group_name":                   broadcast.Group.Name,
			"content_type":                 broadcast.Group.ContentType,
			"total_benchmark_time_seconds": broadcast.TotalBenchmarkTime.Seconds(),
			"liveness_metrics":             broadcast.LivenessMetrics,
			"overall_validation":           broadcast.OverallValidation,
			"stream_measurements":          cleanStreamMeasurements(broadcast.StreamMeasurements),
			"alignment_measurements":       cleanAlignmentMeasurements(broadcast.AlignmentMeasurements),
			"fingerprint_comparisons":      cleanFingerprintComparisons(broadcast.FingerprintComparisons),
		}

		if broadcast.Error != nil {
			cleanBroadcast["error"] = broadcast.Error.Error()
		}

		cleanSummary["broadcast_measurements"].(map[string]any)[name] = cleanBroadcast
	}

	return cleanSummary
}

// cleanStreamMeasurements removes raw audio and fingerprint data
func cleanStreamMeasurements(measurements map[string]*latency.StreamMeasurement) map[string]any {
	clean := make(map[string]any)

	for name, measurement := range measurements {
		cleanMeasurement := map[string]any{
			"endpoint": map[string]any{
				"url":          measurement.Endpoint.URL,
				"type":         measurement.Endpoint.Type,
				"role":         measurement.Endpoint.Role,
				"content_type": measurement.Endpoint.ContentType,
				"enabled":      measurement.Endpoint.Enabled,
			},
			"time_to_first_byte":    measurement.TimeToFirstByte.Milliseconds(),
			"audio_extraction_time": measurement.AudioExtractionTime.Milliseconds(),
			"fingerprint_time":      measurement.FingerprintTime.Milliseconds(),
			"total_processing_time": measurement.TotalProcessingTime.Milliseconds(),
			"stream_validation":     measurement.StreamValidation,
			"timestamp":             measurement.Timestamp,
		}

		// Add audio duration but not raw PCM data
		if measurement.AudioData != nil {
			cleanMeasurement["audio_duration_seconds"] = measurement.AudioData.Duration.Seconds()
		}

		if measurement.Error != nil {
			cleanMeasurement["error"] = measurement.Error.Error()
		}

		clean[name] = cleanMeasurement
	}

	return clean
}

// cleanAlignmentMeasurements removes raw audio data from alignment results
func cleanAlignmentMeasurements(measurements map[string]*latency.AlignmentMeasurement) map[string]any {
	clean := make(map[string]any)

	for name, measurement := range measurements {
		cleanMeasurement := map[string]any{
			"latency_seconds":    measurement.LatencySeconds,
			"is_valid_alignment": measurement.IsValidAlignment,
			"comparison_time_ms": measurement.ComparisonTime.Milliseconds(),
			"timestamp":          measurement.Timestamp,
		}

		// Include alignment result but without raw feature data
		if measurement.AlignmentResult != nil {
			cleanMeasurement["alignment_result"] = map[string]any{
				"temporal_offset_seconds": measurement.AlignmentResult.TemporalOffset,
				"offset_confidence":       measurement.AlignmentResult.OffsetConfidence,
				"overall_similarity":      measurement.AlignmentResult.OverallSimilarity,
				"alignment_quality":       measurement.AlignmentResult.AlignmentQuality,
				"time_stretch":            measurement.AlignmentResult.TimeStretch,
				"method":                  measurement.AlignmentResult.Method,
			}
		}

		if measurement.Error != nil {
			cleanMeasurement["error"] = measurement.Error.Error()
		}

		clean[name] = cleanMeasurement
	}

	return clean
}

// cleanFingerprintComparisons removes raw fingerprint data
func cleanFingerprintComparisons(comparisons map[string]*latency.FingerprintComparison) map[string]any {
	clean := make(map[string]any)

	for name, comparison := range comparisons {
		cleanComparison := map[string]any{
			"is_valid_match":     comparison.IsValidMatch,
			"comparison_time_ms": comparison.ComparisonTime.Milliseconds(),
			"timestamp":          comparison.Timestamp,
		}

		// Include similarity result but without raw feature data
		if comparison.SimilarityResult != nil {
			cleanComparison["similarity_result"] = map[string]any{
				"overall_similarity":      comparison.SimilarityResult.OverallSimilarity,
				"confidence":              comparison.SimilarityResult.Confidence,
				"hash_similarity":         comparison.SimilarityResult.HashSimilarity,
				"temporal_offset_seconds": comparison.SimilarityResult.TemporalOffset,
				"feature_distances":       comparison.SimilarityResult.FeatureDistances,
			}
		}

		if comparison.Error != nil {
			cleanComparison["error"] = comparison.Error.Error()
		}

		clean[name] = cleanComparison
	}

	return clean
}

// writeToFile writes data to the specified output file
func (app *BenchmarkApp) writeToFile(data []byte) error {
	// Ensure directory exists
	dir := filepath.Dir(app.ctx.OutputFile)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Write file
	if err := os.WriteFile(app.ctx.OutputFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write output file: %w", err)
	}

	app.logger.Info("Results written to file", logging.Fields{
		"output_file": app.ctx.OutputFile,
		"size_bytes":  len(data),
	})

	return nil
}

// sanitizeForJSON recursively cleans infinite and NaN values from any data structure
func sanitizeForJSON(data any) any {
	switch v := data.(type) {
	case float64:
		if math.IsInf(v, 0) || math.IsNaN(v) {
			return 0.0
		}
		return v
	case float32:
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			return float32(0.0)
		}
		return v
	case map[string]any:
		result := make(map[string]any)
		for k, val := range v {
			result[k] = sanitizeForJSON(val)
		}
		return result
	case []any:
		result := make([]any, len(v))
		for i, val := range v {
			result[i] = sanitizeForJSON(val)
		}
		return result
	case []float64:
		result := make([]float64, len(v))
		for i, val := range v {
			if math.IsInf(val, 0) || math.IsNaN(val) {
				result[i] = 0.0
			} else {
				result[i] = val
			}
		}
		return result
	default:
		// Use reflection to handle structs and other complex types
		return sanitizeWithReflection(data)
	}
}

// sanitizeWithReflection uses reflection to sanitize struct fields
func sanitizeWithReflection(data any) any {
	if data == nil {
		return nil
	}

	val := reflect.ValueOf(data)
	if val.Kind() == reflect.Ptr {
		if val.IsNil() {
			return nil
		}
		val = val.Elem()
	}

	switch val.Kind() {
	case reflect.Struct:
		result := make(map[string]any)
		typ := val.Type()
		for i := 0; i < val.NumField(); i++ {
			field := val.Field(i)
			fieldType := typ.Field(i)

			// Skip unexported fields
			if !field.CanInterface() {
				continue
			}

			// Get JSON tag name or use field name
			jsonTag := fieldType.Tag.Get("json")
			fieldName := fieldType.Name
			if jsonTag != "" && jsonTag != "-" {
				// Parse JSON tag (handle omitempty, etc.)
				parts := strings.Split(jsonTag, ",")
				if parts[0] != "" {
					fieldName = parts[0]
				}
			}

			result[fieldName] = sanitizeForJSON(field.Interface())
		}
		return result
	case reflect.Slice:
		result := make([]any, val.Len())
		for i := 0; i < val.Len(); i++ {
			result[i] = sanitizeForJSON(val.Index(i).Interface())
		}
		return result
	case reflect.Map:
		result := make(map[string]any)
		for _, key := range val.MapKeys() {
			keyStr := fmt.Sprintf("%v", key.Interface())
			result[keyStr] = sanitizeForJSON(val.MapIndex(key).Interface())
		}
		return result
	case reflect.Float64:
		f := val.Float()
		if math.IsInf(f, 0) || math.IsNaN(f) {
			return 0.0
		}
		return f
	case reflect.Float32:
		f := val.Float()
		if math.IsInf(f, 0) || math.IsNaN(f) {
			return float32(0.0)
		}
		return float32(f)
	default:
		return val.Interface()
	}
}
