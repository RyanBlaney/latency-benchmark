package app

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
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
	ConfigFile       string
	OutputFile       string
	OutputFormat     string
	Timeout          time.Duration
	SegmentDuration  time.Duration
	MaxConcurrent    int
	Verbose          bool
	Quiet            bool
	DetailedAnalysis bool
	SkipFingerprint  bool

	// Runtime context
	Logger logging.Logger
	Config *BenchmarkConfig
}

// BenchmarkApp handles the benchmark application lifecycle
type BenchmarkApp struct {
	ctx    *Context
	config *BenchmarkConfig
	logger logging.Logger
}

// NewBenchmarkApp creates a new benchmark application
func NewBenchmarkApp(ctx *Context) (*BenchmarkApp, error) {
	// Set up logging
	logger := setupLogging(ctx)
	ctx.Logger = logger

	// Load configuration
	config, err := loadAndMergeConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to load configuration: %w", err)
	}
	ctx.Config = config

	logger.LogInfo("Benchmark application initialized", map[string]interface{}{
		"config_file":     ctx.ConfigFile,
		"output_format":   ctx.OutputFormat,
		"timeout":         ctx.Timeout.Seconds(),
		"broadcast_groups": len(config.BroadcastGroups),
	})

	return &BenchmarkApp{
		ctx:    ctx,
		config: config,
		logger: logger,
	}, nil
}

// Run executes the benchmark
func (app *BenchmarkApp) Run(ctx context.Context) error {
	app.logger.LogInfo("Starting CDN benchmark execution", map[string]interface{}{
		"enabled_groups": len(app.config.GetEnabledBroadcastGroups()),
	})

	// Create and run benchmark orchestrator
	orchestrator, err := benchmark.NewOrchestrator(app.config, app.logger)
	if err != nil {
		return fmt.Errorf("failed to create benchmark orchestrator: %w", err)
	}

	summary, err := orchestrator.RunBenchmark(ctx)
	if err != nil {
		return fmt.Errorf("benchmark execution failed: %w", err)
	}

	// Generate detailed analytics if requested
	var performanceMetrics *benchmark.PerformanceMetrics
	var qualityMetrics *benchmark.QualityMetrics
	var reliabilityMetrics *benchmark.ReliabilityMetrics
	var insights []string

	if app.ctx.DetailedAnalysis {
		app.logger.LogInfo("Generating detailed analytics", nil)
		metricsCalculator := benchmark.NewMetricsCalculator(app.logger)
		performanceMetrics = metricsCalculator.CalculatePerformanceMetrics(summary)
		qualityMetrics = metricsCalculator.CalculateQualityMetrics(summary)
		reliabilityMetrics = metricsCalculator.CalculateReliabilityMetrics(summary)
		insights = metricsCalculator.GenerateInsights(performanceMetrics, qualityMetrics, reliabilityMetrics)
	}

	// Output results
	if err := app.outputResults(summary, performanceMetrics, qualityMetrics, reliabilityMetrics, insights); err != nil {
		return fmt.Errorf("failed to output results: %w", err)
	}

	// Print summary to console if not quiet
	if !app.ctx.Quiet {
		app.printSummary(summary, insights)
	}

	// Return error if all broadcasts failed
	if summary.FailedBroadcasts > 0 && summary.SuccessfulBroadcasts == 0 {
		return fmt.Errorf("all benchmark measurements failed")
	}

	return nil
}

// setupLogging configures logging based on context
func setupLogging(ctx *Context) logging.Logger {
	level := "info"
	if ctx.Verbose {
		level = "debug"
	} else if ctx.Quiet {
		level = "error"
	}

	// TODO: Create logger with appropriate level
	// For now, return default logger
	return logging.NewDefaultLogger()
}

// loadAndMergeConfig loads configuration from file and merges with CLI flags
func loadAndMergeConfig(ctx *Context) (*BenchmarkConfig, error) {
	// Load base configuration
	baseConfig, err := configs.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load base configuration: %w", err)
	}

	// Load benchmark-specific configuration from file
	benchmarkConfig, err := loadBenchmarkConfigFromFile(ctx.ConfigFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load benchmark configuration: %w", err)
	}

	// Merge configurations
	mergedConfig := mergeBenchmarkConfig(baseConfig, benchmarkConfig, ctx)

	// Validate final configuration
	if err := mergedConfig.Validate(); err != nil {
		return nil, fmt.Errorf("invalid merged configuration: %w", err)
	}

	return mergedConfig, nil
}

// outputResults handles all result output
func (app *BenchmarkApp) outputResults(summary *latency.BenchmarkSummary, performance *benchmark.PerformanceMetrics, quality *benchmark.QualityMetrics, reliability *benchmark.ReliabilityMetrics, insights []string) error {
	// Create comprehensive output structure
	outputData := map[string]interface{}{
		"benchmark_summary": summary,
		"timestamp":         time.Now(),
		"configuration": map[string]interface{}{
			"segment_duration":  app.ctx.SegmentDuration.Seconds(),
			"timeout":           app.ctx.Timeout.Seconds(),
			"detailed_analysis": app.ctx.DetailedAnalysis,
			"skip_fingerprint":  app.ctx.SkipFingerprint,
		},
	}

	// Add detailed metrics if available
	if performance != nil {
		outputData["performance_metrics"] = performance
	}
	if quality != nil {
		outputData["quality_metrics"] = quality
	}
	if reliability != nil {
		outputData["reliability_metrics"] = reliability
	}
	if len(insights) > 0 {
		outputData["insights"] = insights
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
		return fmt.Errorf("failed to format output data: %w", err)
	}

	// Write to file or stdout
	if app.ctx.OutputFile != "" {
		return app.writeToFile(formattedData)
	}

	_, err = os.Stdout.Write(formattedData)
	return err
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

	app.logger.LogInfo("Results written to file", map[string]interface{}{
		"output_file": app.ctx.OutputFile,
		"size_bytes":  len(data),
	})

	return nil
}

// printSummary prints a human-readable summary to stdout
func (app *BenchmarkApp) printSummary(summary *latency.BenchmarkSummary, insights []string) {
	fmt.Printf("\n🎯 CDN BENCHMARK SUMMARY\n")
	fmt.Printf("========================\n")
	fmt.Printf("Total Duration:     %.1fs\n", summary.TotalDuration.Seconds())
	fmt.Printf("Successful Groups:  %d\n", summary.SuccessfulBroadcasts)
	fmt.Printf("Failed Groups:      %d\n", summary.FailedBroadcasts)
	fmt.Printf("Overall Health:     %.1f%%\n", summary.OverallHealthScore*100)

	if summary.AverageLatencyMetrics != nil && summary.SuccessfulBroadcasts > 0 {
		fmt.Printf("\n📊 AVERAGE LATENCY METRICS\n")
		fmt.Printf("==========================\n")

		if summary.AverageLatencyMetrics.AvgCDNLatencyHLS > 0 {
			fmt.Printf("HLS CDN Latency:    %.2fs\n", summary.AverageLatencyMetrics.AvgCDNLatencyHLS)
		}

		if summary.AverageLatencyMetrics.AvgCDNLatencyICEcast > 0 {
			fmt.Printf("ICEcast CDN Latency: %.2fs\n", summary.AverageLatencyMetrics.AvgCDNLatencyICEcast)
		}

		if summary.AverageLatencyMetrics.AvgCrossProtocolLag > 0 {
			fmt.Printf("Cross-Protocol Lag: %.2fs\n", summary.AverageLatencyMetrics.AvgCrossProtocolLag)
		}

		if summary.AverageLatencyMetrics.AvgTimeToFirstByte > 0 {
			fmt.Printf("Avg TTFB:          %.0fms\n", summary.AverageLatencyMetrics.AvgTimeToFirstByte)
		}
	}

	// Print per-broadcast results
	fmt.Printf("\n📋 BROADCAST RESULTS\n")
	fmt.Printf("====================\n")

	for groupName, broadcast := range summary.BroadcastMeasurements {
		status := "✅"
		if broadcast.Error != nil {
			status = "❌"
		} else if broadcast.OverallValidation != nil && broadcast.OverallValidation.OverallHealthScore < 0.7 {
			status = "⚠️"
		}

		fmt.Printf("%s %s", status, groupName)

		if broadcast.Error != nil {
			fmt.Printf(" - ERROR: %v\n", broadcast.Error)
			continue
		}

		if broadcast.OverallValidation != nil {
			fmt.Printf(" (Health: %.1f%%", broadcast.OverallValidation.OverallHealthScore*100)

			if broadcast.LivenessMetrics != nil {
				if broadcast.LivenessMetrics.CDNLatencyHLS != 0 {
					fmt.Printf(", HLS: %.2fs", broadcast.LivenessMetrics.CDNLatencyHLS)
				}
				if broadcast.LivenessMetrics.CDNLatencyICEcast != 0 {
					fmt.Printf(", ICEcast: %.2fs", broadcast.LivenessMetrics.CDNLatencyICEcast)
				}
			}
			fmt.Printf(")\n")

			// Show validation issues if any
			if len(broadcast.OverallValidation.StreamValidityIssues) > 0 {
				for _, issue := range broadcast.OverallValidation.StreamValidityIssues {
					fmt.Printf("    ⚠️  %s\n", issue)
				}
			}
		}
	}

	// Print insights if available
	if len(insights) > 0 {
		fmt.Printf("\n💡 INSIGHTS & RECOMMENDATIONS\n")
		fmt.Printf("=============================\n")
		for i, insight := range insights {
			fmt.Printf("%d. %s\n", i+1, insight)
		}
	}

	fmt.Printf("\n")
}
