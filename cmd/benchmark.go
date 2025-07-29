package cmd

import (
	"context"
	"fmt"
	"time"

	"github.com/spf13/cobra"
	"github.com/tunein/cdn-benchmark-cli/internal/app"
)

var (
	benchmarkConfigFile          string
	benchmarkBroadcastConfigFile string
	benchmarkOutputFile          string
	benchmarkOutputFormat        string
	benchmarkTimeout             time.Duration
	benchmarkSegmentDuration     time.Duration
	benchmarkMaxConcurrent       int
	benchmarkVerbose             bool
	benchmarkQuiet               bool
	benchmarkDetailedAnalysis    bool
	benchmarkSkipFingerprint     bool
)

var benchmarkCmd = &cobra.Command{
	Use:   "benchmark",
	Short: "Run comprehensive CDN latency benchmark",
	Long: `Run a comprehensive benchmark of CDN latency across multiple broadcast groups.

This command measures end-to-end latency by:
1. Extracting audio segments from HLS and ICEcast streams (both source and CDN)
2. Generating audio fingerprints optimized for the content type
3. Measuring temporal alignment between stream pairs to calculate latency
4. Validating stream quality and fingerprint similarity
5. Calculating comprehensive latency and performance metrics

The benchmark requires a configuration file specifying broadcast groups with their
respective stream endpoints.`,
	Example: `  # Run benchmark with broadcast configuration
  cdn-benchmark-cli benchmark --broadcasts broadcasts.yaml

  # Run with custom app config and broadcast config
  cdn-benchmark-cli benchmark \
    --config app-config.yaml \
    --broadcasts broadcasts.yaml \
    --timeout 30m \
    --output results.json \
    --format json

  # Quick alignment-only benchmark
  cdn-benchmark-cli benchmark \
    --broadcasts broadcasts.yaml \
    --skip-fingerprint \
    --segment-duration 15s`,
	RunE: runBenchmark,
}

func init() {
	rootCmd.AddCommand(benchmarkCmd)

	// Required flags
	benchmarkCmd.Flags().StringVarP(&benchmarkConfigFile, "config", "c", "", "application configuration file (optional)")
	benchmarkCmd.Flags().StringVarP(&benchmarkBroadcastConfigFile, "broadcasts", "b", "", "broadcast groups configuration file (required)")
	benchmarkCmd.MarkFlagRequired("broadcasts")

	// Output options
	benchmarkCmd.Flags().StringVarP(&benchmarkOutputFile, "output", "o", "", "output file (default: stdout)")
	benchmarkCmd.Flags().StringVar(&benchmarkOutputFormat, "format", "json", "output format (json, yaml, csv, table)")

	// Performance options
	benchmarkCmd.Flags().DurationVar(&benchmarkTimeout, "timeout", time.Minute*30, "overall benchmark timeout")
	benchmarkCmd.Flags().DurationVarP(&benchmarkSegmentDuration, "segment-duration", "t", time.Second*90, "audio segment duration")
	benchmarkCmd.Flags().IntVar(&benchmarkMaxConcurrent, "max-concurrent", 3, "maximum concurrent broadcast measurements")

	// Analysis options
	benchmarkCmd.Flags().BoolVar(&benchmarkDetailedAnalysis, "detailed-analysis", false, "enable detailed feature analysis")
	benchmarkCmd.Flags().BoolVar(&benchmarkSkipFingerprint, "skip-fingerprint", false, "skip fingerprint comparison")

	// Logging options
	benchmarkCmd.Flags().BoolVarP(&benchmarkVerbose, "verbose", "v", false, "verbose output")
	benchmarkCmd.Flags().BoolVarP(&benchmarkQuiet, "quiet", "q", false, "quiet output (errors only)")

	// Add subcommands
	benchmarkCmd.AddCommand(generateConfigCmd)
	benchmarkCmd.AddCommand(generateBroadcastsCmd)
	benchmarkCmd.AddCommand(validateConfigCmd)
	benchmarkCmd.AddCommand(validateBroadcastsCmd)
}

func runBenchmark(cmd *cobra.Command, args []string) error {
	// Create application context
	appCtx := &app.Context{
		ConfigFile:          benchmarkConfigFile,
		BroadcastConfigFile: benchmarkBroadcastConfigFile,
		OutputFile:          benchmarkOutputFile,
		OutputFormat:        benchmarkOutputFormat,
		Timeout:             benchmarkTimeout,
		SegmentDuration:     benchmarkSegmentDuration,
		MaxConcurrent:       benchmarkMaxConcurrent,
		Verbose:             benchmarkVerbose,
		Quiet:               benchmarkQuiet,
		DetailedAnalysis:    benchmarkDetailedAnalysis,
		SkipFingerprint:     benchmarkSkipFingerprint,
	}

	// Initialize application
	benchmarkApp, err := app.NewBenchmarkApp(appCtx)
	if err != nil {
		return fmt.Errorf("failed to initialize benchmark app: %w", err)
	}

	// Run benchmark
	ctx, cancel := context.WithTimeout(context.Background(), benchmarkTimeout)
	defer cancel()

	return benchmarkApp.Run(ctx)
}

// Helper commands

var generateConfigCmd = &cobra.Command{
	Use:   "generate-config [output-file]",
	Short: "Generate example application configuration",
	Long:  `Generate an example application configuration file with benchmark settings.`,
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		outputFile := "app-config.yaml"
		if len(args) > 0 {
			outputFile = args[0]
		}

		return app.GenerateExampleConfig(outputFile)
	},
}

var generateBroadcastsCmd = &cobra.Command{
	Use:   "generate-broadcasts [output-file]",
	Short: "Generate example broadcast groups configuration",
	Long:  `Generate an example broadcast groups configuration file with sample streams.`,
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		outputFile := "broadcasts.yaml"
		if len(args) > 0 {
			outputFile = args[0]
		}

		return app.GenerateExampleBroadcastConfig(outputFile)
	},
}

var validateConfigCmd = &cobra.Command{
	Use:   "validate-config [config-file]",
	Short: "Validate application configuration file",
	Long:  `Validate an application configuration file without running the benchmark.`,
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		configFile := benchmarkConfigFile
		if len(args) > 0 {
			configFile = args[0]
		}

		if configFile == "" {
			return fmt.Errorf("configuration file is required")
		}

		return app.ValidateConfig(configFile)
	},
}

var validateBroadcastsCmd = &cobra.Command{
	Use:   "validate-broadcasts [broadcasts-file]",
	Short: "Validate broadcast groups configuration file",
	Long:  `Validate a broadcast groups configuration file without running the benchmark.`,
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		configFile := benchmarkBroadcastConfigFile
		if len(args) > 0 {
			configFile = args[0]
		}

		if configFile == "" {
			return fmt.Errorf("broadcast configuration file is required")
		}

		return app.ValidateBroadcastConfig(configFile)
	},
}
