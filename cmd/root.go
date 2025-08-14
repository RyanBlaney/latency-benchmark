package cmd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/RyanBlaney/latency-benchmark/configs"
	"github.com/RyanBlaney/latency-benchmark/internal/app"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

var (
	configFile     string
	verbose        bool
	logLevel       string
	outputFormat   string
	configDir      string
	dataDir        string
	sampleDuration time.Duration

	// Benchmark-specific flags
	benchmarkBroadcastConfigFile string
	benchmarkOutputFile          string
	benchmarkTimeout             time.Duration
	benchmarkSegmentDuration     time.Duration
	benchmarkMaxConcurrent       int
	benchmarkQuiet               bool
	benchmarkDetailedAnalysis    bool
	benchmarkSkipFingerprint     bool
	benchmarkBroadcastIndex      int
)

const (
	ColorReset  = "\033[0m"
	ColorBold   = "\033[1m"
	ColorRed    = "\033[31m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorBlue   = "\033[34m"
	ColorPurple = "\033[35m"
	ColorCyan   = "\033[36m"
	ColorWhite  = "\033[37m"
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "cdn-benchmark-cli",
	Short: "Run comprehensive CDN latency benchmark",
	Long: `Run a comprehensive benchmark of CDN latency across multiple broadcast groups.

This command measures end-to-end latency by:
1. Extracting audio segments from HLS and ICEcast streams (both source and CDN)
2. Generating audio fingerprints optimized for the content type
3. Measuring temporal alignment between stream pairs to calculate latency
4. Validating stream quality and fingerprint similarity
5. Calculating comprehensive latency and performance metrics

The benchmark requires a configuration file specifying broadcast groups with their
respective stream endpoints.

Key features:
- HLS and ICEcast stream support  
- Audio quality analysis and fingerprinting
- Latency measurement and statistical analysis
- Integration with DataDog monitoring
- Configurable test profiles and thresholds`,
	Example: `  # Run benchmark with broadcast configuration
  cdn-benchmark-cli --broadcasts broadcasts.yaml

  # Run with custom app config and broadcast config
  cdn-benchmark-cli \
    --config app-config.yaml \
    --broadcasts broadcasts.yaml \
    --index 0 \
    --timeout 30m \
    --output results.json \
    --format json

  # Quick alignment-only benchmark
  cdn-benchmark-cli \
    --broadcasts broadcasts.yaml \
    --skip-fingerprint \
    --segment-duration 15s`,
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		return initializeConfig(cmd)
	},
	RunE: runBenchmark,
}

// Execute adds all child commands to the root command and sets flags appropriately
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func init() {
	cobra.OnInitialize(initConfig)

	// Global persistent flags
	rootCmd.PersistentFlags().StringVar(&configDir, "config-dir", "",
		"config directory (default is $HOME/.config/cdn-benchmark)")

	rootCmd.PersistentFlags().StringVar(&configFile, "config", "",
		"config file (default is $HOME/.config/cdn-benchmark/cdn-benchmark.yaml)")

	rootCmd.PersistentFlags().StringVar(&dataDir, "data-dir", "",
		"data directory (default is $HOME/.local/share/cdn-benchmark)")

	// Output and logging flags
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false,
		"verbose output")
	rootCmd.PersistentFlags().StringVar(&logLevel, "log-level", "info",
		"log level (debug, info, error)")
	rootCmd.PersistentFlags().StringVarP(&outputFormat, "output", "o", "json",
		"output format (json, table, csv, yaml)")

	// Benchmark-specific flags
	rootCmd.Flags().StringVarP(&benchmarkBroadcastConfigFile, "broadcasts", "b", "", "broadcast groups configuration file (required)")
	rootCmd.MarkFlagRequired("broadcasts")

	rootCmd.Flags().StringVar(&benchmarkOutputFile, "output-file", "", "output file (default: stdout)")
	rootCmd.Flags().DurationVar(&benchmarkTimeout, "timeout", time.Minute*30, "overall benchmark timeout")
	rootCmd.Flags().DurationVarP(&benchmarkSegmentDuration, "segment-duration", "t", 0, "audio segment duration")
	rootCmd.Flags().IntVar(&benchmarkMaxConcurrent, "max-concurrent", 3, "maximum concurrent broadcast measurements")

	// Analysis options
	rootCmd.Flags().BoolVar(&benchmarkDetailedAnalysis, "detailed-analysis", false, "enable detailed feature analysis")
	rootCmd.Flags().BoolVar(&benchmarkSkipFingerprint, "skip-fingerprint", false, "skip fingerprint comparison")

	// Additional logging options
	rootCmd.Flags().BoolVarP(&benchmarkQuiet, "quiet", "q", false, "quiet output (errors only)")
	rootCmd.Flags().IntVarP(&benchmarkBroadcastIndex, "index", "i", 0, "job index to select broadcast")

	// Bind flags to viper with correct keys to avoid namespace conflicts
	viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose"))
	viper.BindPFlag("log_level", rootCmd.PersistentFlags().Lookup("log-level"))
	viper.BindPFlag("output_format", rootCmd.PersistentFlags().Lookup("output")) // Bind --output flag to output_format config
	viper.BindPFlag("config_dir", rootCmd.PersistentFlags().Lookup("config-dir"))
	viper.BindPFlag("data_dir", rootCmd.PersistentFlags().Lookup("data-dir"))
}

func runBenchmark(cmd *cobra.Command, args []string) error {
	// Determine the output format - prioritize command line flag if explicitly set, then config file
	finalOutputFormat := "json" // Default to JSON

	// Check if output flag was explicitly set by user
	outputFlagChanged := cmd.Flags().Changed("output")

	if outputFlagChanged {
		// User explicitly set the output format via flag
		finalOutputFormat = outputFormat
	} else if viper.IsSet("output_format") {
		// Use config file setting
		finalOutputFormat = viper.GetString("output_format")
	} else if viper.IsSet("benchmark.output_format") {
		// Check benchmark-specific output format in config
		finalOutputFormat = viper.GetString("benchmark.output_format")
	}

	// Create application context
	appCtx := &app.Context{
		ConfigFile:          configFile,
		BroadcastConfigFile: benchmarkBroadcastConfigFile,
		OutputFile:          benchmarkOutputFile,
		OutputFormat:        finalOutputFormat,
		Timeout:             benchmarkTimeout,
		SegmentDuration:     benchmarkSegmentDuration,
		MaxConcurrent:       benchmarkMaxConcurrent,
		Verbose:             verbose, // Use the global verbose flag
		Quiet:               benchmarkQuiet,
		DetailedAnalysis:    benchmarkDetailedAnalysis,
		SkipFingerprint:     benchmarkSkipFingerprint,
		BroadcastIndex:      benchmarkBroadcastIndex,
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

// initConfig reads in config file and ENV variables if set
func initConfig() {
	if configFile != "" {
		// Use config file from the flag
		viper.SetConfigFile(configFile)
	} else {
		// Find home directory
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error finding home directory: %v\n", err)
			os.Exit(1)
		}

		// Search config in home directory and /etc
		viper.AddConfigPath(home)
		viper.AddConfigPath(filepath.Join(home, ".config", "cdn-benchmark"))
		viper.AddConfigPath("/etc/cdn-benchmark")
		viper.AddConfigPath("./configs")
		viper.SetConfigName("cdn-benchmark")
		viper.SetConfigType("yaml")
	}

	// Environment variable support
	viper.SetEnvPrefix("CDN_BENCHMARK")
	viper.SetEnvKeyReplacer(strings.NewReplacer("-", "_", ".", "_"))
	viper.AutomaticEnv()

	// Set default values
	setDefaults()

	// If a config file is found, read it in
	if err := viper.ReadInConfig(); err == nil {
		if viper.GetBool("verbose") {
			fmt.Fprintf(os.Stderr, "Using config file: %s\n", viper.ConfigFileUsed())
		}
	}
}

// initializeConfig initializes configuration after flags are parsed
func initializeConfig(cmd *cobra.Command) error {
	// Bind all flags to viper, but skip the ones we've already handled manually
	return bindFlags(cmd, viper.GetViper())
}

// bindFlags binds each cobra flag to its associated viper configuration
func bindFlags(cmd *cobra.Command, v *viper.Viper) error {
	var lastErr error

	// List of flags that have already been bound manually in init()
	// to avoid conflicts with configuration struct fields
	skipFlags := map[string]bool{
		"verbose":    true,
		"log-level":  true,
		"output":     true, // Skip this one - it's bound to output_format
		"config-dir": true,
		"data-dir":   true,
	}

	cmd.Flags().VisitAll(func(f *pflag.Flag) {
		// Skip flags that were already bound manually
		if skipFlags[f.Name] {
			return
		}

		// Environment variable name
		envVarSuffix := strings.ToUpper(strings.ReplaceAll(f.Name, "-", "_"))

		// Apply the viper config value to the flag when the flag is not set and viper has a value
		if !f.Changed && v.IsSet(f.Name) {
			val := v.Get(f.Name)
			if err := cmd.Flags().Set(f.Name, fmt.Sprintf("%v", val)); err != nil {
				lastErr = err
			}
		}

		// Bind the flag to viper
		if err := v.BindPFlag(f.Name, f); err != nil {
			lastErr = err
		}

		// Bind to environment variable
		if err := v.BindEnv(f.Name, "CDN_BENCHMARK_"+envVarSuffix); err != nil {
			lastErr = err
		}
	})

	return lastErr
}

// setDefaults sets default configuration values using the configs package
func setDefaults() {
	defaultConfig := configs.GetDefaultConfig()

	// Application defaults
	viper.SetDefault("verbose", defaultConfig.Verbose)
	viper.SetDefault("log_level", defaultConfig.LogLevel)
	viper.SetDefault("output_format", defaultConfig.OutputFormat)
	viper.SetDefault("config_dir", defaultConfig.ConfigDir)
	viper.SetDefault("data_dir", defaultConfig.DataDir)

	// Test defaults
	viper.SetDefault("test.timeout", defaultConfig.Test.Timeout)
	viper.SetDefault("test.retry_attempts", defaultConfig.Test.RetryAttempts)
	viper.SetDefault("test.retry_delay", defaultConfig.Test.RetryDelay)
	viper.SetDefault("test.concurrent", defaultConfig.Test.Concurrent)
	viper.SetDefault("test.max_concurrency", defaultConfig.Test.MaxConcurrency)

	// Stream defaults
	viper.SetDefault("stream.connection_timeout", defaultConfig.Stream.ConnectionTimeout)
	viper.SetDefault("stream.read_timeout", defaultConfig.Stream.ReadTimeout)
	viper.SetDefault("stream.buffer_size", defaultConfig.Stream.BufferSize)
	viper.SetDefault("stream.max_redirects", defaultConfig.Stream.MaxRedirects)
	viper.SetDefault("stream.user_agent", defaultConfig.Stream.UserAgent)

	// Set default headers if any
	for key, value := range defaultConfig.Stream.Headers {
		viper.SetDefault(fmt.Sprintf("stream.headers.%s", key), value)
	}

	// Audio processing defaults
	viper.SetDefault("audio.sample_rate", defaultConfig.Audio.SampleRate)
	viper.SetDefault("audio.channels", defaultConfig.Audio.Channels)
	viper.SetDefault("audio.buffer_duration", defaultConfig.Audio.BufferDuration)
	viper.SetDefault("audio.window_size", defaultConfig.Audio.WindowSize)
	viper.SetDefault("audio.overlap", defaultConfig.Audio.Overlap)
	viper.SetDefault("audio.window_function", defaultConfig.Audio.WindowFunction)
	viper.SetDefault("audio.fft_size", defaultConfig.Audio.FFTSize)
	viper.SetDefault("audio.mel_bins", defaultConfig.Audio.MelBins)

	// Quality thresholds
	viper.SetDefault("quality.min_similarity", defaultConfig.Quality.MinSimilarity)
	viper.SetDefault("quality.max_latency", defaultConfig.Quality.MaxLatency)
	viper.SetDefault("quality.min_bitrate", defaultConfig.Quality.MinBitrate)
	viper.SetDefault("quality.max_dropouts", defaultConfig.Quality.MaxDropouts)
	viper.SetDefault("quality.buffer_health", defaultConfig.Quality.BufferHealth)

	// Output defaults
	viper.SetDefault("output.precision", defaultConfig.Output.Precision)
	viper.SetDefault("output.include_metadata", defaultConfig.Output.IncludeMetadata)
	viper.SetDefault("output.timestamps", defaultConfig.Output.Timestamps)
	viper.SetDefault("output.colors", defaultConfig.Output.Colors)
	viper.SetDefault("output.pager", defaultConfig.Output.Pager)

	// Regional defaults
	for regionKey, regionConfig := range defaultConfig.Regions {
		viper.SetDefault(fmt.Sprintf("regions.%s.name", regionKey), regionConfig.Name)
		viper.SetDefault(fmt.Sprintf("regions.%s.endpoint", regionKey), regionConfig.Endpoint)
		viper.SetDefault(fmt.Sprintf("regions.%s.location", regionKey), regionConfig.Location)
		viper.SetDefault(fmt.Sprintf("regions.%s.enabled", regionKey), regionConfig.Enabled)

		// Set region headers
		for headerKey, headerValue := range regionConfig.Headers {
			viper.SetDefault(fmt.Sprintf("regions.%s.headers.%s", regionKey, headerKey), headerValue)
		}
	}

	// Test profiles defaults
	for profileKey, profileConfig := range defaultConfig.Profiles {
		viper.SetDefault(fmt.Sprintf("profiles.%s.name", profileKey), profileConfig.Name)
		viper.SetDefault(fmt.Sprintf("profiles.%s.description", profileKey), profileConfig.Description)
		viper.SetDefault(fmt.Sprintf("profiles.%s.duration", profileKey), profileConfig.Duration)
		viper.SetDefault(fmt.Sprintf("profiles.%s.regions", profileKey), profileConfig.Regions)
		viper.SetDefault(fmt.Sprintf("profiles.%s.metrics", profileKey), profileConfig.Metrics)

		// Profile thresholds
		viper.SetDefault(fmt.Sprintf("profiles.%s.thresholds.min_similarity", profileKey), profileConfig.Thresholds.MinSimilarity)
		viper.SetDefault(fmt.Sprintf("profiles.%s.thresholds.max_latency", profileKey), profileConfig.Thresholds.MaxLatency)
		viper.SetDefault(fmt.Sprintf("profiles.%s.thresholds.min_bitrate", profileKey), profileConfig.Thresholds.MinBitrate)
		viper.SetDefault(fmt.Sprintf("profiles.%s.thresholds.max_dropouts", profileKey), profileConfig.Thresholds.MaxDropouts)
		viper.SetDefault(fmt.Sprintf("profiles.%s.thresholds.buffer_health", profileKey), profileConfig.Thresholds.BufferHealth)

		// Profile tags
		for tagKey, tagValue := range profileConfig.Tags {
			viper.SetDefault(fmt.Sprintf("profiles.%s.tags.%s", profileKey, tagKey), tagValue)
		}
	}
}

// GetConfig returns the current viper instance
func GetConfig() *viper.Viper {
	return viper.GetViper()
}

// LoadAppConfig loads and validates the application configuration
func LoadAppConfig() (*configs.Config, error) {
	config, err := configs.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load configuration: %w", err)
	}

	if err := configs.ValidateConfig(config); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return config, nil
}
