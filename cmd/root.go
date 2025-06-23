package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

var (
	configFile   string
	verbose      bool
	logLevel     string
	outputFormat string
	configDir    string
	dataDir      string
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "cdn-benchmark",
	Short: "TuneIn CDN Performance Benchmark Suite",
	Long: `A comprehensive CDN performance benchmarking tool for TuneIn Radio.
This tool provides automated testing and analysis of CDN performance across
multiple geographic regions with support for HLS and ICEcast streams.

Key features:
- Multi-region performance testing
- HLS and ICEcast stream support  
- Audio quality analysis and fingerprinting
- Latency measurement and statistical analysis
- Integration with DataDog monitoring
- Configurable test profiles and thresholds`,
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		return initializeConfig(cmd)
	},
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
	rootCmd.PersistentFlags().StringVarP(&outputFormat, "output", "o", "table",
		"output format (json, table, csv, yaml)")

	viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose"))
	viper.BindPFlag("log_level", rootCmd.PersistentFlags().Lookup("log-level"))
	viper.BindPFlag("output_format", rootCmd.PersistentFlags().Lookup("output"))
	viper.BindPFlag("config_dir", rootCmd.PersistentFlags().Lookup("config-dir"))
	viper.BindPFlag("data_dir", rootCmd.PersistentFlags().Lookup("data-dir"))
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
	// Bind all flags to viper
	return bindFlags(cmd, viper.GetViper())
}

// bindFlags binds each cobra flag to its associated viper configuration
func bindFlags(cmd *cobra.Command, v *viper.Viper) error {
	var lastErr error

	cmd.Flags().VisitAll(func(f *pflag.Flag) {
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

// setDefaults sets default configuration values
func setDefaults() {
	// Application defaults
	viper.SetDefault("verbose", false)
	viper.SetDefault("log_level", "info")
	viper.SetDefault("output_format", "table")

	// Directory defaults
	home, _ := os.UserHomeDir()
	viper.SetDefault("config_dir", filepath.Join(home, ".config", "cdn-benchmark"))
	viper.SetDefault("data_dir", filepath.Join(home, ".local", "share", "cdn-benchmark"))

	// Test defaults
	viper.SetDefault("test.timeout", "30s")
	viper.SetDefault("test.retry_attempts", 3)
	viper.SetDefault("test.retry_delay", "5s")

	// Stream defaults
	viper.SetDefault("stream.connection_timeout", "10s")
	viper.SetDefault("stream.read_timeout", "30s")
	viper.SetDefault("stream.buffer_size", 8192)
	viper.SetDefault("stream.max_redirects", 3)

	// Audio processing defaults
	viper.SetDefault("audio.sample_rate", 44100)
	viper.SetDefault("audio.channels", 2)
	viper.SetDefault("audio.buffer_duration", "1s")
	viper.SetDefault("audio.window_size", 2048)
	viper.SetDefault("audio.overlap", 0.5)

	// Quality thresholds
	viper.SetDefault("quality.min_similarity", 0.95)
	viper.SetDefault("quality.max_latency", "5s")
	viper.SetDefault("quality.min_bitrate", 128)

	// Output defaults
	viper.SetDefault("output.precision", 3)
	viper.SetDefault("output.include_metadata", true)
	viper.SetDefault("output.timestamps", true)
}

// GetConfig returns the current viper instance
func GetConfig() *viper.Viper {
	return viper.GetViper()
}
