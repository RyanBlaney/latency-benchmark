package configs

import (
	"fmt"
	"time"

	"github.com/spf13/viper"
)

// Config represents the application configuration
type Config struct {
	// Application settings
	Verbose      bool   `mapstructure:"verbose"`
	LogLevel     string `mapstructure:"log_level"`
	OutputFormat string `mapstructure:"output_format"`
	ConfigDir    string `mapstructure:"config_dir"`
	DataDir      string `mapstructure:"data_dir"`

	// Test configuration
	Test TestConfig `mapstructure:"test"`

	// Stream configuration
	Stream StreamConfig `mapstructure:"stream"`

	// Audio processing configuration
	Audio AudioConfig `mapstructure:"audio"`

	// Quality thresholds
	Quality QualityConfig `mapstructure:"quality"`

	// Output configuration
	Output OutputConfig `mapstructure:"output"`

	// Regional configuration
	Regions map[string]RegionConfig `mapstructure:"regions"`

	// Test profiles
	Profiles map[string]TestProfile `mapstructure:"profiles"`
}

// TestConfig contains test execution settings
type TestConfig struct {
	Timeout        time.Duration `mapstructure:"timeout"`
	RetryAttempts  int           `mapstructure:"retry_attempts"`
	RetryDelay     time.Duration `mapstructure:"retry_delay"`
	Concurrent     bool          `mapstructure:"concurrent"`
	MaxConcurrency int           `mapstructure:"max_concurrency"`
}

// StreamConfig contains stream handling settings
type StreamConfig struct {
	ConnectionTimeout time.Duration     `mapstructure:"connection_timeout"`
	ReadTimeout       time.Duration     `mapstructure:"read_timeout"`
	BufferSize        int               `mapstructure:"buffer_size"`
	MaxRedirects      int               `mapstructure:"max_redirects"`
	UserAgent         string            `mapstructure:"user_agent"`
	Headers           map[string]string `mapstructure:"headers"`
}

// AudioConfig contains audio processing settings
type AudioConfig struct {
	SampleRate     int           `mapstructure:"sample_rate"`
	Channels       int           `mapstructure:"channels"`
	BufferDuration time.Duration `mapstructure:"buffer_duration"`
	WindowSize     int           `mapstructure:"window_size"`
	Overlap        float64       `mapstructure:"overlap"`
	WindowFunction string        `mapstructure:"window_function"`
	FFTSize        int           `mapstructure:"fft_size"`
	MelBins        int           `mapstructure:"mel_bins"`
}

// QualityConfig contains quality threshold settings
type QualityConfig struct {
	MinSimilarity float64       `mapstructure:"min_similarity"`
	MaxLatency    time.Duration `mapstructure:"max_latency"`
	MinBitrate    int           `mapstructure:"min_bitrate"`
	MaxDropouts   int           `mapstructure:"max_dropouts"`
	BufferHealth  float64       `mapstructure:"buffer_health"`
}

// OutputConfig contains output formatting settings
type OutputConfig struct {
	Precision       int  `mapstructure:"precision"`
	IncludeMetadata bool `mapstructure:"include_metadata"`
	Timestamps      bool `mapstructure:"timestamps"`
	Colors          bool `mapstructure:"colors"`
	Pager           bool `mapstructure:"pager"`
}

// RegionConfig contains region-specific settings
type RegionConfig struct {
	Name     string            `mapstructure:"name"`
	Endpoint string            `mapstructure:"endpoint"`
	Location string            `mapstructure:"location"`
	Headers  map[string]string `mapstructure:"headers"`
	Enabled  bool              `mapstructure:"enabled"`
}

// TestProfile contains a complete test configuration
type TestProfile struct {
	Name        string            `mapstructure:"name"`
	Description string            `mapstructure:"description"`
	Streams     []StreamEndpoint  `mapstructure:"streams"`
	Regions     []string          `mapstructure:"regions"`
	Duration    time.Duration     `mapstructure:"duration"`
	Metrics     []string          `mapstructure:"metrics"`
	Thresholds  QualityConfig     `mapstructure:"thresholds"`
	Tags        map[string]string `mapstructure:"tags"`
}

// StreamEndpoint contains stream endpoint configuration
type StreamEndpoint struct {
	Name      string            `mapstructure:"name"`
	URL       string            `mapstructure:"url"`
	Type      string            `mapstructure:"type"`
	Reference string            `mapstructure:"reference"`
	Headers   map[string]string `mapstructure:"headers"`
	Expected  StreamMetadata    `mapstructure:"expected"`
}

// StreamMetadata contains expected stream metadata
type StreamMetadata struct {
	Bitrate    int    `mapstructure:"bitrate"`
	SampleRate int    `mapstructure:"sample_rate"`
	Channels   int    `mapstructure:"channels"`
	Codec      string `mapstructure:"codec"`
	Format     string `mapstructure:"format"`
}

// LoadConfig loads configuration from viper
func LoadConfig() (*Config, error) {
	config := &Config{}

	if err := viper.Unmarshal(config); err != nil {
		return nil, fmt.Errorf("unable to decode configuration: %w", err)
	}

	return config, nil
}

// ValidateConfig validates the configuration
func ValidateConfig(config *Config) error {
	if config.Test.Timeout <= 0 {
		return fmt.Errorf("test timeout must be positive")
	}

	if config.Test.RetryAttempts < 0 {
		return fmt.Errorf("retry attempts cannot be negative")
	}

	if config.Audio.SampleRate <= 0 {
		return fmt.Errorf("audio sample rate must be positive")
	}

	if config.Audio.Channels <= 0 {
		return fmt.Errorf("audio channels must be positive")
	}

	if config.Quality.MinSimilarity < 0 || config.Quality.MinSimilarity > 1 {
		return fmt.Errorf("minimum similarity must be between 0 and 1")
	}

	return nil
}

