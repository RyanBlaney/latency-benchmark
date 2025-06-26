// configs/config.go
package configs

import (
	"fmt"
	"time"

	"github.com/spf13/viper"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/hls"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/icecast"
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

	// Stream-specific configurations
	HLS     HLSConfig     `mapstructure:"hls"`
	ICEcast ICEcastConfig `mapstructure:"icecast"`
}

// HLSConfig contains HLS-specific configuration
type HLSConfig struct {
	Parser            HLSParserConfig            `mapstructure:"parser"`
	MetadataExtractor HLSMetadataExtractorConfig `mapstructure:"metadata_extractor"`
	Detection         HLSDetectionConfig         `mapstructure:"detection"`
	HTTP              HLSHTTPConfig              `mapstructure:"http"`
	Audio             HLSAudioConfig             `mapstructure:"audio"`
}

// HLSParserConfig holds configuration for M3U8 parsing
type HLSParserConfig struct {
	StrictMode         bool              `mapstructure:"strict_mode"`
	MaxSegmentAnalysis int               `mapstructure:"max_segment_analysis"`
	CustomTagHandlers  map[string]string `mapstructure:"custom_tag_handlers"`
	IgnoreUnknownTags  bool              `mapstructure:"ignore_unknown_tags"`
	ValidateURIs       bool              `mapstructure:"validate_uris"`
}

// HLSMetadataExtractorConfig holds configuration for metadata extraction
type HLSMetadataExtractorConfig struct {
	EnableURLPatterns     bool                   `mapstructure:"enable_url_patterns"`
	EnableHeaderMappings  bool                   `mapstructure:"enable_header_mappings"`
	EnableSegmentAnalysis bool                   `mapstructure:"enable_segment_analysis"`
	DefaultValues         map[string]interface{} `mapstructure:"default_values"`
}

// HLSDetectionConfig holds configuration for stream detection
type HLSDetectionConfig struct {
	URLPatterns     []string `mapstructure:"url_patterns"`
	ContentTypes    []string `mapstructure:"content_types"`
	RequiredHeaders []string `mapstructure:"required_headers"`
	TimeoutSeconds  int      `mapstructure:"timeout_seconds"`
}

// HLSHTTPConfig holds HTTP-related configuration for HLS
type HLSHTTPConfig struct {
	UserAgent         string            `mapstructure:"user_agent"`
	AcceptHeader      string            `mapstructure:"accept_header"`
	ConnectionTimeout time.Duration     `mapstructure:"connection_timeout"`
	ReadTimeout       time.Duration     `mapstructure:"read_timeout"`
	MaxRedirects      int               `mapstructure:"max_redirects"`
	CustomHeaders     map[string]string `mapstructure:"custom_headers"`
	BufferSize        int               `mapstructure:"buffer_size"`
}

// HLSAudioConfig holds audio-specific configuration for HLS
type HLSAudioConfig struct {
	SampleDuration  time.Duration `mapstructure:"sample_duration"`
	BufferDuration  time.Duration `mapstructure:"buffer_duration"`
	MaxSegments     int           `mapstructure:"max_segments"`
	FollowLive      bool          `mapstructure:"follow_live"`
	AnalyzeSegments bool          `mapstructure:"analyze_segments"`
}

// ICEcastConfig contains ICEcast-specific configuration
type ICEcastConfig struct {
	MetadataExtractor ICEcastMetadataExtractorConfig `mapstructure:"metadata_extractor"`
	Detection         ICEcastDetectionConfig         `mapstructure:"detection"`
	HTTP              ICEcastHTTPConfig              `mapstructure:"http"`
	Audio             ICEcastAudioConfig             `mapstructure:"audio"`
}

// ICEcastMetadataExtractorConfig holds configuration for metadata extraction
type ICEcastMetadataExtractorConfig struct {
	EnableHeaderMappings bool                   `mapstructure:"enable_header_mappings"`
	EnableICYMetadata    bool                   `mapstructure:"enable_icy_metadata"`
	DefaultValues        map[string]interface{} `mapstructure:"default_values"`
	ICYMetadataTimeout   time.Duration          `mapstructure:"icy_metadata_timeout"`
}

// ICEcastDetectionConfig holds configuration for stream detection
type ICEcastDetectionConfig struct {
	URLPatterns     []string `mapstructure:"url_patterns"`
	ContentTypes    []string `mapstructure:"content_types"`
	RequiredHeaders []string `mapstructure:"required_headers"`
	TimeoutSeconds  int      `mapstructure:"timeout_seconds"`
	CommonPorts     []string `mapstructure:"common_ports"`
}

// ICEcastHTTPConfig holds HTTP-related configuration for ICEcast
type ICEcastHTTPConfig struct {
	UserAgent         string            `mapstructure:"user_agent"`
	AcceptHeader      string            `mapstructure:"accept_header"`
	ConnectionTimeout time.Duration     `mapstructure:"connection_timeout"`
	ReadTimeout       time.Duration     `mapstructure:"read_timeout"`
	MaxRedirects      int               `mapstructure:"max_redirects"`
	CustomHeaders     map[string]string `mapstructure:"custom_headers"`
	RequestICYMeta    bool              `mapstructure:"request_icy_meta"`
}

// ICEcastAudioConfig holds audio-specific configuration for ICEcast
type ICEcastAudioConfig struct {
	BufferSize       int           `mapstructure:"buffer_size"`
	SampleDuration   time.Duration `mapstructure:"sample_duration"`
	MaxReadAttempts  int           `mapstructure:"max_read_attempts"`
	ReadTimeout      time.Duration `mapstructure:"read_timeout"`
	HandleICYMeta    bool          `mapstructure:"handle_icy_meta"`
	MetadataInterval int           `mapstructure:"metadata_interval"`
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

// ToHLSConfig converts the main config to an HLS config
func (c *Config) ToHLSConfig() *hls.Config {
	// Create a map representation for the ConfigFromAppConfig function
	configMap := map[string]interface{}{
		"stream": map[string]interface{}{
			"user_agent":         c.Stream.UserAgent,
			"connection_timeout": c.Stream.ConnectionTimeout,
			"read_timeout":       c.Stream.ReadTimeout,
			"max_redirects":      c.Stream.MaxRedirects,
			"buffer_size":        c.Stream.BufferSize,
			"headers":            c.Stream.Headers,
		},
		"audio": map[string]interface{}{
			"buffer_duration": c.Audio.BufferDuration,
			"sample_rate":     c.Audio.SampleRate,
			"channels":        c.Audio.Channels,
		},
		"hls": map[string]interface{}{
			"parser": map[string]interface{}{
				"strict_mode":          c.HLS.Parser.StrictMode,
				"max_segment_analysis": c.HLS.Parser.MaxSegmentAnalysis,
				"ignore_unknown_tags":  c.HLS.Parser.IgnoreUnknownTags,
				"validate_uris":        c.HLS.Parser.ValidateURIs,
				"custom_tag_handlers":  c.HLS.Parser.CustomTagHandlers,
			},
			"metadata_extractor": map[string]interface{}{
				"enable_url_patterns":     c.HLS.MetadataExtractor.EnableURLPatterns,
				"enable_header_mappings":  c.HLS.MetadataExtractor.EnableHeaderMappings,
				"enable_segment_analysis": c.HLS.MetadataExtractor.EnableSegmentAnalysis,
				"default_values":          c.HLS.MetadataExtractor.DefaultValues,
			},
			"detection": map[string]interface{}{
				"url_patterns":     c.HLS.Detection.URLPatterns,
				"content_types":    c.HLS.Detection.ContentTypes,
				"required_headers": c.HLS.Detection.RequiredHeaders,
				"timeout_seconds":  c.HLS.Detection.TimeoutSeconds,
			},
			"http": map[string]interface{}{
				"user_agent":         c.HLS.HTTP.UserAgent,
				"accept_header":      c.HLS.HTTP.AcceptHeader,
				"connection_timeout": c.HLS.HTTP.ConnectionTimeout,
				"read_timeout":       c.HLS.HTTP.ReadTimeout,
				"max_redirects":      c.HLS.HTTP.MaxRedirects,
				"custom_headers":     c.HLS.HTTP.CustomHeaders,
				"buffer_size":        c.HLS.HTTP.BufferSize,
			},
			"audio": map[string]interface{}{
				"sample_duration":  c.HLS.Audio.SampleDuration,
				"buffer_duration":  c.HLS.Audio.BufferDuration,
				"max_segments":     c.HLS.Audio.MaxSegments,
				"follow_live":      c.HLS.Audio.FollowLive,
				"analyze_segments": c.HLS.Audio.AnalyzeSegments,
			},
		},
	}

	return hls.ConfigFromAppConfig(configMap)
}

// ToICEcastConfig converts the main config to an ICEcast config map
func (c *Config) ToICEcastConfig() *icecast.Config {
	configMap := map[string]interface{}{
		"stream": map[string]interface{}{
			"user_agent":         c.Stream.UserAgent,
			"connection_timeout": c.Stream.ConnectionTimeout,
			"read_timeout":       c.Stream.ReadTimeout,
			"max_redirects":      c.Stream.MaxRedirects,
			"headers":            c.Stream.Headers,
		},
		"audio": map[string]interface{}{
			"buffer_size": c.Audio.BufferDuration,
			"sample_rate": c.Audio.SampleRate,
			"channels":    c.Audio.Channels,
		},
		"icecast": map[string]interface{}{
			"metadata_extractor": map[string]interface{}{
				"enable_header_mappings": c.ICEcast.MetadataExtractor.EnableHeaderMappings,
				"enable_icy_metadata":    c.ICEcast.MetadataExtractor.EnableICYMetadata,
				"default_values":         c.ICEcast.MetadataExtractor.DefaultValues,
				"icy_metadata_timeout":   c.ICEcast.MetadataExtractor.ICYMetadataTimeout,
			},
			"detection": map[string]interface{}{
				"url_patterns":     c.ICEcast.Detection.URLPatterns,
				"content_types":    c.ICEcast.Detection.ContentTypes,
				"required_headers": c.ICEcast.Detection.RequiredHeaders,
				"timeout_seconds":  c.ICEcast.Detection.TimeoutSeconds,
				"common_ports":     c.ICEcast.Detection.CommonPorts,
			},
			"http": map[string]interface{}{
				"user_agent":         c.ICEcast.HTTP.UserAgent,
				"accept_header":      c.ICEcast.HTTP.AcceptHeader,
				"connection_timeout": c.ICEcast.HTTP.ConnectionTimeout,
				"read_timeout":       c.ICEcast.HTTP.ReadTimeout,
				"max_redirects":      c.ICEcast.HTTP.MaxRedirects,
				"custom_headers":     c.ICEcast.HTTP.CustomHeaders,
				"request_icy_meta":   c.ICEcast.HTTP.RequestICYMeta,
			},
			"audio": map[string]interface{}{
				"buffer_size":       c.ICEcast.Audio.BufferSize,
				"sample_duration":   c.ICEcast.Audio.SampleDuration,
				"max_read_attempts": c.ICEcast.Audio.MaxReadAttempts,
				"read_timeout":      c.ICEcast.Audio.ReadTimeout,
				"handle_icy_meta":   c.ICEcast.Audio.HandleICYMeta,
				"metadata_interval": c.ICEcast.Audio.MetadataInterval,
			},
		},
	}
	return icecast.ConfigFromAppConfig(configMap)
}

// LoadConfig loads configuration from viper with proper handling of conflicting keys
// LoadConfig loads configuration from viper with proper handling of conflicting keys
func LoadConfig() (*Config, error) {
	config := &Config{}

	// Create a copy of viper settings to avoid modifying the global instance
	configViper := viper.New()

	// Configure Viper to read from config file
	configViper.SetConfigName("config")
	configViper.SetConfigType("yaml")
	configViper.AddConfigPath("$HOME/.config/cdn-benchmark")
	configViper.AddConfigPath(".")
	configViper.AddConfigPath("./configs")

	// Try to read the config file
	if err := configViper.ReadInConfig(); err != nil {
		// Config file not found is not an error - we'll use defaults
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("error reading config file: %w", err)
		}
		fmt.Printf("No config file found, using defaults\n")
	} else {
		fmt.Printf("Using config file: %s\n", configViper.ConfigFileUsed())
	}

	// Copy all settings from the global viper instance (flags), excluding conflicting flag keys
	conflictingKeys := map[string]bool{
		"output":  true, // conflicts with output config section
		"regions": true, // conflicts with regions config section
	}

	for _, key := range viper.AllKeys() {
		if !conflictingKeys[key] {
			configViper.Set(key, viper.Get(key))
		}
	}

	// Handle the output flag conflict by mapping it to output_format
	if viper.IsSet("output") {
		configViper.Set("output_format", viper.GetString("output"))
	}

	// Handle regions flag conflict by mapping it to benchmark.regions
	if viper.IsSet("regions") {
		configViper.Set("benchmark.regions", viper.GetStringSlice("regions"))
	}

	// Ensure output configuration section exists with defaults if not already set
	if !configViper.IsSet("output.precision") {
		defaultOutput := GetDefaultOutputConfig()
		configViper.Set("output.precision", defaultOutput.Precision)
		configViper.Set("output.include_metadata", defaultOutput.IncludeMetadata)
		configViper.Set("output.timestamps", defaultOutput.Timestamps)
		configViper.Set("output.colors", defaultOutput.Colors)
		configViper.Set("output.pager", defaultOutput.Pager)
	}

	// Ensure regions configuration section exists with defaults if not already set
	if !configViper.IsSet("regions.local") {
		defaultRegions := GetDefaultRegions()
		for regionKey, regionConfig := range defaultRegions {
			configViper.Set(fmt.Sprintf("regions.%s.name", regionKey), regionConfig.Name)
			configViper.Set(fmt.Sprintf("regions.%s.endpoint", regionKey), regionConfig.Endpoint)
			configViper.Set(fmt.Sprintf("regions.%s.location", regionKey), regionConfig.Location)
			configViper.Set(fmt.Sprintf("regions.%s.enabled", regionKey), regionConfig.Enabled)

			// Set region headers
			for headerKey, headerValue := range regionConfig.Headers {
				configViper.Set(fmt.Sprintf("regions.%s.headers.%s", regionKey, headerKey), headerValue)
			}
		}
	}

	// Set HLS defaults if not already configured
	setHLSDefaults(configViper)

	if err := configViper.Unmarshal(config); err != nil {
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

	// Validate HLS configuration
	if config.HLS.HTTP.ConnectionTimeout <= 0 {
		return fmt.Errorf("HLS connection timeout must be positive")
	}

	if config.HLS.HTTP.ReadTimeout <= 0 {
		return fmt.Errorf("HLS read timeout must be positive")
	}

	if config.HLS.HTTP.MaxRedirects < 0 {
		return fmt.Errorf("HLS max redirects cannot be negative")
	}

	if config.HLS.HTTP.BufferSize <= 0 {
		return fmt.Errorf("HLS buffer size must be positive")
	}

	if config.HLS.Audio.SampleDuration <= 0 {
		return fmt.Errorf("HLS sample duration must be positive")
	}

	if config.HLS.Audio.BufferDuration <= 0 {
		return fmt.Errorf("HLS buffer duration must be positive")
	}

	if config.HLS.Audio.MaxSegments <= 0 {
		return fmt.Errorf("HLS max segments must be positive")
	}

	return nil
}
