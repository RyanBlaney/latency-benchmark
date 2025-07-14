package configs

import (
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/viper"
)

// setDefaults sets default configuration values for all components
func setDefaults(v *viper.Viper) {
	// Global stream defaults (apply to all stream types)
	if !v.IsSet("stream.connection_timeout") {
		v.Set("stream.connection_timeout", 10*time.Second)
	}
	if !v.IsSet("stream.read_timeout") {
		v.Set("stream.read_timeout", 30*time.Second)
	}
	if !v.IsSet("stream.buffer_size") {
		v.Set("stream.buffer_size", 8192)
	}
	if !v.IsSet("stream.max_redirects") {
		v.Set("stream.max_redirects", 3)
	}
	if !v.IsSet("stream.user_agent") {
		v.Set("stream.user_agent", "TuneIn-CDN-Benchmark/1.0")
	}
	if !v.IsSet("stream.headers") {
		v.Set("stream.headers", map[string]string{})
	}

	// Global test defaults
	if !v.IsSet("test.timeout") {
		v.Set("test.timeout", 30*time.Second)
	}
	if !v.IsSet("test.retry_attempts") {
		v.Set("test.retry_attempts", 3)
	}
	if !v.IsSet("test.retry_delay") {
		v.Set("test.retry_delay", 5*time.Second)
	}
	if !v.IsSet("test.concurrent") {
		v.Set("test.concurrent", false)
	}
	if !v.IsSet("test.max_concurrency") {
		v.Set("test.max_concurrency", 4)
	}

	// Global audio defaults
	if !v.IsSet("audio.sample_rate") {
		v.Set("audio.sample_rate", 44100)
	}
	if !v.IsSet("audio.channels") {
		v.Set("audio.channels", 2)
	}
	if !v.IsSet("audio.buffer_duration") {
		v.Set("audio.buffer_duration", 1*time.Second)
	}
	if !v.IsSet("audio.window_size") {
		v.Set("audio.window_size", 2048)
	}
	if !v.IsSet("audio.overlap") {
		v.Set("audio.overlap", 0.5)
	}
	if !v.IsSet("audio.window_function") {
		v.Set("audio.window_function", "hann")
	}
	if !v.IsSet("audio.fft_size") {
		v.Set("audio.fft_size", 2048)
	}
	if !v.IsSet("audio.mel_bins") {
		v.Set("audio.mel_bins", 128)
	}

	// Global quality defaults
	if !v.IsSet("quality.min_similarity") {
		v.Set("quality.min_similarity", 0.95)
	}
	if !v.IsSet("quality.max_latency") {
		v.Set("quality.max_latency", 5*time.Second)
	}
	if !v.IsSet("quality.min_bitrate") {
		v.Set("quality.min_bitrate", 128)
	}
	if !v.IsSet("quality.max_dropouts") {
		v.Set("quality.max_dropouts", 3)
	}
	if !v.IsSet("quality.buffer_health") {
		v.Set("quality.buffer_health", 0.8)
	}

	// Global output defaults
	if !v.IsSet("output.precision") {
		v.Set("output.precision", 3)
	}
	if !v.IsSet("output.include_metadata") {
		v.Set("output.include_metadata", true)
	}
	if !v.IsSet("output.timestamps") {
		v.Set("output.timestamps", true)
	}
	if !v.IsSet("output.colors") {
		v.Set("output.colors", true)
	}
	if !v.IsSet("output.pager") {
		v.Set("output.pager", false)
	}

	// Application defaults
	if !v.IsSet("verbose") {
		v.Set("verbose", false)
	}
	if !v.IsSet("log_level") {
		v.Set("log_level", "info")
	}
	if !v.IsSet("output_format") {
		v.Set("output_format", "table")
	}

	// HLS-specific defaults (these override global stream settings for HLS)
	setHLSDefaults(v)

	// ICEcast-specific defaults (these override global stream settings for ICEcast)
	setICEcastDefaults(v)
}

// setHLSDefaults sets HLS-specific configuration defaults
func setHLSDefaults(v *viper.Viper) {
	// HLS parser defaults
	if !v.IsSet("hls.parser.strict_mode") {
		v.Set("hls.parser.strict_mode", false)
	}
	if !v.IsSet("hls.parser.max_segment_analysis") {
		v.Set("hls.parser.max_segment_analysis", 10)
	}
	if !v.IsSet("hls.parser.ignore_unknown_tags") {
		v.Set("hls.parser.ignore_unknown_tags", false)
	}
	if !v.IsSet("hls.parser.validate_uris") {
		v.Set("hls.parser.validate_uris", false)
	}
	if !v.IsSet("hls.parser.custom_tag_handlers") {
		v.Set("hls.parser.custom_tag_handlers", map[string]string{})
	}

	// HLS detection defaults
	if !v.IsSet("hls.detection.url_patterns") {
		v.Set("hls.detection.url_patterns", []string{
			`\.m3u8$`,
			`/playlist\.m3u8`,
			`/master\.m3u8`,
			`/index\.m3u8`,
		})
	}
	if !v.IsSet("hls.detection.content_types") {
		v.Set("hls.detection.content_types", []string{
			"application/vnd.apple.mpegurl",
			"application/x-mpegurl",
			"vnd.apple.mpegurl",
		})
	}
	if !v.IsSet("hls.detection.timeout_seconds") {
		v.Set("hls.detection.timeout_seconds", 5)
	}
	if !v.IsSet("hls.detection.required_headers") {
		v.Set("hls.detection.required_headers", []string{})
	}

	// HLS HTTP defaults (these override global stream settings)
	if !v.IsSet("hls.http.user_agent") {
		v.Set("hls.http.user_agent", "TuneIn-CDN-Benchmark-HLS/1.0")
	}
	if !v.IsSet("hls.http.accept_header") {
		v.Set("hls.http.accept_header", "application/vnd.apple.mpegurl,application/x-mpegurl,text/plain")
	}
	if !v.IsSet("hls.http.connection_timeout") {
		v.Set("hls.http.connection_timeout", 5*time.Second)
	}
	if !v.IsSet("hls.http.read_timeout") {
		v.Set("hls.http.read_timeout", 15*time.Second)
	}
	if !v.IsSet("hls.http.max_redirects") {
		v.Set("hls.http.max_redirects", 5)
	}
	if !v.IsSet("hls.http.buffer_size") {
		v.Set("hls.http.buffer_size", 16384)
	}
	if !v.IsSet("hls.http.custom_headers") {
		v.Set("hls.http.custom_headers", map[string]string{})
	}

	// HLS audio defaults
	if !v.IsSet("hls.audio.sample_duration") {
		v.Set("hls.audio.sample_duration", 30*time.Second)
	}
	if !v.IsSet("hls.audio.buffer_duration") {
		v.Set("hls.audio.buffer_duration", 2*time.Second)
	}
	if !v.IsSet("hls.audio.max_segments") {
		v.Set("hls.audio.max_segments", 10)
	}
	if !v.IsSet("hls.audio.follow_live") {
		v.Set("hls.audio.follow_live", false)
	}
	if !v.IsSet("hls.audio.analyze_segments") {
		v.Set("hls.audio.analyze_segments", false)
	}

	// HLS metadata extractor defaults
	if !v.IsSet("hls.metadata_extractor.enable_url_patterns") {
		v.Set("hls.metadata_extractor.enable_url_patterns", true)
	}
	if !v.IsSet("hls.metadata_extractor.enable_header_mappings") {
		v.Set("hls.metadata_extractor.enable_header_mappings", true)
	}
	if !v.IsSet("hls.metadata_extractor.enable_segment_analysis") {
		v.Set("hls.metadata_extractor.enable_segment_analysis", true)
	}
	if !v.IsSet("hls.metadata_extractor.default_values") {
		v.Set("hls.metadata_extractor.default_values", map[string]any{
			"codec":       "aac",
			"channels":    2,
			"sample_rate": 44100,
		})
	}
}

// setICEcastDefaults sets ICEcast-specific configuration defaults
func setICEcastDefaults(v *viper.Viper) {
	// ICEcast detection defaults
	if !v.IsSet("icecast.detection.url_patterns") {
		v.Set("icecast.detection.url_patterns", []string{
			`\.mp3$`,
			`\.aac$`,
			`\.ogg$`,
			`/stream$`,
			`/listen$`,
			`/audio$`,
			`/radio$`,
			`:8000/`,
			`:8080/`,
		})
	}
	if !v.IsSet("icecast.detection.content_types") {
		v.Set("icecast.detection.content_types", []string{
			"audio/mpeg",
			"audio/mp3",
			"audio/aac",
			"audio/ogg",
			"application/ogg",
		})
	}
	if !v.IsSet("icecast.detection.timeout_seconds") {
		v.Set("icecast.detection.timeout_seconds", 15)
	}
	if !v.IsSet("icecast.detection.required_headers") {
		v.Set("icecast.detection.required_headers", []string{})
	}
	if !v.IsSet("icecast.detection.common_ports") {
		v.Set("icecast.detection.common_ports", []string{"8000", "8080", "8443", "9000"})
	}

	// ICEcast HTTP defaults (these override global stream settings)
	if !v.IsSet("icecast.http.user_agent") {
		v.Set("icecast.http.user_agent", "TuneIn-CDN-Benchmark-ICEcast/1.0")
	}
	if !v.IsSet("icecast.http.accept_header") {
		v.Set("icecast.http.accept_header", "audio/*, application/ogg, */*")
	}
	if !v.IsSet("icecast.http.connection_timeout") {
		v.Set("icecast.http.connection_timeout", 45*time.Second)
	}
	if !v.IsSet("icecast.http.read_timeout") {
		v.Set("icecast.http.read_timeout", 60*time.Second)
	}
	if !v.IsSet("icecast.http.max_redirects") {
		v.Set("icecast.http.max_redirects", 3)
	}
	if !v.IsSet("icecast.http.custom_headers") {
		v.Set("icecast.http.custom_headers", map[string]string{})
	}
	if !v.IsSet("icecast.http.request_icy_meta") {
		v.Set("icecast.http.request_icy_meta", true)
	}

	// ICEcast audio defaults
	if !v.IsSet("icecast.audio.buffer_size") {
		v.Set("icecast.audio.buffer_size", 4096)
	}
	if !v.IsSet("icecast.audio.buffer_duration") {
		v.Set("icecast.audio.buffer_duration", 2*time.Second)
	}
	if !v.IsSet("icecast.audio.sample_duration") {
		v.Set("icecast.audio.sample_duration", 30*time.Second)
	}
	if !v.IsSet("icecast.audio.max_read_attempts") {
		v.Set("icecast.audio.max_read_attempts", 10)
	}
	if !v.IsSet("icecast.audio.read_timeout") {
		v.Set("icecast.audio.read_timeout", 30*time.Second)
	}
	if !v.IsSet("icecast.audio.handle_icy_meta") {
		v.Set("icecast.audio.handle_icy_meta", true)
	}
	if !v.IsSet("icecast.audio.metadata_interval") {
		v.Set("icecast.audio.metadata_interval", 8192)
	}

	// ICEcast metadata extractor defaults
	if !v.IsSet("icecast.metadata_extractor.enable_header_mappings") {
		v.Set("icecast.metadata_extractor.enable_header_mappings", true)
	}
	if !v.IsSet("icecast.metadata_extractor.enable_icy_metadata") {
		v.Set("icecast.metadata_extractor.enable_icy_metadata", true)
	}
	if !v.IsSet("icecast.metadata_extractor.icy_metadata_timeout") {
		v.Set("icecast.metadata_extractor.icy_metadata_timeout", 15*time.Second)
	}
	if !v.IsSet("icecast.metadata_extractor.default_values") {
		v.Set("icecast.metadata_extractor.default_values", map[string]any{
			"codec":       "mp3",
			"channels":    2,
			"sample_rate": 44100,
		})
	}
}

// GetDefaultConfig returns a Config struct with all default values set
func GetDefaultConfig() *Config {
	home, _ := os.UserHomeDir()

	return &Config{
		// Application settings defaults
		Verbose:      false,
		LogLevel:     "info",
		OutputFormat: "table",
		ConfigDir:    filepath.Join(home, ".config", "cdn-benchmark"),
		DataDir:      filepath.Join(home, ".local", "share", "cdn-benchmark"),

		// Test configuration defaults
		Test: GetDefaultTestConfig(),

		// Stream configuration defaults
		Stream: GetDefaultStreamConfig(),

		// Audio processing configuration defaults
		Audio: GetDefaultAudioConfig(),

		// Quality thresholds defaults
		Quality: GetDefaultQualityConfig(),

		// Output configuration defaults
		Output: GetDefaultOutputConfig(),

		// Regional configuration defaults
		Regions: GetDefaultRegions(),

		// Test profiles defaults
		Profiles: GetDefaultProfiles(),
	}
}

// GetDefaultTestConfig returns default test execution settings
func GetDefaultTestConfig() TestConfig {
	return TestConfig{
		Timeout:        30 * time.Second,
		RetryAttempts:  3,
		RetryDelay:     5 * time.Second,
		Concurrent:     false,
		MaxConcurrency: 4,
	}
}

// GetDefaultStreamConfig returns default stream handling settings
func GetDefaultStreamConfig() StreamConfig {
	return StreamConfig{
		ConnectionTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		BufferSize:        8192,
		MaxRedirects:      3,
		UserAgent:         "TuneIn-CDN-Benchmark/1.0",
		Headers:           make(map[string]string),
	}
}

// GetDefaultAudioConfig returns default audio processing settings
func GetDefaultAudioConfig() AudioConfig {
	return AudioConfig{
		SampleRate:     44100,
		Channels:       2,
		BufferDuration: 1 * time.Second,
		WindowSize:     2048,
		Overlap:        0.5,
		WindowFunction: "hann",
		FFTSize:        2048,
		MelBins:        128,
	}
}

// GetDefaultQualityConfig returns default quality threshold settings
func GetDefaultQualityConfig() QualityConfig {
	return QualityConfig{
		MinSimilarity: 0.95,
		MaxLatency:    5 * time.Second,
		MinBitrate:    128,
		MaxDropouts:   3,
		BufferHealth:  0.8,
	}
}

// GetDefaultOutputConfig returns default output formatting settings
func GetDefaultOutputConfig() OutputConfig {
	return OutputConfig{
		Precision:       3,
		IncludeMetadata: true,
		Timestamps:      true,
		Colors:          true,
		Pager:           false,
	}
}

// GetDefaultRegions returns default regional configuration
func GetDefaultRegions() map[string]RegionConfig {
	return map[string]RegionConfig{
		"local": {
			Name:     "Local Development",
			Endpoint: "http://localhost:8080",
			Location: "localhost",
			Headers:  make(map[string]string),
			Enabled:  true,
		},
		"us-west-1": {
			Name:     "US West 1 (N. California)",
			Endpoint: "https://us-west-1.tunein.com",
			Location: "San Francisco, CA",
			Headers: map[string]string{
				"X-Region":     "us-west-1",
				"X-Datacenter": "sfo",
			},
			Enabled: false, // Disabled by default for production regions
		},
		"us-east-1": {
			Name:     "US East 1 (N. Virginia)",
			Endpoint: "https://us-east-1.tunein.com",
			Location: "Ashburn, VA",
			Headers: map[string]string{
				"X-Region":     "us-east-1",
				"X-Datacenter": "iad",
			},
			Enabled: false,
		},
		"eu-west-1": {
			Name:     "EU West 1 (Ireland)",
			Endpoint: "https://eu-west-1.tunein.com",
			Location: "Dublin, Ireland",
			Headers: map[string]string{
				"X-Region":     "eu-west-1",
				"X-Datacenter": "dub",
			},
			Enabled: false,
		},
	}
}

// GetDefaultProfiles returns default test profiles
func GetDefaultProfiles() map[string]TestProfile {
	return map[string]TestProfile{
		"quick": {
			Name:        "Quick Test",
			Description: "Fast connectivity and basic quality check",
			Streams:     []StreamEndpoint{},
			Regions:     []string{"local"},
			Duration:    30 * time.Second,
			Metrics:     []string{"latency", "connectivity", "basic_quality"},
			Thresholds:  GetDefaultQualityConfig(),
			Tags: map[string]string{
				"profile": "quick",
				"type":    "development",
			},
		},
		"standard": {
			Name:        "Standard Test",
			Description: "Comprehensive testing with quality analysis",
			Streams:     []StreamEndpoint{},
			Regions:     []string{"local"},
			Duration:    2 * time.Minute,
			Metrics:     []string{"latency", "quality", "reliability"},
			Thresholds:  GetDefaultQualityConfig(),
			Tags: map[string]string{
				"profile": "standard",
				"type":    "testing",
			},
		},
		"comprehensive": {
			Name:        "Comprehensive Test",
			Description: "Full suite with spectral analysis and multi-region testing",
			Streams:     []StreamEndpoint{},
			Regions:     []string{"local"},
			Duration:    10 * time.Minute,
			Metrics:     []string{"latency", "quality", "reliability", "spectral_analysis"},
			Thresholds: QualityConfig{
				MinSimilarity: 0.98, // Higher threshold for comprehensive testing
				MaxLatency:    2 * time.Second,
				MinBitrate:    192,
				MaxDropouts:   1,
				BufferHealth:  0.9,
			},
			Tags: map[string]string{
				"profile": "comprehensive",
				"type":    "production",
			},
		},
	}
}

// GetDefaultStreamEndpoint returns a default stream endpoint configuration
func GetDefaultStreamEndpoint() StreamEndpoint {
	return StreamEndpoint{
		Name:      "Default Stream",
		URL:       "",
		Type:      "auto", // Auto-detect stream type
		Reference: "",
		Headers:   make(map[string]string),
		Expected:  GetDefaultStreamMetadata(),
	}
}

// GetDefaultStreamMetadata returns default expected stream metadata
func GetDefaultStreamMetadata() StreamMetadata {
	return StreamMetadata{
		Bitrate:    128,
		SampleRate: 44100,
		Channels:   2,
		Codec:      "aac",
		Format:     "auto",
	}
}

// DefaultStreamHeaders returns common default headers for stream requests
func DefaultStreamHeaders() map[string]string {
	return map[string]string{
		"User-Agent":    "TuneIn-CDN-Benchmark/1.0",
		"Accept":        "audio/*, application/vnd.apple.mpegurl, application/x-mpegurl",
		"Cache-Control": "no-cache",
		"Connection":    "keep-alive",
	}
}

// ProductionQualityConfig returns stricter quality thresholds for production
func ProductionQualityConfig() QualityConfig {
	return QualityConfig{
		MinSimilarity: 0.99,
		MaxLatency:    1 * time.Second,
		MinBitrate:    256,
		MaxDropouts:   0,
		BufferHealth:  0.95,
	}
}

// DevelopmentQualityConfig returns relaxed quality thresholds for development
func DevelopmentQualityConfig() QualityConfig {
	return QualityConfig{
		MinSimilarity: 0.85,
		MaxLatency:    10 * time.Second,
		MinBitrate:    64,
		MaxDropouts:   10,
		BufferHealth:  0.5,
	}
}

// HighQualityAudioConfig returns high-quality audio processing settings
func HighQualityAudioConfig() AudioConfig {
	return AudioConfig{
		SampleRate:     48000,
		Channels:       2,
		BufferDuration: 2 * time.Second,
		WindowSize:     4096,
		Overlap:        0.75,
		WindowFunction: "hamming",
		FFTSize:        4096,
		MelBins:        256,
	}
}

// FastAudioConfig returns optimized audio processing settings for speed
func FastAudioConfig() AudioConfig {
	return AudioConfig{
		SampleRate:     22050,
		Channels:       1,
		BufferDuration: 500 * time.Millisecond,
		WindowSize:     1024,
		Overlap:        0.25,
		WindowFunction: "hann",
		FFTSize:        1024,
		MelBins:        64,
	}
}

// ProductionTestConfig returns production-optimized test settings
func ProductionTestConfig() TestConfig {
	return TestConfig{
		Timeout:        5 * time.Minute,
		RetryAttempts:  5,
		RetryDelay:     10 * time.Second,
		Concurrent:     true,
		MaxConcurrency: 8,
	}
}

// DevelopmentTestConfig returns development-optimized test settings
func DevelopmentTestConfig() TestConfig {
	return TestConfig{
		Timeout:        1 * time.Minute,
		RetryAttempts:  1,
		RetryDelay:     2 * time.Second,
		Concurrent:     false,
		MaxConcurrency: 2,
	}
}

// GetProductionStreamConfig returns production-optimized stream settings
func GetProductionStreamConfig() StreamConfig {
	return StreamConfig{
		ConnectionTimeout: 15 * time.Second,
		ReadTimeout:       60 * time.Second,
		BufferSize:        16384,
		MaxRedirects:      5,
		UserAgent:         "TuneIn-CDN-Benchmark-Prod/1.0",
		Headers: map[string]string{
			"X-Forwarded-For": "10.0.0.1",
			"Cache-Control":   "no-cache",
			"Accept":          "audio/*, application/vnd.apple.mpegurl",
		},
	}
}

// GetDevelopmentStreamConfig returns development-optimized stream settings
func GetDevelopmentStreamConfig() StreamConfig {
	return StreamConfig{
		ConnectionTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		BufferSize:        4096,
		MaxRedirects:      2,
		UserAgent:         "TuneIn-CDN-Benchmark-Dev/1.0",
		Headers:           DefaultStreamHeaders(),
	}
}

// GetDefaultOutputConfigForFormat returns output config optimized for specific format
func GetDefaultOutputConfigForFormat(format string) OutputConfig {
	base := GetDefaultOutputConfig()

	switch format {
	case "json":
		base.Colors = false
		base.Pager = false
		base.Precision = 6
	case "csv":
		base.Colors = false
		base.Pager = false
		base.IncludeMetadata = false
		base.Timestamps = false
	case "table":
		base.Colors = true
		base.Pager = true
		base.Precision = 2
	default:
		// Keep defaults
	}

	return base
}
