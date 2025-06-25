package configs

import (
	"os"
	"path/filepath"
	"time"
)

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
		AcceptHeader:      "application/vnd.apple.mpegurl,application/x-mpegurl,text/plain",
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
