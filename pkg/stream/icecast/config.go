package icecast

import (
	"maps"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/logging"
)

// Config holds configuration for ICEcast processing
type Config struct {
	MetadataExtractor *MetadataExtractorConfig `json:"metadata_extractor"`
	Detection         *DetectionConfig         `json:"detection"`
	HTTP              *HTTPConfig              `json:"http"`
	Audio             *AudioConfig             `json:"audio"`
}

// MetadataExtractorConfig holds configuration for metadata extraction
type MetadataExtractorConfig struct {
	EnableHeaderMappings bool                  `json:"enable_header_mappings"`
	EnableICYMetadata    bool                  `json:"enable_icy_metadata"`
	CustomHeaderMappings []CustomHeaderMapping `json:"custom_header_mappings"`
	DefaultValues        map[string]any        `json:"default_values"`
	ICYMetadataTimeout   time.Duration         `json:"icy_metadata_timeout"`
}

// DetectionConfig holds configuration for stream detection
type DetectionConfig struct {
	URLPatterns     []string `json:"url_patterns"`
	ContentTypes    []string `json:"content_types"`
	RequiredHeaders []string `json:"required_headers"`
	TimeoutSeconds  int      `json:"timeout_seconds"`
	CommonPorts     []string `json:"common_ports"`
}

// CustomHeaderMapping defines a custom header mapping for configuration
type CustomHeaderMapping struct {
	HeaderKey   string `json:"header_key"`
	MetadataKey string `json:"metadata_key"`
	Transform   string `json:"transform"`
	Description string `json:"description"`
}

// HTTPConfig holds HTTP-related configuration for ICEcast
type HTTPConfig struct {
	UserAgent         string            `json:"user_agent"`
	AcceptHeader      string            `json:"accept_header"`
	ConnectionTimeout time.Duration     `json:"connection_timeout"`
	ReadTimeout       time.Duration     `json:"read_timeout"`
	MaxRedirects      int               `json:"max_redirects"`
	CustomHeaders     map[string]string `json:"custom_headers"`
	RequestICYMeta    bool              `json:"request_icy_meta"`
}

// GetHTTPHeaders returns configured HTTP headers for ICEcast requests
func (httpConfig *HTTPConfig) GetHTTPHeaders() map[string]string {
	headers := make(map[string]string)

	// Set standard headers
	headers["User-Agent"] = httpConfig.UserAgent
	headers["Accept"] = httpConfig.AcceptHeader

	// Add ICEcast-specific header if requested
	if httpConfig.RequestICYMeta {
		headers["Icy-MetaData"] = "1"
	}

	// Add custom headers
	maps.Copy(headers, httpConfig.CustomHeaders)

	return headers
}

// AudioConfig holds audio-specific configuration for ICEcast
type AudioConfig struct {
	BufferSize       int           `json:"buffer_size"`
	SampleDuration   time.Duration `json:"sample_duration"`
	MaxReadAttempts  int           `json:"max_read_attempts"`
	ReadTimeout      time.Duration `json:"read_timeout"`
	HandleICYMeta    bool          `json:"handle_icy_meta"`
	MetadataInterval int           `json:"metadata_interval"`
}

// DefaultConfig returns the default ICEcast configuration
func DefaultConfig() *Config {
	return &Config{
		MetadataExtractor: &MetadataExtractorConfig{
			EnableHeaderMappings: true,
			EnableICYMetadata:    true,
			CustomHeaderMappings: []CustomHeaderMapping{},
			DefaultValues: map[string]any{
				"codec":       "mp3",
				"channels":    2,
				"sample_rate": 44100,
				"format":      "mp3",
			},
			ICYMetadataTimeout: 5 * time.Second,
		},
		Detection: &DetectionConfig{
			URLPatterns: []string{
				`\.mp3$`,
				`\.aac$`,
				`\.ogg$`,
				`/stream$`,
				`/listen$`,
				`/audio$`,
				`/radio$`,
			},
			ContentTypes: []string{
				"audio/mpeg",
				"audio/mp3",
				"audio/aac",
				"audio/ogg",
				"application/ogg",
			},
			RequiredHeaders: []string{},
			TimeoutSeconds:  5,
			CommonPorts:     []string{"8000", "8080", "8443", "9000"},
		},
		HTTP: &HTTPConfig{
			UserAgent:         "TuneIn-CDN-Benchmark/1.0",
			AcceptHeader:      "audio/*,*/*",
			ConnectionTimeout: 5 * time.Second,
			ReadTimeout:       15 * time.Second,
			MaxRedirects:      5,
			CustomHeaders:     make(map[string]string),
			RequestICYMeta:    true,
		},
		Audio: &AudioConfig{
			BufferSize:       4096,
			SampleDuration:   30 * time.Second,
			MaxReadAttempts:  3,
			ReadTimeout:      10 * time.Second,
			HandleICYMeta:    true,
			MetadataInterval: 0, // Will be determined from stream
		},
	}
}

// ConfigFromAppConfig creates an ICEcast config from application config
// This allows ICEcast library to remain a standalone library while integrating with the main app
func ConfigFromAppConfig(appConfig any) *Config {
	config := DefaultConfig()

	if appCfg, ok := appConfig.(map[string]any); ok {
		if streamCfg, exists := appCfg["stream"].(map[string]any); exists {
			if userAgent, ok := streamCfg["user_agent"].(string); ok && userAgent != "" {
				config.HTTP.UserAgent = userAgent
			}
			if headers, ok := streamCfg["headers"].(map[string]string); ok {
				config.HTTP.CustomHeaders = headers
			}
			if connTimeout, ok := streamCfg["connection_timeout"].(time.Duration); ok {
				config.HTTP.ConnectionTimeout = connTimeout
			}
			if readTimeout, ok := streamCfg["read_timeout"].(time.Duration); ok {
				config.HTTP.ReadTimeout = readTimeout
			}
			if maxRedirects, ok := streamCfg["max_redirects"].(int); ok {
				config.HTTP.MaxRedirects = maxRedirects
			}
		}

		// Apply audio config if available
		if audioCfg, exists := appCfg["audio"].(map[string]interface{}); exists {
			if bufferSize, ok := audioCfg["buffer_size"].(int); ok {
				config.Audio.BufferSize = bufferSize
			}
			if sampleRate, ok := audioCfg["sample_rate"].(int); ok {
				config.MetadataExtractor.DefaultValues["sample_rate"] = sampleRate
			}
			if channels, ok := audioCfg["channels"].(int); ok {
				config.MetadataExtractor.DefaultValues["channels"] = channels
			}
		}

		// Apply ICEcast-specific config if available
		if icecastCfg, exists := appCfg["icecast"].(map[string]interface{}); exists {
			applyICEcastSpecificConfig(config, icecastCfg)
		}
	}

	return config
}

// ConfigFromMap creates an ICEcast config from a map (useful for testing and flexibility)
func ConfigFromMap(configMap map[string]interface{}) *Config {
	return ConfigFromAppConfig(configMap)
}

// applyICEcastSpecificConfig applies ICEcast-specific configuration overrides
func applyICEcastSpecificConfig(config *Config, icecastCfg map[string]interface{}) {
	// Apply metadata extractor config
	if metaCfg, exists := icecastCfg["metadata_extractor"].(map[string]interface{}); exists {
		if enableHeaders, ok := metaCfg["enable_header_mappings"].(bool); ok {
			config.MetadataExtractor.EnableHeaderMappings = enableHeaders
		}
		if enableICY, ok := metaCfg["enable_icy_metadata"].(bool); ok {
			config.MetadataExtractor.EnableICYMetadata = enableICY
		}
		if timeout, ok := metaCfg["icy_metadata_timeout"].(time.Duration); ok {
			config.MetadataExtractor.ICYMetadataTimeout = timeout
		}
	}

	// Apply detection config
	if detectionCfg, exists := icecastCfg["detection"].(map[string]interface{}); exists {
		if patterns, ok := detectionCfg["url_patterns"].([]string); ok {
			config.Detection.URLPatterns = patterns
		}
		if contentTypes, ok := detectionCfg["content_types"].([]string); ok {
			config.Detection.ContentTypes = contentTypes
		}
		if timeout, ok := detectionCfg["timeout_seconds"].(int); ok {
			config.Detection.TimeoutSeconds = timeout
		}
		if ports, ok := detectionCfg["common_ports"].([]string); ok {
			config.Detection.CommonPorts = ports
		}
	}

	// Apply HTTP config
	if httpCfg, exists := icecastCfg["http"].(map[string]interface{}); exists {
		if userAgent, ok := httpCfg["user_agent"].(string); ok {
			config.HTTP.UserAgent = userAgent
		}
		if acceptHeader, ok := httpCfg["accept_header"].(string); ok {
			config.HTTP.AcceptHeader = acceptHeader
		}
		if headers, ok := httpCfg["custom_headers"].(map[string]string); ok {
			config.HTTP.CustomHeaders = headers
		}
		if requestICY, ok := httpCfg["request_icy_meta"].(bool); ok {
			config.HTTP.RequestICYMeta = requestICY
		}
	}

	// Apply audio config
	if audioCfg, exists := icecastCfg["audio"].(map[string]interface{}); exists {
		if bufferSize, ok := audioCfg["buffer_size"].(int); ok {
			config.Audio.BufferSize = bufferSize
		}
		if duration, ok := audioCfg["sample_duration"].(time.Duration); ok {
			config.Audio.SampleDuration = duration
		}
		if maxAttempts, ok := audioCfg["max_read_attempts"].(int); ok {
			config.Audio.MaxReadAttempts = maxAttempts
		}
		if handleICY, ok := audioCfg["handle_icy_meta"].(bool); ok {
			config.Audio.HandleICYMeta = handleICY
		}
		if interval, ok := audioCfg["metadata_interval"].(int); ok {
			config.Audio.MetadataInterval = interval
		}
	}
}

// GetHTTPHeaders returns all HTTP headers that should be set for requests
func (c *Config) GetHTTPHeaders() map[string]string {
	headers := make(map[string]string)

	// Set standard headers
	headers["User-Agent"] = c.HTTP.UserAgent
	headers["Accept"] = c.HTTP.AcceptHeader

	// Request ICY metadata if enabled
	if c.HTTP.RequestICYMeta {
		headers["Icy-MetaData"] = "1"
	}

	// Add custom headers
	for k, v := range c.HTTP.CustomHeaders {
		headers[k] = v
	}

	return headers
}

// Validate validates the configuration
func (c *Config) Validate() error {
	if c.HTTP.ConnectionTimeout <= 0 {
		return common.NewStreamErrorWithFields(common.StreamTypeICEcast, "",
			common.ErrCodeInvalidFormat, "HTTP connection timeout must be positive", nil,
			logging.Fields{"timeout": c.HTTP.ConnectionTimeout})
	}

	if c.HTTP.ReadTimeout <= 0 {
		return common.NewStreamErrorWithFields(common.StreamTypeICEcast, "",
			common.ErrCodeInvalidFormat, "HTTP read timeout must be positive", nil,
			logging.Fields{"timeout": c.HTTP.ReadTimeout})
	}

	if c.HTTP.MaxRedirects < 0 {
		return common.NewStreamErrorWithFields(common.StreamTypeICEcast, "",
			common.ErrCodeInvalidFormat, "max redirects cannot be negative", nil,
			logging.Fields{"max_redirects": c.HTTP.MaxRedirects})
	}

	if c.Audio.BufferSize <= 0 {
		return common.NewStreamErrorWithFields(common.StreamTypeICEcast, "",
			common.ErrCodeInvalidFormat, "buffer size must be positive", nil,
			logging.Fields{"buffer_size": c.Audio.BufferSize})
	}

	if c.Audio.SampleDuration <= 0 {
		return common.NewStreamErrorWithFields(common.StreamTypeICEcast, "",
			common.ErrCodeInvalidFormat, "sample duration must be positive", nil,
			logging.Fields{"sample_duration": c.Audio.SampleDuration})
	}

	if c.Audio.MaxReadAttempts <= 0 {
		return common.NewStreamErrorWithFields(common.StreamTypeICEcast, "",
			common.ErrCodeInvalidFormat, "max read attempts must be positive", nil,
			logging.Fields{"max_read_attempts": c.Audio.MaxReadAttempts})
	}

	if c.MetadataExtractor.ICYMetadataTimeout <= 0 {
		return common.NewStreamErrorWithFields(common.StreamTypeICEcast, "",
			common.ErrCodeInvalidFormat, "ICY metadata timeout must be positive", nil,
			logging.Fields{"icy_timeout": c.MetadataExtractor.ICYMetadataTimeout})
	}

	return nil
}
