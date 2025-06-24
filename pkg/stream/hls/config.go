package hls

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// Config holds configuration for HLS processing
type Config struct {
	Parser            *ParserConfig            `json:"parser"`
	MetadataExtractor *MetadataExtractorConfig `json:"metadata_extractor"`
	Detection         *DetectionConfig         `json:"detection"`
}

// ParserConfig holds configuration for M3U8 parsing
type ParserConfig struct {
	StrictMode         bool              `json:"strict_mode"`
	MaxSegmentAnalysis int               `json:"max_segment_analysis"`
	CustomTagHandlers  map[string]string `json:"custom_tag_handlers"`
	IgnoreUnknownTags  bool              `json:"ignore_unknown_tags"`
	ValidateURIs       bool              `json:"validate_uris"`
}

// MetadataExtractorConfig holds configuration for metadata extraction
type MetadataExtractorConfig struct {
	EnableURLPatterns     bool                   `json:"enable_url_patterns"`
	EnableHeaderMappings  bool                   `json:"enable_header_mappings"`
	EnableSegmentAnalysis bool                   `json:"enable_segment_analysis"`
	CustomPatterns        []CustomURLPattern     `json:"custom_patterns"`
	CustomHeaderMappings  []CustomHeaderMapping  `json:"custom_header_mappings"`
	DefaultValues         map[string]interface{} `json:"default_values"`
}

// DetectionConfig holds configuration for stream detection
type DetectionConfig struct {
	URLPatterns     []string `json:"url_patterns"`
	ContentTypes    []string `json:"content_types"`
	RequiredHeaders []string `json:"required_headers"`
	TimeoutSeconds  int      `json:"timeout_seconds"`
}

// CustomURLPattern defines a custom URL pattern for configuration
type CustomURLPattern struct {
	Pattern     string            `json:"pattern"`
	Fields      map[string]string `json:"fields"`
	Priority    int               `json:"priority"`
	Description string            `json:"description"`
}

// CustomHeaderMapping defines a custom header mapping for configuration
type CustomHeaderMapping struct {
	HeaderKey   string `json:"header_key"`
	MetadataKey string `json:"metadata_key"`
	Transform   string `json:"transform"`
	Description string `json:"description"`
}

// DefaultConfig returns the default HLS configuration
func DefaultConfig() *Config {
	return &Config{
		Parser: &ParserConfig{
			StrictMode:         false,
			MaxSegmentAnalysis: 10,
			CustomTagHandlers:  make(map[string]string),
			IgnoreUnknownTags:  false,
			ValidateURIs:       false,
		},
		MetadataExtractor: &MetadataExtractorConfig{
			EnableURLPatterns:     true,
			EnableHeaderMappings:  true,
			EnableSegmentAnalysis: true,
			CustomPatterns:        []CustomURLPattern{},
			CustomHeaderMappings:  []CustomHeaderMapping{},
			DefaultValues: map[string]interface{}{
				"codec":       "aac",
				"channels":    2,
				"sample_rate": 44100,
			},
		},
		Detection: &DetectionConfig{
			URLPatterns: []string{
				`\.m3u8$`,
				`/playlist\.m3u8`,
				`/master\.m3u8`,
				`/index\.m3u8`,
			},
			ContentTypes: []string{
				"application/vnd.apple.mpegurl",
				"application/x-mpegurl",
				"vnd.apple.mpegurl",
			},
			RequiredHeaders: []string{},
			TimeoutSeconds:  5,
		},
	}
}

// ConfigurableMetadataExtractor is a metadata extractor that can be configured
type ConfigurableMetadataExtractor struct {
	*MetadataExtractor
	config *MetadataExtractorConfig
}

// NewConfigurableMetadataExtractor creates a configurable metadata extractor
func NewConfigurableMetadataExtractor(config *MetadataExtractorConfig) *ConfigurableMetadataExtractor {
	if config == nil {
		config = DefaultConfig().MetadataExtractor
	}

	extractor := &ConfigurableMetadataExtractor{
		MetadataExtractor: NewMetadataExtractor(),
		config:            config,
	}

	// Apply custom configurations
	extractor.applyConfig()

	return extractor
}

// applyConfig applies the configuration to the metadata extractor
func (cme *ConfigurableMetadataExtractor) applyConfig() {
	// Add custom URL patterns
	for _, customPattern := range cme.config.CustomPatterns {
		if regex, err := regexp.Compile(customPattern.Pattern); err == nil {
			pattern := URLPattern{
				Pattern:     regex,
				Priority:    customPattern.Priority,
				Description: customPattern.Description,
				Extractor:   cme.createCustomPatternExtractor(customPattern.Fields),
			}
			cme.AddURLPattern(pattern)
		}
	}

	// Add custom header mappings
	for _, customMapping := range cme.config.CustomHeaderMappings {
		mapping := HeaderMapping{
			HeaderKey:   customMapping.HeaderKey,
			MetadataKey: customMapping.MetadataKey,
			Transformer: cme.createCustomTransformer(customMapping.Transform),
		}
		cme.AddHeaderMapping(mapping)
	}
}

// createCustomPatternExtractor creates an extractor function for custom patterns
func (cme *ConfigurableMetadataExtractor) createCustomPatternExtractor(fields map[string]string) func([]string, *common.StreamMetadata) {
	return func(matches []string, metadata *common.StreamMetadata) {
		for field, pattern := range fields {
			if groupIndex, err := strconv.Atoi(pattern); err == nil && groupIndex < len(matches) {
				value := matches[groupIndex]
				cme.setMetadataField(metadata, field, value)
			}
		}
	}
}

// createCustomTransformer creates a transformer function based on configuration
func (cme *ConfigurableMetadataExtractor) createCustomTransformer(transform string) func(string) interface{} {
	return func(value string) interface{} {
		switch transform {
		case "int":
			if i, err := strconv.Atoi(value); err == nil {
				return i
			}
		case "float":
			if f, err := strconv.ParseFloat(value, 64); err == nil {
				return f
			}
		case "bool":
			if b, err := strconv.ParseBool(value); err == nil {
				return b
			}
		case "lower":
			return strings.ToLower(value)
		case "upper":
			return strings.ToUpper(value)
		case "title":
			return titleCaser.String(value)
		default:
			return value
		}
		return nil
	}
}

// setMetadataField sets a field in the metadata based on field name
func (cme *ConfigurableMetadataExtractor) setMetadataField(metadata *common.StreamMetadata, field, value string) {
	switch field {
	case "bitrate":
		if i, err := strconv.Atoi(value); err == nil {
			metadata.Bitrate = i
		}
	case "sample_rate":
		if i, err := strconv.Atoi(value); err == nil {
			metadata.SampleRate = i
		}
	case "channels":
		if i, err := strconv.Atoi(value); err == nil {
			metadata.Channels = i
		}
	case "codec":
		metadata.Codec = value
	case "format":
		metadata.Format = value
	case "station":
		metadata.Station = value
	case "genre":
		metadata.Genre = value
	case "title":
		metadata.Title = value
	case "artist":
		metadata.Artist = value
	default:
		// Store in headers for unknown fields
		if metadata.Headers == nil {
			metadata.Headers = make(map[string]string)
		}
		metadata.Headers[field] = value
	}
}

// ExtractMetadata extracts metadata with configuration overrides
func (cme *ConfigurableMetadataExtractor) ExtractMetadata(playlist *M3U8Playlist, streamURL string) *common.StreamMetadata {
	metadata := cme.MetadataExtractor.ExtractMetadata(playlist, streamURL)

	// Apply default values for missing fields
	cme.applyDefaults(metadata)

	return metadata
}

// applyDefaults applies default values from configuration
func (cme *ConfigurableMetadataExtractor) applyDefaults(metadata *common.StreamMetadata) {
	for field, defaultValue := range cme.config.DefaultValues {
		switch field {
		case "codec":
			if metadata.Codec == "" {
				if codec, ok := defaultValue.(string); ok {
					metadata.Codec = codec
				}
			}
		case "channels":
			if metadata.Channels == 0 {
				if channels, ok := defaultValue.(int); ok {
					metadata.Channels = channels
				} else if channelsFloat, ok := defaultValue.(float64); ok {
					metadata.Channels = int(channelsFloat)
				}
			}
		case "sample_rate":
			if metadata.SampleRate == 0 {
				if rate, ok := defaultValue.(int); ok {
					metadata.SampleRate = rate
				} else if rateFloat, ok := defaultValue.(float64); ok {
					metadata.SampleRate = int(rateFloat)
				}
			}
		case "bitrate":
			if metadata.Bitrate == 0 {
				if bitrate, ok := defaultValue.(int); ok {
					metadata.Bitrate = bitrate
				} else if bitrateFloat, ok := defaultValue.(float64); ok {
					metadata.Bitrate = int(bitrateFloat)
				}
			}
		}
	}
}

// ConfigurableParser is a parser that can be configured
type ConfigurableParser struct {
	*Parser
	config *ParserConfig
}

// NewConfigurableParser creates a configurable parser
func NewConfigurableParser(config *ParserConfig) *ConfigurableParser {
	if config == nil {
		config = DefaultConfig().Parser
	}

	parser := &ConfigurableParser{
		Parser: NewParser(),
		config: config,
	}

	return parser
}

// ParseM3U8Content parses with configuration options
func (cp *ConfigurableParser) ParseM3U8Content(reader any) (*M3U8Playlist, error) {
	// Convert reader to io.Reader if needed
	// This would be expanded based on actual implementation needs

	// For now, delegate to the base parser
	// In a real implementation, you'd apply config options here

	// TODO: apply config options
	return cp.Parser.ParseM3U8Content(reader.(interface {
		Read([]byte) (int, error)
	}))
}
