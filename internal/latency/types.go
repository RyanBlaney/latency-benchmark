package latency

import (
	"fmt"
	"time"

	"github.com/tunein/cdn-benchmark-cli/configs"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint"
	"github.com/tunein/cdn-benchmark-cli/pkg/audio/fingerprint/extractors"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// BenchmarkConfig extends the base config with benchmark-specific settings
type BenchmarkConfig struct {
	// Base configuration from existing configs package
	*configs.Config

	// Benchmark-specific settings
	Benchmark BenchmarkSettings `json:"benchmark" yaml:"benchmark"`
}

// BroadcastConfig contains the broadcast groups configuration (separate file)
type BroadcastConfig struct {
	// Metadata about the broadcast configuration
	Version     string    `json:"version" yaml:"version"`
	Environment string    `json:"environment" yaml:"environment"`
	UpdatedAt   time.Time `json:"updated_at" yaml:"updated_at"`
	Description string    `json:"description" yaml:"description"`

	// The actual broadcast groups
	BroadcastGroups map[string]*BroadcastGroup `json:"broadcast_groups" yaml:"broadcast_groups"`
}

// BenchmarkSettings contains benchmark execution settings
type BenchmarkSettings struct {
	// Global timeouts
	BenchmarkTimeout time.Duration `json:"benchmark_timeout" yaml:"benchmark_timeout"`
	OperationTimeout time.Duration `json:"operation_timeout" yaml:"operation_timeout"`

	// Concurrency settings
	MaxConcurrentBroadcasts int `json:"max_concurrent_broadcasts" yaml:"max_concurrent_broadcasts"`
	MaxConcurrentStreams    int `json:"max_concurrent_streams" yaml:"max_concurrent_streams"`

	// Audio analysis settings
	AudioSegmentDuration     time.Duration `json:"audio_segment_duration" yaml:"audio_segment_duration"`
	MinAlignmentConfidence   float64       `json:"min_alignment_confidence" yaml:"min_alignment_confidence"`
	MaxAlignmentOffset       float64       `json:"max_alignment_offset" yaml:"max_alignment_offset"`
	MinFingerprintSimilarity float64       `json:"min_fingerprint_similarity" yaml:"min_fingerprint_similarity"`

	// Analysis flags
	EnableDetailedAnalysis    bool `json:"enable_detailed_analysis" yaml:"enable_detailed_analysis"`
	SkipFingerprintComparison bool `json:"skip_fingerprint_comparison" yaml:"skip_fingerprint_comparison"`

	// Retry settings
	MaxRetries int           `json:"max_retries" yaml:"max_retries"`
	RetryDelay time.Duration `json:"retry_delay" yaml:"retry_delay"`

	// Validation settings
	ValidateHLSStructure       bool    `json:"validate_hls_structure" yaml:"validate_hls_structure"`
	ValidateICEcastMetadata    bool    `json:"validate_icecast_metadata" yaml:"validate_icecast_metadata"`
	ValidateAudioFormat        bool    `json:"validate_audio_format" yaml:"validate_audio_format"`
	ValidateBitrateConsistency bool    `json:"validate_bitrate_consistency" yaml:"validate_bitrate_consistency"`
	MinHealthScore             float64 `json:"min_health_score" yaml:"min_health_score"`
	FailFast                   bool    `json:"fail_fast" yaml:"fail_fast"`

	// Output settings
	OutputFormat            string `json:"output_format" yaml:"output_format"`
	IncludeRawData          bool   `json:"include_raw_data" yaml:"include_raw_data"`
	IncludeFeatureBreakdown bool   `json:"include_feature_breakdown" yaml:"include_feature_breakdown"`
	PrettyPrint             bool   `json:"pretty_print" yaml:"pretty_print"`
	GenerateSummary         bool   `json:"generate_summary" yaml:"generate_summary"`
}

// Validate validates the benchmark configuration
func (c *BenchmarkConfig) Validate() error {
	// Validate base configuration first
	if err := configs.ValidateConfig(c.Config); err != nil {
		return fmt.Errorf("base configuration invalid: %w", err)
	}

	// Validate benchmark settings
	if c.Benchmark.BenchmarkTimeout <= 0 {
		return fmt.Errorf("benchmark timeout must be positive")
	}

	if c.Benchmark.OperationTimeout <= 0 {
		return fmt.Errorf("operation timeout must be positive")
	}

	if c.Benchmark.MaxConcurrentBroadcasts <= 0 {
		return fmt.Errorf("max concurrent broadcasts must be positive")
	}

	if c.Benchmark.MaxConcurrentStreams <= 0 {
		return fmt.Errorf("max concurrent streams must be positive")
	}

	if c.Benchmark.AudioSegmentDuration <= 0 {
		return fmt.Errorf("audio segment duration must be positive")
	}

	if c.Benchmark.MinAlignmentConfidence < 0 || c.Benchmark.MinAlignmentConfidence > 1 {
		return fmt.Errorf("min alignment confidence must be between 0 and 1")
	}

	if c.Benchmark.MaxAlignmentOffset <= 0 {
		return fmt.Errorf("max alignment offset must be positive")
	}

	if c.Benchmark.MinFingerprintSimilarity < 0 || c.Benchmark.MinFingerprintSimilarity > 1 {
		return fmt.Errorf("min fingerprint similarity must be between 0 and 1")
	}

	if c.Benchmark.MinHealthScore < 0 || c.Benchmark.MinHealthScore > 1 {
		return fmt.Errorf("min health score must be between 0 and 1")
	}

	return nil
}

// Validate validates the broadcast configuration
func (c *BroadcastConfig) Validate() error {
	if len(c.BroadcastGroups) == 0 {
		return fmt.Errorf("at least one broadcast group is required")
	}

	for name, group := range c.BroadcastGroups {
		if err := c.validateBroadcastGroup(name, group); err != nil {
			return fmt.Errorf("invalid broadcast group %s: %w", name, err)
		}
	}

	return nil
}

// validateBroadcastGroup validates a single broadcast group
func (c *BroadcastConfig) validateBroadcastGroup(name string, group *BroadcastGroup) error {
	if group.Name == "" {
		return fmt.Errorf("broadcast group name is required")
	}

	if len(group.Streams) == 0 {
		return fmt.Errorf("at least one stream is required")
	}

	// Check for required stream combinations
	hlsSource := group.GetStreamByTypeAndRole(StreamTypeHLS, StreamRoleSource)
	hlsCDN := group.GetStreamByTypeAndRole(StreamTypeHLS, StreamRoleCDN)
	icecastSource := group.GetStreamByTypeAndRole(StreamTypeICEcast, StreamRoleSource)
	icecastCDN := group.GetStreamByTypeAndRole(StreamTypeICEcast, StreamRoleCDN)

	// Validate that we have at least some meaningful comparisons
	hasHLSPair := hlsSource != nil && hlsCDN != nil
	hasICEcastPair := icecastSource != nil && icecastCDN != nil
	hasCrossProtocol := hlsSource != nil && icecastSource != nil

	if !hasHLSPair && !hasICEcastPair && !hasCrossProtocol {
		return fmt.Errorf("broadcast group must have at least one of: HLS source+CDN pair, ICEcast source+CDN pair, or cross-protocol sources")
	}

	// Validate individual streams
	for streamName, stream := range group.Streams {
		if err := c.validateStreamEndpoint(streamName, stream); err != nil {
			return fmt.Errorf("invalid stream %s: %w", streamName, err)
		}
	}

	return nil
}

// validateStreamEndpoint validates a single stream endpoint
func (c *BroadcastConfig) validateStreamEndpoint(name string, stream *StreamEndpoint) error {
	if stream.URL == "" {
		return fmt.Errorf("stream URL is required")
	}

	if stream.Type != StreamTypeHLS && stream.Type != StreamTypeICEcast {
		return fmt.Errorf("invalid stream type: %s (must be hls or icecast)", stream.Type)
	}

	if stream.Role != StreamRoleSource && stream.Role != StreamRoleCDN {
		return fmt.Errorf("invalid stream role: %s (must be source or cdn)", stream.Role)
	}

	if stream.ContentType == "" {
		return fmt.Errorf("content type is required")
	}

	validContentTypes := map[string]bool{
		"music":  true,
		"news":   true,
		"talk":   true,
		"sports": true,
		"mixed":  true,
	}

	if !validContentTypes[stream.ContentType] {
		return fmt.Errorf("invalid content type: %s (must be music, news, talk, sports, or mixed)", stream.ContentType)
	}

	return nil
}

func (c *BroadcastConfig) GetEnabledBroadcastGroups() map[string]*BroadcastGroup {
	enabled := make(map[string]*BroadcastGroup)
	for name, group := range c.BroadcastGroups {
		if group.Enabled {
			enabled[name] = group
		}
	}
	return enabled
}

// StreamType represents the type of stream being measured
type StreamType string

const (
	StreamTypeHLS     StreamType = "hls"
	StreamTypeICEcast StreamType = "icecast"
)

// StreamRole represents the role of the stream in the CDN architecture
type StreamRole string

const (
	StreamRoleSource StreamRole = "source"
	StreamRoleCDN    StreamRole = "cdn"
)

// StreamEndpoint represents a single stream endpoint
type StreamEndpoint struct {
	URL         string     `json:"url" yaml:"url"`
	Type        StreamType `json:"type" yaml:"type"`
	Role        StreamRole `json:"role" yaml:"role"`
	ContentType string     `json:"content_type" yaml:"content_type"`
	Enabled     bool       `json:"enabled" yaml:"enabled"`
}

// BroadcastGroup represents a collection of related streams for a single broadcast
type BroadcastGroup struct {
	Name        string                     `json:"name" yaml:"name"`
	Description string                     `json:"description,omitempty" yaml:"description,omitempty"`
	ContentType string                     `json:"content_type" yaml:"content_type"`
	Streams     map[string]*StreamEndpoint `json:"streams" yaml:"streams"`
	Enabled     bool                       `json:"enabled" yaml:"enabled"`
}

// GetStreamByTypeAndRole returns a stream endpoint by type and role
func (bg *BroadcastGroup) GetStreamByTypeAndRole(streamType StreamType, role StreamRole) *StreamEndpoint {
	for _, stream := range bg.Streams {
		if stream.Type == streamType && stream.Role == role && stream.Enabled {
			return stream
		}
	}
	return nil
}

// StreamMeasurement represents the measurement results for a single stream
type StreamMeasurement struct {
	Endpoint            *StreamEndpoint               `json:"endpoint"`
	AudioData           *common.AudioData             `json:"-"` // Don't serialize raw audio
	Fingerprint         *fingerprint.AudioFingerprint `json:"-"` // Don't serialize fingerprint
	TimeToFirstByte     time.Duration                 `json:"time_to_first_byte"`
	AudioExtractionTime time.Duration                 `json:"audio_extraction_time"`
	FingerprintTime     time.Duration                 `json:"fingerprint_time"`
	TotalProcessingTime time.Duration                 `json:"total_processing_time"`
	StreamValidation    *StreamValidation             `json:"stream_validation"`
	Error               error                         `json:"error,omitempty"`
	Timestamp           time.Time                     `json:"timestamp"`
}

// StreamValidation represents the validation results for a stream
type StreamValidation struct {
	IsValid           bool     `json:"is_valid"`
	ValidationErrors  []string `json:"validation_errors,omitempty"`
	PlaylistStructure bool     `json:"playlist_structure,omitempty"` // For HLS
	HTTPHeaders       bool     `json:"http_headers,omitempty"`       // For ICEcast
	AudioFormat       bool     `json:"audio_format"`
	BitrateConsistent bool     `json:"bitrate_consistent"`
}

// AlignmentMeasurement represents alignment comparison between two streams
type AlignmentMeasurement struct {
	Stream1          *StreamMeasurement            `json:"stream1"`
	Stream2          *StreamMeasurement            `json:"stream2"`
	AlignmentResult  *extractors.AlignmentFeatures `json:"alignment_result"`
	LatencySeconds   float64                       `json:"latency_seconds"`
	IsValidAlignment bool                          `json:"is_valid_alignment"`
	ComparisonTime   time.Duration                 `json:"comparison_time"`
	Error            error                         `json:"error,omitempty"`
	Timestamp        time.Time                     `json:"timestamp"`
}

// FingerprintComparison represents fingerprint similarity comparison
type FingerprintComparison struct {
	Stream1           *StreamMeasurement            `json:"stream1"`
	Stream2           *StreamMeasurement            `json:"stream2"`
	SimilarityResult  *fingerprint.SimilarityResult `json:"similarity_result"`
	AlignmentFeatures *extractors.AlignmentFeatures `json:"alignment_features,omitempty"`
	IsValidMatch      bool                          `json:"is_valid_match"`
	ComparisonTime    time.Duration                 `json:"comparison_time"`
	Error             error                         `json:"error,omitempty"`
	Timestamp         time.Time                     `json:"timestamp"`
}

// BroadcastMeasurement represents all measurements for a broadcast group
type BroadcastMeasurement struct {
	Group                  *BroadcastGroup                   `json:"group"`
	StreamMeasurements     map[string]*StreamMeasurement     `json:"stream_measurements"`
	AlignmentMeasurements  map[string]*AlignmentMeasurement  `json:"alignment_measurements"`
	FingerprintComparisons map[string]*FingerprintComparison `json:"fingerprint_comparisons"`
	LivenessMetrics        *LivenessMetrics                  `json:"liveness_metrics"`
	OverallValidation      *OverallValidation                `json:"overall_validation"`
	TotalBenchmarkTime     time.Duration                     `json:"total_benchmark_time"`
	Error                  error                             `json:"error,omitempty"`
	Timestamp              time.Time                         `json:"timestamp"`
}

// LivenessMetrics represents how far behind live each stream is
type LivenessMetrics struct {
	HLSSourceLag      float64 `json:"hls_source_lag_seconds"`
	HLSCDNLag         float64 `json:"hls_cdn_lag_seconds"`
	ICEcastSourceLag  float64 `json:"icecast_source_lag_seconds"`
	ICEcastCDNLag     float64 `json:"icecast_cdn_lag_seconds"`
	CDNLatencyHLS     float64 `json:"cdn_latency_hls_seconds"`     // HLS CDN - HLS Source
	CDNLatencyICEcast float64 `json:"cdn_latency_icecast_seconds"` // ICEcast CDN - ICEcast Source
	CrossProtocolLag  float64 `json:"cross_protocol_lag_seconds"`  // HLS Source - ICEcast Source
}

// OverallValidation represents the overall health of all streams
type OverallValidation struct {
	AllStreamsValid         bool     `json:"all_streams_valid"`
	ValidStreamCount        int      `json:"valid_stream_count"`
	InvalidStreamCount      int      `json:"invalid_stream_count"`
	StreamValidityIssues    []string `json:"stream_validity_issues,omitempty"`
	FingerprintMatchesValid bool     `json:"fingerprint_matches_valid"`
	AlignmentQualityGood    bool     `json:"alignment_quality_good"`
	OverallHealthScore      float64  `json:"overall_health_score"` // 0.0 to 1.0
}

// BenchmarkSummary represents a summary of multiple broadcast measurements
type BenchmarkSummary struct {
	BroadcastMeasurements map[string]*BroadcastMeasurement `json:"broadcast_measurements"`
	StartTime             time.Time                        `json:"start_time"`
	EndTime               time.Time                        `json:"end_time"`
	TotalDuration         time.Duration                    `json:"total_duration"`
	SuccessfulBroadcasts  int                              `json:"successful_broadcasts"`
	FailedBroadcasts      int                              `json:"failed_broadcasts"`
	AverageLatencyMetrics *AverageLatencyMetrics           `json:"average_latency_metrics"`
	OverallHealthScore    float64                          `json:"overall_health_score"`
}

// AverageLatencyMetrics represents average latency across all broadcasts
type AverageLatencyMetrics struct {
	AvgHLSSourceLag        float64 `json:"avg_hls_source_lag_seconds"`
	AvgHLSCDNLag           float64 `json:"avg_hls_cdn_lag_seconds"`
	AvgICEcastSourceLag    float64 `json:"avg_icecast_source_lag_seconds"`
	AvgICEcastCDNLag       float64 `json:"avg_icecast_cdn_lag_seconds"`
	AvgCDNLatencyHLS       float64 `json:"avg_cdn_latency_hls_seconds"`
	AvgCDNLatencyICEcast   float64 `json:"avg_cdn_latency_icecast_seconds"`
	AvgCrossProtocolLag    float64 `json:"avg_cross_protocol_lag_seconds"`
	AvgTimeToFirstByte     float64 `json:"avg_time_to_first_byte_ms"`
	AvgAudioExtractionTime float64 `json:"avg_audio_extraction_time_ms"`
}
