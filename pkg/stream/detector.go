package stream

import (
	"context"
	"net/http"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/hls"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/icecast"
)

// Detector implements the StreamDetector interface and provides stream type
// detection and lightweight probing capabilities. It acts as an orchestrator
// that delegates stream-specific operations to the appropriate packages.
//
// The detector maintains an HTTP client with configurable timeout and redirect
// behavior, which is shared across all detection operations for consistency.
type Detector struct {
	client *http.Client
}

// NewDetector creates a new stream detector with default configuration:
// - 10 second timeout
// - Maximum of 3 redirects
//
// This is suitable for most general-purpose stream detection scenarios.
func NewDetector() *Detector {
	return &Detector{
		client: &http.Client{
			Timeout: 10 * time.Second,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				// Follow up to 3 redirects
				if len(via) >= 3 {
					return http.ErrUseLastResponse
				}
				return nil
			},
		},
	}
}

// NewDetectorWithTimeout creates a new stream detector with a custom timeout.
// The redirect limit remains at 3.
//
// Parameters:
//   - timeoutSecs: HTTP request timeout in seconds
//
// Use this when you need a different timeout than the default 10 seconds,
// such as faster detection for CI/CD or longer timeouts for slow networks.
func NewDetectorWithTimeout(timeoutSecs int) *Detector {
	return &Detector{
		client: &http.Client{
			Timeout: time.Duration(timeoutSecs) * time.Second,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				// Follow up to 3 redirects
				if len(via) >= 3 {
					return http.ErrUseLastResponse
				}
				return nil
			},
		},
	}
}

// NewDetectorWithParams creates a new stream detector with custom timeout and redirect limits.
// This provides full control over the HTTP client behavior.
//
// Parameters:
//   - timeoutSecs: HTTP request timeout in seconds
//   - redirects: Maximum number of redirects to follow
//
// Use this for fine-tuned control, such as:
// - Setting higher redirect limits for complex CDN setups
// - Disabling redirects entirely (redirects = 0) for strict URL validation
// - Adjusting timeouts for specific network conditions
func NewDetectorWithParams(timeoutSecs int, redirects int) *Detector {
	return &Detector{
		client: &http.Client{
			Timeout: time.Duration(timeoutSecs) * time.Second,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) >= redirects {
					return http.ErrUseLastResponse
				}
				return nil
			},
		},
	}
}

// DetectType determines the stream type for the given URL by delegating to
// stream-specific detection functions. It implements a fallback strategy:
//
//  1. Try HLS detection first
//  2. Fall back to ICEcast detection
//  3. Return unsupported if no match
//
// This method does not return an error for unsupported stream types; it only
// returns errors for actual failure conditions (network errors, etc.).
//
// Parameters:
//   - ctx: Context for cancellation and timeout control
//   - streamURL: The URL to analyze for stream type detection
//
// Returns:
//   - StreamType: The detected stream type or StreamTypeUnsupported
//   - error: Only for actual failures, not for unsupported types
//
// Example:
//
//	detector := stream.NewDetector()
//	streamType, err := detector.DetectType(ctx, "https://example.com/playlist.m3u8")
//	if err != nil {
//	    // Handle detection failure
//	}
//	if streamType == common.StreamTypeHLS {
//	    // Handle HLS stream
//	}
func (sd *Detector) DetectType(ctx context.Context, streamURL string) (common.StreamType, error) {
	// Try HLS detection
	streamType, err := hls.DetectType(ctx, sd.client, streamURL)
	if err == nil && streamType == common.StreamTypeHLS {
		return common.StreamTypeHLS, nil
	}

	// Try ICEcast detection
	streamType, err = icecast.DetectType(ctx, sd.client, streamURL)
	if err == nil && streamType == common.StreamTypeICEcast {
		return common.StreamTypeICEcast, nil
	}

	return common.StreamTypeUnsupported, nil
}

// ProbeStream performs a lightweight probe to gather basic stream metadata.
// It first detects the stream type, then delegates to the appropriate
// stream-specific probing function to extract detailed metadata.
//
// This method provides a unified interface for metadata extraction across
// different stream types while letting each package handle its own
// stream-specific logic and metadata extraction strategies.
//
// The probe operation is designed to be lightweight and fast, typically
// using HEAD requests or minimal content parsing to gather essential
// metadata without downloading significant amounts of stream data.
//
// Parameters:
//   - ctx: Context for cancellation and timeout control
//   - streamURL: The URL to probe for metadata
//
// Returns:
//   - *StreamMetadata: Extracted metadata including stream type, format,
//     bitrate, codec, station info, and other stream-specific properties
//   - error: Returns error if stream type detection fails, probing fails,
//     or if the stream type is unsupported
//
// Example:
//
//	detector := stream.NewDetector()
//	metadata, err := detector.ProbeStream(ctx, "https://live.example.com/stream")
//	if err != nil {
//	    // Handle probing failure
//	}
//	fmt.Printf("Stream: %s (%s) - %d kbps\n",
//	    metadata.Station, metadata.Type, metadata.Bitrate)
func (d *Detector) ProbeStream(ctx context.Context, streamURL string) (*common.StreamMetadata, error) {
	streamType, err := d.DetectType(ctx, streamURL)
	if err != nil {
		return nil, err
	}

	// Delegate entirely to the appropriate package for stream-specific probing
	switch streamType {
	case common.StreamTypeHLS:
		return hls.ProbeStream(ctx, d.client, streamURL)

	case common.StreamTypeICEcast:
		return icecast.ProbeStream(ctx, d.client, streamURL)

	default:
		return nil, common.NewStreamError(
			streamType, streamURL, common.ErrCodeUnsupported,
			"unsupported stream type for probing", nil,
		)
	}
}

