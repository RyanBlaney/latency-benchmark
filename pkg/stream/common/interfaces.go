package common

// StreamType represents the type of an audio stream
type StreamType string

const (
	StreamTypeHLS         StreamType = "hls"
	StreamTypeICEcast     StreamType = "icecast"
	StreamTypeUnsupported StreamType = "unsupported"
)

// StreamMetadata contains metadata and info about the stream
type StreamMetadata struct {
	URL     string     `json:"url"`
	Type    StreamType `json:"type"`
	Bitrate int        `json:"bitrate,omitempty"`
}
