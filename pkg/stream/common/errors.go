package common

func (e *StreamError) Error() string {
	if e.Cause != nil {
		return e.Message + ": " + e.Cause.Error()
	}
	return e.Message
}

// StreamError represents stream-related errors
type StreamError struct {
	Type    StreamType `json:"type"`
	URL     string     `json:"url"`
	Code    string     `json:"code"`
	Message string     `json:"message"`
	Cause   error      `json:"-"`
}

func (e *StreamError) Unwrap() error {
	return e.Cause
}

// Common error codes
const (
	ErrCodeConnection    = "CONNECTION_FAILEeD"
	ErrCodeTimeout       = "TIMEOUT"
	ErrCodeInvalidFormat = "INVALID_FORMAT"
	ErrCodeDecoding      = "DECODING_FAILED"
	ErrCodeMetadata      = "METADATA_ERROR"
	ErrCodeUnsupported   = "UNSUPPORTED_STREAM"
)

// NewStreamError creates a new stream error
func NewStreamError(streamType StreamType, url, code, message string, cause error) *StreamError {
	return &StreamError{
		Type:    streamType,
		URL:     url,
		Code:    code,
		Message: message,
		Cause:   cause,
	}
}
