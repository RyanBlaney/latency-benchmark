package stream

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"testing"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

var testFactoryHLSStream = "https://tni-drct-msnbc-int-jg89w.fast.nbcuni.com/live/master.m3u8"
var testFactoryIcecastStream = "http://stream1.skyviewnetworks.com:8010/MSNBC"

// MockStreamDetector is a simple mock implementation of StreamDetector for testing.
type MockStreamDetector struct {
	mockDetectType func(ctx context.Context, url string) (common.StreamType, error)
}

// DetectType mocks the detection of stream types.
func (m *MockStreamDetector) DetectType(ctx context.Context, url string) (common.StreamType, error) {
	return m.mockDetectType(ctx, url)
}

// TestNewFactory checks if the factory is correctly initialized with default handlers.
func TestNewFactory(t *testing.T) {
	factory := NewFactory()

	expectedHandlers := map[common.StreamType]struct{}{
		common.StreamTypeHLS:     {},
		common.StreamTypeICEcast: {},
	}

	for streamType := range expectedHandlers {
		handler, err := factory.CreateHandler(streamType)
		if err != nil {
			t.Errorf("Error creating handler for %s: %v", streamType, err)
			continue
		}
		delete(expectedHandlers, streamType)
		if handler != nil {
			defer handler.Close()
		}
	}

	if len(expectedHandlers) > 0 {
		for streamType := range expectedHandlers {
			t.Errorf("Handler not created for %s", streamType)
		}
	}
}

// TestCreateHandler checks if the factory returns the correct handler instance.
func TestCreateHandler(t *testing.T) {
	factory := NewFactory()

	type test struct {
		streamType  common.StreamType
		expectedErr error
	}

	tests := []test{
		{common.StreamTypeHLS, nil},
		{common.StreamTypeICEcast, nil},
		{common.StreamTypeUnsupported, errors.New("unsupported stream type: unsupported")},
	}

	for _, tt := range tests {
		handler, err := factory.CreateHandler(tt.streamType)
		if (err != nil && err.Error() != tt.expectedErr.Error()) ||
			(err == nil && tt.expectedErr != nil) {
			t.Errorf("CreateHandler(%s): want %v, got %v", tt.streamType, tt.expectedErr, err)
		}
		if handler != nil {
			defer handler.Close()
		}
	}
}

// TestDetectAndCreate checks if the factory detects the correct stream type and creates a handler.
func TestDetectAndCreate(t *testing.T) {
	factory := NewFactory()

	type test struct {
		url         string
		streamType  common.StreamType
		expectedErr error
	}

	tests := []test{
		{testFactoryHLSStream, common.StreamTypeHLS, nil},
		{testFactoryIcecastStream, common.StreamTypeICEcast, nil},
		{"ftp://example.com/file.txt", common.StreamTypeUnsupported, errors.New("unsupported stream type: unsupported")},
	}

	for _, tt := range tests {
		handler, err := factory.DetectAndCreate(context.Background(), tt.url)
		if (err != nil && err.Error() != tt.expectedErr.Error()) ||
			(err == nil && tt.expectedErr != nil) {
			t.Errorf("DetectAndCreate(%s): want %v, got %v", tt.url, tt.expectedErr, err)
		}
		if handler != nil {
			defer handler.Close()
		}
	}
}

// TestRegisterHandler checks if the factory registers custom handlers correctly.
func TestRegisterHandler(t *testing.T) {
	factory := NewFactory()

	type test struct {
		streamType  common.StreamType
		handler     func() common.StreamHandler
		expectedErr error
	}

	tests := []test{
		{common.StreamTypeHLS, func() common.StreamHandler {
			return &MockStreamHandler{}
		}, errors.New("unsupported stream type: unsupported")},
	}

	for _, tt := range tests {
		regErr := factory.RegisterHandler(tt.streamType, tt.handler())
		if regErr != nil {
			t.Errorf("RegisterHandler(%s): failed to register handler: %v", tt.streamType, regErr)
		}

		handler, err := factory.CreateHandler(tt.streamType)
		if err != nil && err.Error() != tt.expectedErr.Error() {
			t.Errorf("RegisterHandler(%s): want %v, got %v", tt.streamType, tt.expectedErr, err)
		}
		if handler != nil {
			defer handler.Close()
		}
	}
}

// TestSupportedTypes checks if the factory returns all supported stream types.
func TestSupportedTypes(t *testing.T) {
	factory := NewFactory()

	expectedTypes := []common.StreamType{
		common.StreamTypeHLS,
		common.StreamTypeICEcast,
	}

	actualTypes := factory.SupportedTypes()
	if len(actualTypes) != len(expectedTypes) {
		t.Errorf("SupportedTypes(): want %v, got %v", expectedTypes, actualTypes)
	}
	for _, type_ := range expectedTypes {
		if !contains(actualTypes, type_) {
			t.Errorf("SupportedTypes(): missing type %s", type_)
		}
	}
}

// contains checks if a slice contains a specific value.
func contains(slice []common.StreamType, value common.StreamType) bool {
	return slices.Contains(slice, value)
}

// MockStreamHandler is a simple mock implementation of StreamHandler.
type MockStreamHandler struct{}

// Type returns the stream type this handler supports.
func (m *MockStreamHandler) Type() common.StreamType {
	// Example return value
	return common.StreamTypeHLS
}

// CanHandle determines if this handler can process the given URL.
func (m *MockStreamHandler) CanHandle(ctx context.Context, url string) bool {
	// Example return value
	return true
}

// Connect establishes a connection to the stream and prepares it for data reading.
func (m *MockStreamHandler) Connect(ctx context.Context, url string) error {
	// Example return value
	return nil
}

// GetMetadata retrieves metadata about the stream.
func (m *MockStreamHandler) GetMetadata() (*common.StreamMetadata, error) {
	// Example return value
	return &common.StreamMetadata{}, nil
}

// ReadAudio reads audio data from the stream.
func (m *MockStreamHandler) ReadAudio(ctx context.Context) (*common.AudioData, error) {
	// Example return value
	return &common.AudioData{}, nil
}

// GetStats returns current streaming statistics.
func (m *MockStreamHandler) GetStats() *common.StreamStats {
	// Example return value
	return &common.StreamStats{}
}

// GetClient returns the HTTP Client associated with this handler.
func (m *MockStreamHandler) GetClient() *http.Client {
	// Example return value
	return &http.Client{}
}

// Close closes the connection to the stream, releasing any resources held by the handler.
func (m *MockStreamHandler) Close() error {
	// Example return value
	return nil
}
