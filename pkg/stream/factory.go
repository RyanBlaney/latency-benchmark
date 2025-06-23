package stream

import (
	"context"
	"fmt"
	"sync"

	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	//"github.com/tunein/cdn-benchmark-cli/pkg/stream/hls"
	//"github.com/tunein/cdn-benchmark-cli/pkg/stream/icecast"
)

// Factory implements StreamManager interface
type Factory struct {
	handlers map[common.StreamType]func() common.StreamHandler
	detector common.StreamDetector
	mu       sync.RWMutex
}

// NewFactory creates a new stream factory with default handlers
func NewFactory() *Factory {
	f := &Factory{
		handlers: make(map[common.StreamType]func() common.StreamHandler),
		detector: NewDetector(),
	}

	// Register default handlers
	f.RegisterHandlerFactory(common.StreamTypeHLS, func() common.StreamHandler {
		return hls.NewHandler()
	})
	f.RegisterHandlerFactory(common.StreamTypeICEcast, func() common.StreamHandler {
		return icecast.NewHandler()
	})

	return f
}

// CreateHandler creates a handler for the given stream type
func (f *Factory) CreateHandler(streamType common.StreamType) (common.StreamHandler, error) {
	f.mu.RLock()
	handlerFactory, exists := f.handlers[streamType]
	f.mu.RUnlock()

	if !exists {
		return nil, common.NewStreamError(
			streamType, "", common.ErrCodeUnsupported,
			fmt.Sprintf("unsupported stream type: %s", streamType),
			nil,
		)
	}

	return handlerFactory(), nil
}

// DetectAndCreate detects stream type and creates appropriate handler
func (f *Factory) DetectAndCreate(ctx context.Context, url string) (common.StreamHandler, error) {
	streamType, err := f.detector.DetectType(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("failed to detect stream type: %w", err)
	}

	handler, err := f.CreateHandler(streamType)
	if err != nil {
		return nil, err
	}

	return handler, nil
}

// RegisterHandler registers a custom stream handler instance (deprecated)
func (f *Factory) RegisterHandler(streamType common.StreamType, handler common.StreamHandler) error {
	return f.RegisterHandlerFactory(streamType, func() common.StreamHandler {
		return handler
	})
}

// RegisterHandlerFactory registers a handler factory function
func (f *Factory) RegisterHandlerFactory(streamType common.StreamType, factory func() common.StreamHandler) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.handlers[streamType] = factory
	return nil
}

// SupportedTypes returns list of supported stream types
func (f *Factory) SupportedTypes() []common.StreamType {
	f.mu.RLock()
	defer f.mu.RUnlock()

	types := make([]common.StreamType, 0, len(f.handlers))
	for streamType := range f.handlers {
		types = append(types, streamType)
	}
	return types
}
