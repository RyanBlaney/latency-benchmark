package icecast

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"maps"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// Handler implements StreamHandler for ICEcast streams with full configuration support
type Handler struct {
	client            *http.Client
	url               string
	metadata          *common.StreamMetadata
	stats             *common.StreamStats
	connected         bool
	response          *http.Response
	metadataExtractor *MetadataExtractor
	config            *Config
	icyMetaInt        int    // ICY metadata interval
	icyTitle          string // Current ICY title
	bytesRead         int64  // Bytes read since last metadata
	startTime         time.Time
}

// NewHandler creates a new ICEcast stream handler with default configuration
func NewHandler() *Handler {
	return NewHandlerWithConfig(nil)
}

// NewHandlerWithConfig creates a new ICEcast stream handler with custom configuration
func NewHandlerWithConfig(config *Config) *Handler {
	if config == nil {
		config = DefaultConfig()
	}

	// Create HTTP client with configured timeouts
	client := &http.Client{
		Timeout: config.HTTP.ConnectionTimeout + config.HTTP.ReadTimeout,
		Transport: &http.Transport{
			MaxIdleConns:       10,
			IdleConnTimeout:    30 * time.Second,
			DisableCompression: false,
		},
	}

	return &Handler{
		client:            client,
		stats:             &common.StreamStats{},
		metadataExtractor: NewConfigurableMetadataExtractor(config.MetadataExtractor).MetadataExtractor,
		config:            config,
	}
}

// Type returns the stream type for this handler
func (h *Handler) Type() common.StreamType {
	return common.StreamTypeICEcast
}

// CanHandle determines if this handler can process the given URL
func (h *Handler) CanHandle(ctx context.Context, url string) bool {
	// Use configurable detection for better accuracy
	streamType, err := ConfigurableDetection(ctx, url, h.config)
	if err == nil && streamType == common.StreamTypeICEcast {
		return true
	}

	// Fallback to individual detection methods for backward compatibility
	detector := NewDetectorWithConfig(h.config.Detection)

	if st := detector.DetectFromURL(url); st == common.StreamTypeICEcast {
		return true
	}

	if st := detector.DetectFromHeaders(ctx, url, h.config.HTTP); st == common.StreamTypeICEcast {
		return true
	}

	// Content validation as final check
	return detector.IsValidICEcastContent(ctx, url, h.config.HTTP)
}

// Connect establishes connection to the ICEcast stream
func (h *Handler) Connect(ctx context.Context, url string) error {
	if h.connected {
		return common.NewStreamError(common.StreamTypeICEcast, url,
			common.ErrCodeConnection, "already connected", nil)
	}

	h.url = url
	h.startTime = time.Now()

	logger := logging.WithFields(logging.Fields{
		"component": "icecast_handler",
		"function":  "Connect",
		"url":       url,
	})

	// Create context with configured timeout
	connectCtx, cancel := context.WithTimeout(ctx, h.config.HTTP.ConnectionTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(connectCtx, "GET", url, nil)
	if err != nil {
		return common.NewStreamError(common.StreamTypeICEcast, url,
			common.ErrCodeConnection, "failed to create request", err)
	}

	// Start with minimal headers - some ICEcast streams don't like extra headers
	req.Header.Set("User-Agent", h.config.HTTP.UserAgent)
	req.Header.Set("Accept", "*/*")

	logger.Info("Attempting connection with minimal headers")

	resp, err := h.client.Do(req)
	if err != nil {
		// If the minimal approach fails, log and return error
		logger.Error(err, "Connection failed with minimal headers")
		return common.NewStreamError(common.StreamTypeICEcast, url,
			common.ErrCodeConnection, "connection failed", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()

		// For some ICEcast servers, try with ICY headers if basic request failed
		if resp.StatusCode == http.StatusBadRequest || resp.StatusCode == http.StatusMethodNotAllowed {
			logger.Debug("Basic request failed, trying with ICY headers", logging.Fields{
				"status_code": resp.StatusCode,
			})

			// Try again with ICY metadata header
			req2, err := http.NewRequestWithContext(connectCtx, "GET", url, nil)
			if err != nil {
				return common.NewStreamError(common.StreamTypeICEcast, url,
					common.ErrCodeConnection, "failed to create retry request", err)
			}

			req2.Header.Set("User-Agent", h.config.HTTP.UserAgent)
			req2.Header.Set("Accept", "*/*")
			req2.Header.Set("Icy-MetaData", "1")

			resp2, err := h.client.Do(req2)
			if err != nil {
				return common.NewStreamError(common.StreamTypeICEcast, url,
					common.ErrCodeConnection, "retry connection failed", err)
			}

			if resp2.StatusCode != http.StatusOK {
				resp2.Body.Close()
				return common.NewStreamErrorWithFields(common.StreamTypeICEcast, url,
					common.ErrCodeConnection, fmt.Sprintf("HTTP %d: %s", resp2.StatusCode, resp2.Status), nil,
					logging.Fields{
						"status_code": resp2.StatusCode,
						"status_text": resp2.Status,
					})
			}

			resp = resp2
		} else {
			return common.NewStreamErrorWithFields(common.StreamTypeICEcast, url,
				common.ErrCodeConnection, fmt.Sprintf("HTTP %d: %s", resp.StatusCode, resp.Status), nil,
				logging.Fields{
					"status_code": resp.StatusCode,
					"status_text": resp.Status,
				})
		}
	}

	h.response = resp

	logger.Info("ICEcast connection successful", logging.Fields{
		"status_code":  resp.StatusCode,
		"content_type": resp.Header.Get("Content-Type"),
	})

	// Parse ICY metadata interval if present (might not be present for all streams)
	if metaInt := resp.Header.Get("icy-metaint"); metaInt != "" {
		if interval, err := strconv.Atoi(metaInt); err == nil {
			h.icyMetaInt = interval
			logger.Debug("ICY metadata interval detected", logging.Fields{
				"interval": interval,
			})
		}
	} else {
		logger.Debug("No ICY metadata interval found - stream may not support metadata")
	}

	// Store response headers (may be minimal or empty for some streams)
	headers := make(map[string]string)
	for key, values := range resp.Header {
		if len(values) > 0 {
			headers[strings.ToLower(key)] = values[0]
		}
	}

	// Extract metadata using the configured extractor (handle case where headers are minimal)
	h.metadata = h.metadataExtractor.ExtractMetadata(resp.Header, url)

	// If metadata extraction failed due to missing headers, create basic metadata
	if h.metadata == nil {
		logger.Debug("Creating basic metadata due to minimal headers")
		h.metadata = &common.StreamMetadata{
			URL:         url,
			Type:        common.StreamTypeICEcast,
			ContentType: resp.Header.Get("Content-Type"),
			Headers:     headers,
			Timestamp:   time.Now(),
		}

		// Apply defaults for streams without proper headers
		h.applyDefaultMetadata()
	}

	// Merge HTTP headers with metadata headers
	if h.metadata.Headers == nil {
		h.metadata.Headers = make(map[string]string)
	}
	maps.Copy(h.metadata.Headers, headers)

	h.stats.ConnectionTime = time.Since(h.startTime)
	h.stats.FirstByteTime = h.stats.ConnectionTime
	h.connected = true

	logger.Info("ICEcast handler connected successfully", logging.Fields{
		"has_icy_metadata": h.icyMetaInt > 0,
		"connection_time":  h.stats.ConnectionTime.Milliseconds(),
		"headers_count":    len(headers),
	})

	return nil
}

// GetMetadata retrieves stream metadata
func (h *Handler) GetMetadata() (*common.StreamMetadata, error) {
	if !h.connected {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeConnection, "not connected", nil)
	}

	// Return the metadata extracted during connection or updated during streaming
	if h.metadata == nil {
		// Fallback metadata if extraction failed, using configured defaults
		h.metadata = &common.StreamMetadata{
			URL:       h.url,
			Type:      common.StreamTypeICEcast,
			Headers:   make(map[string]string),
			Timestamp: time.Now(),
		}

		// Apply default values from configuration
		h.applyDefaultMetadata()
	}

	// Update with current ICY title if available
	if h.icyTitle != "" {
		h.metadata.Title = h.icyTitle
		h.metadata.Timestamp = time.Now()
	}

	return h.metadata, nil
}

// applyDefaultMetadata applies default values from configuration
func (h *Handler) applyDefaultMetadata() {
	if defaults := h.config.MetadataExtractor.DefaultValues; defaults != nil {
		if codec, ok := defaults["codec"].(string); ok {
			h.metadata.Codec = codec
		}
		if sampleRate, ok := defaults["sample_rate"].(int); ok {
			h.metadata.SampleRate = sampleRate
		} else if sampleRateFloat, ok := defaults["sample_rate"].(float64); ok {
			h.metadata.SampleRate = int(sampleRateFloat)
		}
		if channels, ok := defaults["channels"].(int); ok {
			h.metadata.Channels = channels
		} else if channelsFloat, ok := defaults["channels"].(float64); ok {
			h.metadata.Channels = int(channelsFloat)
		}
		if bitrate, ok := defaults["bitrate"].(int); ok {
			h.metadata.Bitrate = bitrate
		} else if bitrateFloat, ok := defaults["bitrate"].(float64); ok {
			h.metadata.Bitrate = int(bitrateFloat)
		}
		if format, ok := defaults["format"].(string); ok {
			h.metadata.Format = format
		}
	}
}

// ReadAudio reads audio data from the ICEcast stream
func (h *Handler) ReadAudio(ctx context.Context) (*common.AudioData, error) {
	logger := logging.WithFields(logging.Fields{
		"component": "icecast_handler",
		"function":  "ReadAudio",
	})

	logger.Debug("ReadAudio called", logging.Fields{
		"connected":    h.connected,
		"response_nil": h.response == nil,
		"context_err":  ctx.Err(),
	})

	if !h.connected {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeConnection, "not connected", nil)
	}

	// For Connection: Close servers, always create a fresh connection for audio
	// This completely bypasses any context inheritance issues
	connectionHeader := ""
	if h.response != nil {
		connectionHeader = h.response.Header.Get("Connection")
	}

	needsNewConnection := strings.ToLower(connectionHeader) == "close" || h.response == nil

	if needsNewConnection {
		logger.Debug("Creating dedicated audio connection due to Connection: Close")
		return h.readAudioWithFreshConnection()
	}

	// For persistent connections, try the existing connection first
	return h.readAudioFromExistingConnection()
}

// readAudioWithFreshConnection creates a completely new HTTP connection just for reading audio
func (h *Handler) readAudioWithFreshConnection() (*common.AudioData, error) {
	logger := logging.WithFields(logging.Fields{
		"component": "icecast_handler",
		"function":  "readAudioWithFreshConnection",
		"url":       h.url,
	})

	// Create a completely isolated HTTP client for this audio request
	audioClient := &http.Client{
		Timeout: h.config.HTTP.ConnectionTimeout + h.config.HTTP.ReadTimeout,
		Transport: &http.Transport{
			MaxIdleConns:       1,
			IdleConnTimeout:    10 * time.Second,
			DisableCompression: false,
			DisableKeepAlives:  true, // Important: disable keep-alives for one-shot requests
		},
	}

	// Create a background context with a reasonable timeout for audio reading
	// This is NOT derived from the original context to avoid cancellation inheritance
	audioTimeout := 30 * time.Second // Fixed timeout just for audio reading
	audioCtx, cancel := context.WithTimeout(context.Background(), audioTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(audioCtx, "GET", h.url, nil)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeConnection, "failed to create audio request", err)
	}

	// Use minimal headers
	req.Header.Set("User-Agent", h.config.HTTP.UserAgent)
	req.Header.Set("Accept", "*/*")

	logger.Debug("Making fresh HTTP request for audio data")

	resp, err := audioClient.Do(req)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeConnection, "failed to connect for audio", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeConnection, fmt.Sprintf("HTTP %d: %s", resp.StatusCode, resp.Status), nil)
	}

	logger.Debug("Fresh audio connection successful", logging.Fields{
		"status_code":  resp.StatusCode,
		"content_type": resp.Header.Get("Content-Type"),
	})

	// Read audio data using a simple, direct approach
	return h.readAudioDataFromResponse(resp, logger)
}

// readAudioFromExistingConnection reads from the existing persistent connection
func (h *Handler) readAudioFromExistingConnection() (*common.AudioData, error) {
	logger := logging.WithFields(logging.Fields{
		"component": "icecast_handler",
		"function":  "readAudioFromExistingConnection",
	})

	// Test if the existing connection is still alive
	testBuf := make([]byte, 1)
	_, testErr := h.response.Body.Read(testBuf)
	if testErr != nil {
		logger.Debug("Existing connection failed, creating fresh connection", logging.Fields{
			"test_error": testErr.Error(),
		})
		return h.readAudioWithFreshConnection()
	}

	// Connection is alive, put the test byte back and continue
	multiReader := io.MultiReader(bytes.NewReader(testBuf), h.response.Body)
	h.response.Body = io.NopCloser(multiReader)

	return h.readAudioDataFromResponse(h.response, logger)
}

// readAudioDataFromResponse reads audio data from an HTTP response
func (h *Handler) readAudioDataFromResponse(resp *http.Response, logger logging.Logger) (*common.AudioData, error) {
	bufferSize := h.config.Audio.BufferSize
	if bufferSize <= 0 {
		bufferSize = 8192
	}

	// Read a reasonable amount of audio data quickly
	targetBytes := bufferSize * 3
	maxAttempts := 5
	readTimeout := 10 * time.Second // Short timeout for individual reads

	var allAudioBytes []byte
	attempts := 0
	startTime := time.Now()

	logger.Debug("Starting to read audio data", logging.Fields{
		"buffer_size":  bufferSize,
		"target_bytes": targetBytes,
		"max_attempts": maxAttempts,
		"read_timeout": readTimeout.Seconds(),
	})

	for attempts < maxAttempts && len(allAudioBytes) < targetBytes {
		attempts++

		// Set a read deadline to prevent hanging
		if conn, ok := resp.Body.(interface{ SetReadDeadline(time.Time) error }); ok {
			conn.SetReadDeadline(time.Now().Add(readTimeout))
		}

		buffer := make([]byte, bufferSize)
		n, err := resp.Body.Read(buffer)

		logger.Debug("Read attempt completed", logging.Fields{
			"attempt":     attempts,
			"bytes_read":  n,
			"error":       err,
			"total_bytes": len(allAudioBytes),
		})

		if err != nil {
			if err == io.EOF {
				logger.Debug("Reached end of stream")
				break
			}

			// For any other error, if we have some data, use it
			if len(allAudioBytes) > 0 {
				logger.Debug("Got error but have some data, proceeding", logging.Fields{
					"error":      err.Error(),
					"bytes_have": len(allAudioBytes),
				})
				break
			}

			return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
				common.ErrCodeDecoding, fmt.Sprintf("failed to read audio data: %v", err), err)
		}

		if n > 0 {
			allAudioBytes = append(allAudioBytes, buffer[:n]...)
			logger.Debug("Successfully read audio chunk", logging.Fields{
				"chunk_size":  n,
				"total_bytes": len(allAudioBytes),
				"attempt":     attempts,
			})
		} else {
			logger.Debug("Zero bytes read, stopping")
			break
		}

		// Small delay to avoid busy waiting
		time.Sleep(10 * time.Millisecond)
	}

	if len(allAudioBytes) == 0 {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeDecoding, "no audio data received after all attempts", nil)
	}

	logger.Debug("Audio reading completed", logging.Fields{
		"total_bytes": len(allAudioBytes),
		"attempts":    attempts,
		"read_time":   time.Since(startTime).Seconds(),
	})

	// Update statistics
	h.stats.BytesReceived += int64(len(allAudioBytes))

	// Calculate average bitrate
	elapsed := time.Since(h.startTime)
	if elapsed > 0 {
		h.stats.AverageBitrate = float64(h.stats.BytesReceived*8) / elapsed.Seconds() / 1000
	}

	// Convert to PCM
	pcmSamples, err := h.convertToPCM(allAudioBytes)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeDecoding, "failed to convert audio to PCM", err)
	}

	// Get metadata
	metadata, err := h.GetMetadata()
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeMetadata, "failed to get metadata", err)
	}

	// Create metadata copy
	metadataCopy := *metadata
	metadataCopy.Timestamp = time.Now()

	// Calculate duration
	sampleRate := metadataCopy.SampleRate
	if sampleRate <= 0 {
		sampleRate = 44100
	}

	audioData := &common.AudioData{
		PCM:        pcmSamples,
		SampleRate: sampleRate,
		Channels:   metadataCopy.Channels,
		Duration:   time.Duration(len(pcmSamples)) * time.Second / time.Duration(sampleRate),
		Timestamp:  time.Now(),
		Metadata:   &metadataCopy,
	}

	logger.Debug("Audio processing completed successfully", logging.Fields{
		"samples":          len(pcmSamples),
		"duration_seconds": audioData.Duration.Seconds(),
		"sample_rate":      audioData.SampleRate,
		"channels":         audioData.Channels,
	})

	return audioData, nil
}

// ReadAudioWithDuration reads audio data with a specified duration
// For ICEcast streams, this accumulates audio data over time until target duration is reached
// ReadAudioWithDuration reads audio data with a specified duration using the streaming downloader
func (h *Handler) ReadAudioWithDuration(ctx context.Context, duration time.Duration) (*common.AudioData, error) {
	logger := logging.WithFields(logging.Fields{
		"component":       "icecast_handler",
		"function":        "ReadAudioWithDuration",
		"target_duration": duration.Seconds(),
	})

	if !h.connected {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeConnection, "not connected", nil)
	}

	// Get metadata to determine stream parameters
	metadata, err := h.GetMetadata()
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeMetadata, "failed to get metadata", err)
	}

	logger.Info("Starting duration-based audio reading using streaming downloader", logging.Fields{
		"sample_rate":     metadata.SampleRate,
		"channels":        metadata.Channels,
		"target_duration": duration.Seconds(),
		"connection_type": h.response.Header.Get("Connection"),
	})

	// Use the streaming downloader approach for all ICEcast streams
	// This works for both Connection: Close and persistent connections
	downloader := NewAudioDownloader(h.config)

	// Add retry logic for robustness
	audioData, err := downloader.DownloadAudioSampleWithRetry(ctx, h.url, duration, 2)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeDecoding, "failed to download audio using streaming approach", err)
	}

	// Update handler statistics
	h.stats.BytesReceived += int64(len(audioData.PCM) * 4) // Rough estimate for float64 samples

	// Calculate average bitrate
	elapsed := time.Since(h.startTime)
	if elapsed > 0 {
		h.stats.AverageBitrate = float64(h.stats.BytesReceived*8) / elapsed.Seconds() / 1000
	}

	// Update metadata with current timestamp
	if audioData.Metadata != nil {
		audioData.Metadata.Timestamp = time.Now()

		// Merge with handler metadata for consistency
		if metadata != nil {
			audioData.Metadata.Station = metadata.Station
			audioData.Metadata.Genre = metadata.Genre
			audioData.Metadata.Title = metadata.Title
			if metadata.Bitrate > 0 {
				audioData.Metadata.Bitrate = metadata.Bitrate
			}
		}
	}

	logger.Info("ICEcast duration-based audio extraction completed using streaming downloader", logging.Fields{
		"actual_duration":  audioData.Duration.Seconds(),
		"target_duration":  duration.Seconds(),
		"samples":          len(audioData.PCM),
		"sample_rate":      audioData.SampleRate,
		"channels":         audioData.Channels,
		"efficiency_ratio": audioData.Duration.Seconds() / duration.Seconds(),
	})

	return audioData, nil
}

// readAudioWithDurationEfficient uses fewer, larger requests for Connection: Close servers
func (h *Handler) readAudioWithDurationEfficient(duration time.Duration, sampleRate, channels, targetSamples int, logger logging.Logger) (*common.AudioData, error) {
	var allPCMSamples []float64
	startTime := time.Now()

	// Calculate estimated bytes needed for the duration
	// At 128kbps = 16,000 bytes/second, so for N seconds we need N * 16,000 bytes
	estimatedBytesNeeded := int(duration.Seconds() * 16000) // 128kbps = 16KB/s

	// Use fewer, larger chunks - aim for 3-5 requests total
	targetChunks := 4
	bytesPerChunk := estimatedBytesNeeded / targetChunks
	bytesPerChunk = max(bytesPerChunk, 32768)

	// Maximum number of requests to prevent infinite loops
	maxRequests := 8

	logger.Info("Starting efficient chunked reading", logging.Fields{
		"estimated_bytes": estimatedBytesNeeded,
		"target_chunks":   targetChunks,
		"bytes_per_chunk": bytesPerChunk,
		"max_requests":    maxRequests,
		"target_samples":  targetSamples,
	})

	for requestNum := 0; requestNum < maxRequests && len(allPCMSamples) < targetSamples; requestNum++ {
		// Check if we've exceeded our time budget (with generous buffer)
		elapsed := time.Since(startTime)
		if elapsed > duration+(30*time.Second) { // 30 second buffer
			logger.Warn("Duration reading timed out", logging.Fields{
				"elapsed":           elapsed.Seconds(),
				"target_duration":   duration.Seconds(),
				"samples_collected": len(allPCMSamples),
				"requests_made":     requestNum,
			})
			break
		}

		logger.Info("Making efficient request for large audio chunk", logging.Fields{
			"request_num":       requestNum + 1,
			"samples_collected": len(allPCMSamples),
			"target_samples":    targetSamples,
			"bytes_per_chunk":   bytesPerChunk,
		})

		// Create isolated HTTP client for this chunk
		audioClient := &http.Client{
			Timeout: 30 * time.Second, // Longer timeout for larger chunks
			Transport: &http.Transport{
				MaxIdleConns:       1,
				IdleConnTimeout:    5 * time.Second,
				DisableCompression: false,
				DisableKeepAlives:  true,
			},
		}

		// Use background context to avoid inheritance issues
		audioCtx, cancel := context.WithTimeout(context.Background(), 25*time.Second)

		req, err := http.NewRequestWithContext(audioCtx, "GET", h.url, nil)
		if err != nil {
			cancel()
			logger.Warn("Failed to create request for audio chunk", logging.Fields{
				"error":       err.Error(),
				"request_num": requestNum + 1,
			})
			continue
		}

		req.Header.Set("User-Agent", h.config.HTTP.UserAgent)
		req.Header.Set("Accept", "*/*")

		resp, err := audioClient.Do(req)
		if err != nil {
			cancel()
			logger.Warn("Failed to connect for audio chunk", logging.Fields{
				"error":       err.Error(),
				"request_num": requestNum + 1,
			})
			continue
		}

		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			cancel()
			logger.Warn("Bad status for audio chunk", logging.Fields{
				"status_code": resp.StatusCode,
				"request_num": requestNum + 1,
			})
			continue
		}

		// Read a large chunk of audio data efficiently
		chunkData, err := h.readLargeAudioChunk(resp, bytesPerChunk, logger)
		resp.Body.Close()
		cancel()

		if err != nil {
			logger.Warn("Failed to read large audio chunk", logging.Fields{
				"error":       err.Error(),
				"request_num": requestNum + 1,
			})
			continue
		}

		if len(chunkData) == 0 {
			logger.Info("No data in audio chunk, stopping", logging.Fields{
				"request_num": requestNum + 1,
			})
			break
		}

		// Convert chunk to PCM
		pcmSamples, err := h.convertToPCM(chunkData)
		if err != nil {
			logger.Warn("Failed to convert chunk to PCM", logging.Fields{
				"error":       err.Error(),
				"request_num": requestNum + 1,
				"chunk_size":  len(chunkData),
			})
			continue
		}

		// Accumulate samples
		allPCMSamples = append(allPCMSamples, pcmSamples...)

		// Update statistics
		h.stats.BytesReceived += int64(len(chunkData))

		// Calculate current duration
		currentSamples := len(allPCMSamples)
		if channels > 1 {
			currentSamples = currentSamples / channels
		}
		currentDuration := time.Duration(currentSamples) * time.Second / time.Duration(sampleRate)

		logger.Info("Large audio chunk processed efficiently", logging.Fields{
			"request_num":      requestNum + 1,
			"chunk_bytes":      len(chunkData),
			"chunk_samples":    len(pcmSamples),
			"total_samples":    len(allPCMSamples),
			"current_duration": currentDuration.Seconds(),
			"target_duration":  duration.Seconds(),
			"progress_percent": (currentDuration.Seconds() / duration.Seconds()) * 100,
		})

		// Check if we've reached the target duration
		if currentDuration >= duration {
			logger.Info("Target duration reached efficiently", logging.Fields{
				"final_duration": currentDuration.Seconds(),
				"requests_made":  requestNum + 1,
				"efficiency":     fmt.Sprintf("%.1f seconds of audio in %d requests", currentDuration.Seconds(), requestNum+1),
			})
			break
		}

		// Only a small delay between requests since we're doing fewer of them
		time.Sleep(200 * time.Millisecond)
	}

	return h.finalizeDurationAudio(allPCMSamples, duration, sampleRate, channels, targetSamples, startTime, logger)
}

// readLargeAudioChunk efficiently reads a large chunk of audio data from an HTTP response
func (h *Handler) readLargeAudioChunk(resp *http.Response, targetBytes int, logger logging.Logger) ([]byte, error) {
	bufferSize := 32768 // Use large 32KB buffers for efficiency
	bufferSize = max(h.config.Audio.BufferSize, bufferSize)

	var allBytes []byte
	maxAttempts := (targetBytes / bufferSize) + 5 // Reasonable number of reads
	readTimeout := 20 * time.Second               // Generous timeout for large reads

	logger.Info("Starting large chunk read", logging.Fields{
		"target_bytes": targetBytes,
		"buffer_size":  bufferSize,
		"max_attempts": maxAttempts,
		"read_timeout": readTimeout.Seconds(),
	})

	for attempt := 0; attempt < maxAttempts && len(allBytes) < targetBytes; attempt++ {
		buffer := make([]byte, bufferSize)

		// Set read deadline to prevent hanging on large reads
		if conn, ok := resp.Body.(interface{ SetReadDeadline(time.Time) error }); ok {
			conn.SetReadDeadline(time.Now().Add(readTimeout))
		}

		n, err := resp.Body.Read(buffer)
		if err != nil {
			if err == io.EOF {
				logger.Info("Reached end of stream during large read", logging.Fields{
					"bytes_read": len(allBytes),
					"attempt":    attempt + 1,
				})
				break
			}
			if len(allBytes) > 0 {
				// We have some data, return what we got
				logger.Info("Got error but have data, returning partial chunk", logging.Fields{
					"error":      err.Error(),
					"bytes_read": len(allBytes),
					"attempt":    attempt + 1,
				})
				break
			}
			return nil, err
		}

		if n > 0 {
			allBytes = append(allBytes, buffer[:n]...)

			// Log progress for large reads
			if (attempt+1)%10 == 0 || len(allBytes) >= targetBytes {
				logger.Info("Large chunk read progress", logging.Fields{
					"bytes_read":   len(allBytes),
					"target_bytes": targetBytes,
					"progress_pct": (float64(len(allBytes)) / float64(targetBytes)) * 100,
					"attempt":      attempt + 1,
				})
			}
		} else {
			logger.Info("Zero bytes read, ending chunk", logging.Fields{
				"attempt":    attempt + 1,
				"bytes_read": len(allBytes),
			})
			break
		}

		// If we have enough data, stop reading
		if len(allBytes) >= targetBytes {
			logger.Info("Target bytes reached for chunk", logging.Fields{
				"bytes_read":   len(allBytes),
				"target_bytes": targetBytes,
				"attempts":     attempt + 1,
			})
			break
		}

		// Very small delay between reads within a chunk
		time.Sleep(1 * time.Millisecond)
	}

	logger.Info("Large chunk read completed", logging.Fields{
		"final_bytes":  len(allBytes),
		"target_bytes": targetBytes,
		"efficiency":   fmt.Sprintf("%.1f%% of target", (float64(len(allBytes))/float64(targetBytes))*100),
	})

	return allBytes, nil
}

// readAudioWithDurationPersistent handles duration-based reading for persistent connections
func (h *Handler) readAudioWithDurationPersistent(ctx context.Context, duration time.Duration, sampleRate, channels, targetSamples int, logger logging.Logger) (*common.AudioData, error) {
	// Create context with timeout slightly longer than target duration
	readTimeout := duration + (5 * time.Second)
	readCtx, cancel := context.WithTimeout(ctx, readTimeout)
	defer cancel()

	// Accumulate audio data
	var allPCMSamples []float64
	startTime := time.Now()
	readAttempts := 0
	maxAttempts := int(duration.Seconds()) * 10 // Allow up to 10 attempts per second

	// Flag to control the main loop
	shouldStop := false

	logger.Info("Using persistent connection for duration reading", logging.Fields{
		"read_timeout":   readTimeout.Seconds(),
		"max_attempts":   maxAttempts,
		"target_samples": targetSamples,
	})

	for len(allPCMSamples) < targetSamples && readAttempts < maxAttempts && !shouldStop {
		// Check for context cancellation - FIXED: proper break out of loop
		select {
		case <-readCtx.Done():
			logger.Warn("Duration-based read timed out", logging.Fields{
				"collected_samples": len(allPCMSamples),
				"target_samples":    targetSamples,
				"elapsed_time":      time.Since(startTime).Seconds(),
			})
			if len(allPCMSamples) == 0 {
				return nil, readCtx.Err()
			}
			shouldStop = true // FIXED: Set flag instead of ineffective break
			continue          // Continue to exit the loop properly
		default:
			// Continue with reading
		}

		readAttempts++

		// Use larger buffer size for duration reading
		bufferSize := h.config.Audio.BufferSize
		if bufferSize <= 0 {
			bufferSize = 8192 // Larger default for duration reading
		}

		buffer := make([]byte, bufferSize)

		// Handle ICY metadata if present and configured
		var audioBytes []byte
		var err error
		if h.icyMetaInt > 0 && h.config.Audio.HandleICYMeta {
			audioBytes, err = h.readWithICYMetadata(buffer)
			if err != nil {
				if err == io.EOF {
					shouldStop = true
					continue
				}
				logger.Warn("Error reading with ICY metadata", logging.Fields{
					"error": err.Error(),
				})
				continue
			}
		} else {
			// Simple read without ICY metadata handling
			n, err := h.response.Body.Read(buffer)
			if err != nil {
				if err == io.EOF {
					shouldStop = true
					continue
				}
				if err == context.DeadlineExceeded || err == context.Canceled {
					shouldStop = true
					continue
				}
				logger.Warn("Error reading audio data", logging.Fields{
					"error": err.Error(),
				})
				continue
			}
			if n == 0 {
				shouldStop = true
				continue
			}
			audioBytes = buffer[:n]
		}

		if len(audioBytes) == 0 {
			continue
		}

		// Convert to PCM
		pcmSamples, err := h.convertToPCM(audioBytes)
		if err != nil {
			logger.Warn("Failed to convert audio to PCM", logging.Fields{
				"error": err.Error(),
			})
			continue
		}

		// Accumulate samples
		allPCMSamples = append(allPCMSamples, pcmSamples...)

		// Update statistics
		h.stats.BytesReceived += int64(len(audioBytes))

		// Calculate current duration
		currentSamples := len(allPCMSamples)
		if channels > 1 {
			currentSamples = currentSamples / channels
		}
		totalDuration := time.Duration(currentSamples) * time.Second / time.Duration(sampleRate)

		// Log progress less frequently for efficiency
		if readAttempts%50 == 0 || totalDuration >= duration {
			logger.Info("Persistent connection progress", logging.Fields{
				"current_samples":  len(allPCMSamples),
				"target_samples":   targetSamples,
				"current_duration": totalDuration.Seconds(),
				"target_duration":  duration.Seconds(),
				"read_attempts":    readAttempts,
				"progress_percent": (totalDuration.Seconds() / duration.Seconds()) * 100,
			})
		}

		// Check if we've reached the target duration
		if totalDuration >= duration {
			shouldStop = true
			continue
		}

		// Small delay to avoid busy waiting
		time.Sleep(10 * time.Millisecond)
	}

	return h.finalizeDurationAudio(allPCMSamples, duration, sampleRate, channels, targetSamples, startTime, logger)
}

// finalizeDurationAudio creates the final AudioData object from accumulated samples
func (h *Handler) finalizeDurationAudio(allPCMSamples []float64, duration time.Duration, sampleRate, channels, targetSamples int, startTime time.Time, logger logging.Logger) (*common.AudioData, error) {
	// Trim to exact target duration if we have more than needed
	if len(allPCMSamples) > targetSamples {
		allPCMSamples = allPCMSamples[:targetSamples]
	}

	// Calculate final duration based on actual samples
	actualSamples := len(allPCMSamples)
	if channels > 1 {
		actualSamples = actualSamples / channels
	}
	finalDuration := time.Duration(actualSamples) * time.Second / time.Duration(sampleRate)

	// If we got close to the target duration, use the target duration
	if finalDuration >= duration*90/100 { // Within 90% of target
		finalDuration = duration
	}

	// Calculate average bitrate
	elapsed := time.Since(startTime)
	if elapsed > 0 {
		h.stats.AverageBitrate = float64(h.stats.BytesReceived*8) / elapsed.Seconds() / 1000
	}

	// Get current metadata
	metadata, err := h.GetMetadata()
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeMetadata, "failed to get metadata", err)
	}

	// Create a copy of metadata to avoid concurrent modification
	metadataCopy := *metadata
	metadataCopy.Timestamp = time.Now()

	audioData := &common.AudioData{
		PCM:        allPCMSamples,
		SampleRate: sampleRate,
		Channels:   channels,
		Duration:   finalDuration,
		Timestamp:  time.Now(),
		Metadata:   &metadataCopy,
	}

	logger.Info("ICEcast duration-based audio extraction completed", logging.Fields{
		"target_duration":  duration.Seconds(),
		"actual_duration":  audioData.Duration.Seconds(),
		"samples":          len(audioData.PCM),
		"sample_rate":      audioData.SampleRate,
		"channels":         audioData.Channels,
		"extraction_time":  time.Since(startTime).Seconds(),
		"efficiency_ratio": audioData.Duration.Seconds() / time.Since(startTime).Seconds(),
	})

	return audioData, nil
}

// readWithICYMetadata handles reading audio data with embedded ICY metadata
func (h *Handler) readWithICYMetadata(buffer []byte) ([]byte, error) {
	var audioData []byte
	maxRetries := h.config.Audio.MaxReadAttempts

	for len(audioData) < len(buffer) && maxRetries > 0 {
		// Calculate how many audio bytes to read before next metadata block
		audioToRead := h.icyMetaInt - int(h.bytesRead%int64(h.icyMetaInt))
		audioToRead = min(audioToRead, len(buffer)-len(audioData))

		// Read audio data
		audioChunk := make([]byte, audioToRead)
		n, err := h.response.Body.Read(audioChunk)
		if err != nil && err != io.EOF {
			maxRetries--
			if maxRetries <= 0 {
				return nil, err
			}
			continue
		}
		if n == 0 {
			break
		}

		audioData = append(audioData, audioChunk[:n]...)
		h.bytesRead += int64(n)

		// Check if we need to read metadata
		if h.bytesRead%int64(h.icyMetaInt) == 0 {
			if err := h.readICYMetadata(); err != nil {
				logging.Warn("Failed to read ICY metadata", logging.Fields{
					"url":   h.url,
					"error": err.Error(),
				})
			}
		}

		if n < audioToRead {
			break // End of stream or partial read
		}
	}

	return audioData, nil
}

// readICYMetadata reads and parses ICY metadata from the stream
func (h *Handler) readICYMetadata() error {
	// Read metadata length byte
	lengthByte := make([]byte, 1)
	_, err := io.ReadFull(h.response.Body, lengthByte)
	if err != nil {
		return err
	}

	metadataLength := int(lengthByte[0]) * 16
	if metadataLength == 0 {
		return nil // No metadata
	}

	// Read metadata block
	metadataBlock := make([]byte, metadataLength)
	_, err = io.ReadFull(h.response.Body, metadataBlock)
	if err != nil {
		return err
	}

	// Parse metadata (typically in format "StreamTitle='Artist - Title';")
	metadataStr := string(metadataBlock)
	metadataStr = strings.TrimRight(metadataStr, "\x00") // Remove null padding

	if strings.Contains(metadataStr, "StreamTitle=") {
		start := strings.Index(metadataStr, "StreamTitle='")
		if start != -1 {
			start += len("StreamTitle='")
			end := strings.Index(metadataStr[start:], "'")
			if end != -1 {
				newTitle := metadataStr[start : start+end]
				if newTitle != h.icyTitle {
					h.icyTitle = newTitle
					// Update metadata with new title
					if h.metadata != nil {
						h.metadataExtractor.UpdateWithICYMetadata(h.metadata, h.icyTitle)
					}
				}
			}
		}
	}

	return nil
}

// convertToPCM converts raw audio bytes to PCM samples
func (h *Handler) convertToPCM(audioBytes []byte) ([]float64, error) {
	// TODO: decode ICEcast
	// This is a placeholder implementation
	// In a real implementation, you would use LibAV or similar to decode the audio
	// For now, we'll simulate PCM conversion based on the assumed format

	if h.metadata == nil || h.metadata.Codec == "" {
		return nil, common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeDecoding, "no codec information available for PCM conversion", nil)
	}

	// For demonstration, assume 16-bit samples and convert to normalized float64
	// Real implementation would use proper audio decoding libraries
	channels := h.metadata.Channels
	if channels <= 0 {
		channels = 2 // Default to stereo
	}

	sampleCount := len(audioBytes) / 2 // Assuming 16-bit samples
	if channels > 1 {
		sampleCount /= channels
	}

	pcmSamples := make([]float64, sampleCount)
	for i := 0; i < sampleCount && i*2+1 < len(audioBytes); i++ {
		// Convert little-endian 16-bit to normalized float64
		sample := int16(audioBytes[i*2]) | int16(audioBytes[i*2+1])<<8
		pcmSamples[i] = float64(sample) / 32768.0
	}

	return pcmSamples, nil
}

// GetStats returns current streaming statistics
func (h *Handler) GetStats() *common.StreamStats {
	// Update buffer health based on connection state and configuration
	if h.connected && h.response != nil {
		h.stats.BufferHealth = 1.0 // Simplified - real implementation would check buffer levels
	} else {
		h.stats.BufferHealth = 0.0
	}

	return h.stats
}

// Close closes the stream connection
func (h *Handler) Close() error {
	h.connected = false
	if h.response != nil {
		err := h.response.Body.Close()
		h.response = nil
		return err
	}
	return nil
}

// GetClient returns the HTTP Client
func (h *Handler) GetClient() *http.Client {
	return h.client
}

// GetConfig returns the current configuration
func (h *Handler) GetConfig() *Config {
	return h.config
}

// UpdateConfig updates the handler configuration
func (h *Handler) UpdateConfig(config *Config) {
	if config == nil {
		return
	}

	h.config = config

	// Update HTTP client timeout if changed
	totalTimeout := config.HTTP.ConnectionTimeout + config.HTTP.ReadTimeout
	h.client.Timeout = totalTimeout

	// Recreate metadata extractor with new config
	h.metadataExtractor = NewConfigurableMetadataExtractor(config.MetadataExtractor).MetadataExtractor
}

// IsConfigured returns true if the handler has a non-default configuration
func (h *Handler) IsConfigured() bool {
	if h.config == nil {
		return false
	}
	defaultConfig := DefaultConfig()
	return !reflect.DeepEqual(h.config, defaultConfig)
}

// GetConfiguredUserAgent returns the configured user agent
func (h *Handler) GetConfiguredUserAgent() string {
	if h.config != nil && h.config.HTTP != nil {
		return h.config.HTTP.UserAgent
	}
	return "TuneIn-CDN-Benchmark/1.0"
}

// GetCurrentICYTitle returns the current ICY title from metadata
func (h *Handler) GetCurrentICYTitle() string {
	return h.icyTitle
}

// GetICYMetadataInterval returns the ICY metadata interval
func (h *Handler) GetICYMetadataInterval() int {
	return h.icyMetaInt
}

// HasICYMetadata returns true if the stream supports ICY metadata
func (h *Handler) HasICYMetadata() bool {
	return h.icyMetaInt > 0
}

// RefreshMetadata manually refreshes metadata from headers (useful for live streams)
func (h *Handler) RefreshMetadata() error {
	if !h.connected || h.response == nil {
		return common.NewStreamError(common.StreamTypeICEcast, h.url,
			common.ErrCodeConnection, "not connected", nil)
	}

	// Re-extract metadata from current response headers
	h.metadata = h.metadataExtractor.ExtractMetadata(h.response.Header, h.url)

	return nil
}

// GetStreamInfo returns detailed information about the current stream
func (h *Handler) GetStreamInfo() map[string]any {
	info := make(map[string]any)

	if h.metadata != nil {
		info["station"] = h.metadata.Station
		info["genre"] = h.metadata.Genre
		info["bitrate"] = h.metadata.Bitrate
		info["sample_rate"] = h.metadata.SampleRate
		info["channels"] = h.metadata.Channels
		info["codec"] = h.metadata.Codec
		info["format"] = h.metadata.Format
	}

	info["icy_metadata_interval"] = h.icyMetaInt
	info["current_title"] = h.icyTitle
	info["has_icy_metadata"] = h.HasICYMetadata()
	info["bytes_read"] = h.bytesRead
	info["connected"] = h.connected

	if h.stats != nil {
		info["bytes_received"] = h.stats.BytesReceived
		info["average_bitrate"] = h.stats.AverageBitrate
		info["connection_time"] = h.stats.ConnectionTime
		info["buffer_health"] = h.stats.BufferHealth
	}

	return info
}

// GetResponseHeaders returns the HTTP response headers from the stream connection
func (h *Handler) GetResponseHeaders() map[string]string {
	if h.response == nil {
		return make(map[string]string)
	}

	headers := make(map[string]string)
	for key, values := range h.response.Header {
		if len(values) > 0 {
			headers[strings.ToLower(key)] = values[0]
		}
	}

	return headers
}

// GetConnectionTime returns the time elapsed since connection
func (h *Handler) GetConnectionTime() time.Duration {
	if !h.connected {
		return 0
	}
	return time.Since(h.startTime)
}

// GetBytesPerSecond returns the current bytes per second rate
func (h *Handler) GetBytesPerSecond() float64 {
	if !h.connected || h.stats.BytesReceived == 0 {
		return 0
	}

	elapsed := time.Since(h.startTime)
	if elapsed <= 0 {
		return 0
	}

	return float64(h.stats.BytesReceived) / elapsed.Seconds()
}

// IsLive returns whether this appears to be a live stream (always true for ICEcast)
func (h *Handler) IsLive() bool {
	return true // ICEcast streams are typically live
}

// GetAudioFormat returns the detected audio format information
func (h *Handler) GetAudioFormat() map[string]any {
	if h.metadata == nil {
		return make(map[string]any)
	}

	return map[string]any{
		"codec":       h.metadata.Codec,
		"format":      h.metadata.Format,
		"bitrate":     h.metadata.Bitrate,
		"sample_rate": h.metadata.SampleRate,
		"channels":    h.metadata.Channels,
	}
}
