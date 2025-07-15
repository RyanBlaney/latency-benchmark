package icecast

import (
	"context"
	"fmt"
	"io"
	"maps"
	"net/http"
	"time"

	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/transcode"
)

// AudioDownloader handles continuous streaming download from ICEcast servers
type AudioDownloader struct {
	client  *http.Client
	config  *Config
	decoder *transcode.Decoder
}

// NewAudioDownloader creates a new ICEcast audio downloader
func NewAudioDownloader(config *Config) *AudioDownloader {
	if config == nil {
		config = DefaultConfig()
	}

	client := &http.Client{
		Timeout: 0, // No timeout for streaming connections
		Transport: &http.Transport{
			MaxIdleConns:       1,
			IdleConnTimeout:    30 * time.Second,
			DisableCompression: false,
			DisableKeepAlives:  false, // Allow keep-alives for streaming
		},
	}

	// Create FFmpeg decoder optimized for news/talk content (typical for ICEcast)
	decoder := transcode.NewNormalizingDecoder("news")

	return &AudioDownloader{
		client:  client,
		config:  config,
		decoder: decoder,
	}
}

// NewAudioDownloaderForContent creates a new ICEcast audio downloader optimized for specific content
func NewAudioDownloaderForContent(config *Config, contentType string) *AudioDownloader {
	if config == nil {
		config = DefaultConfig()
	}

	client := &http.Client{
		Timeout: 0, // No timeout for streaming connections
		Transport: &http.Transport{
			MaxIdleConns:       1,
			IdleConnTimeout:    30 * time.Second,
			DisableCompression: false,
			DisableKeepAlives:  false, // Allow keep-alives for streaming
		},
	}

	// Create FFmpeg decoder optimized for the specified content type
	decoder := transcode.NewNormalizingDecoder(contentType)

	return &AudioDownloader{
		client:  client,
		config:  config,
		decoder: decoder,
	}
}

// DownloadAudioSample downloads a continuous stream of audio for the specified duration
func (d *AudioDownloader) DownloadAudioSample(ctx context.Context, url string, targetDuration time.Duration) (*common.AudioData, error) {
	logger := logging.WithFields(logging.Fields{
		"component":       "icecast_downloader",
		"function":        "DownloadAudioSample",
		"url":             url,
		"target_duration": targetDuration.Seconds(),
	})

	logger.Info("Starting ICEcast streaming download")

	// Create a context with reasonable timeout (target duration + buffer)
	downloadTimeout := targetDuration + (30 * time.Second)
	downloadCtx, cancel := context.WithTimeout(ctx, downloadTimeout)
	defer cancel()

	// Create HTTP request for streaming
	req, err := http.NewRequestWithContext(downloadCtx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create streaming request: %w", err)
	}

	// Set headers for streaming
	req.Header.Set("User-Agent", d.config.HTTP.UserAgent)
	req.Header.Set("Accept", "*/*")
	req.Header.Set("Connection", "keep-alive")

	logger.Info("Establishing streaming connection")

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to establish streaming connection: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("streaming request failed with status %d: %s", resp.StatusCode, resp.Status)
	}

	logger.Info("Streaming connection established", logging.Fields{
		"status_code":  resp.StatusCode,
		"content_type": resp.Header.Get("Content-Type"),
		"connection":   resp.Header.Get("Connection"),
	})

	// Calculate estimated bytes needed
	// 128kbps = 16KB/s, add buffer for reliability
	estimatedBytesPerSecond := 16000 // 128kbps / 8
	estimatedTotalBytes := int(targetDuration.Seconds() * float64(estimatedBytesPerSecond))
	bufferMultiplier := 1.5 // 50% buffer
	targetBytes := int(float64(estimatedTotalBytes) * bufferMultiplier)

	logger.Info("Starting continuous audio download", logging.Fields{
		"estimated_bytes_per_sec":  estimatedBytesPerSecond,
		"estimated_total_bytes":    estimatedTotalBytes,
		"target_bytes_with_buffer": targetBytes,
	})

	// Download audio data continuously
	audioData, err := d.streamAudioData(resp, targetBytes, targetDuration, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to stream audio data: %w", err)
	}

	logger.Info("ICEcast streaming download completed", logging.Fields{
		"bytes_downloaded": len(audioData),
		"target_bytes":     targetBytes,
		"efficiency":       fmt.Sprintf("%.1f%%", (float64(len(audioData))/float64(targetBytes))*100),
	})

	// Convert raw audio bytes using FFmpeg decoder
	return d.processStreamedAudioWithFFmpeg(audioData, url, logger)
}

// streamAudioData continuously reads audio data from the stream
func (d *AudioDownloader) streamAudioData(resp *http.Response, targetBytes int, targetDuration time.Duration, logger logging.Logger) ([]byte, error) {
	var audioData []byte

	// Use larger buffer for streaming efficiency
	bufferSize := 65536 // 64KB buffer for efficient streaming
	bufferSize = max(d.config.Audio.BufferSize, bufferSize)

	startTime := time.Now()
	readCount := 0
	totalReads := (targetBytes / bufferSize) + 5 // Estimate reads needed + buffer

	logger.Info("Starting streaming read loop", logging.Fields{
		"buffer_size":     bufferSize,
		"target_bytes":    targetBytes,
		"estimated_reads": totalReads,
	})

	for len(audioData) < targetBytes {
		// Check if we've exceeded our time budget
		elapsed := time.Since(startTime)
		if elapsed > targetDuration+(10*time.Second) {
			logger.Info("Streaming reached time limit", logging.Fields{
				"elapsed":         elapsed.Seconds(),
				"target_duration": targetDuration.Seconds(),
				"bytes_collected": len(audioData),
			})
			break
		}

		readCount++
		buffer := make([]byte, bufferSize)

		// Set a reasonable read timeout
		deadline := time.Now().Add(15 * time.Second)
		if conn, ok := resp.Body.(interface{ SetReadDeadline(time.Time) error }); ok {
			conn.SetReadDeadline(deadline)
		}

		n, err := resp.Body.Read(buffer)

		if err != nil {
			if err == io.EOF {
				logger.Info("Reached end of stream", logging.Fields{
					"bytes_collected": len(audioData),
					"reads_completed": readCount,
				})
				break
			}

			// For streaming, some read errors are recoverable
			logger.Warn("Read error during streaming", logging.Fields{
				"error":        err.Error(),
				"bytes_so_far": len(audioData),
				"read_count":   readCount,
			})

			// If we have substantial data, we can continue
			if len(audioData) > targetBytes/4 { // At least 25% of target
				logger.Info("Continuing with partial data due to read error")
				break
			}

			return nil, fmt.Errorf("streaming read failed: %w", err)
		}

		if n > 0 {
			audioData = append(audioData, buffer[:n]...)

			// Log progress periodically
			if readCount%20 == 0 || len(audioData) >= targetBytes {
				progress := (float64(len(audioData)) / float64(targetBytes)) * 100
				rate := float64(len(audioData)) / elapsed.Seconds() / 1024 // KB/s

				logger.Info("Streaming progress", logging.Fields{
					"bytes_collected": len(audioData),
					"target_bytes":    targetBytes,
					"progress_pct":    fmt.Sprintf("%.1f%%", progress),
					"read_count":      readCount,
					"rate_kbps":       fmt.Sprintf("%.1f", rate),
					"elapsed_sec":     elapsed.Seconds(),
				})
			}
		} else {
			logger.Warn("Zero bytes read from stream", logging.Fields{
				"read_count": readCount,
			})
		}

		// Small delay to prevent overwhelming the stream
		time.Sleep(1 * time.Millisecond)
	}

	finalElapsed := time.Since(startTime)
	finalRate := float64(len(audioData)) / finalElapsed.Seconds() / 1024

	logger.Info("Streaming read completed", logging.Fields{
		"final_bytes":   len(audioData),
		"target_bytes":  targetBytes,
		"total_reads":   readCount,
		"elapsed_sec":   finalElapsed.Seconds(),
		"avg_rate_kbps": fmt.Sprintf("%.1f", finalRate),
		"efficiency":    fmt.Sprintf("%.1f%%", (float64(len(audioData))/float64(targetBytes))*100),
	})

	return audioData, nil
}

// processStreamedAudioWithFFmpeg converts streamed audio bytes using FFmpeg decoder
func (d *AudioDownloader) processStreamedAudioWithFFmpeg(audioBytes []byte, url string, logger logging.Logger) (*common.AudioData, error) {
	if len(audioBytes) == 0 {
		return nil, fmt.Errorf("no audio data received from stream")
	}

	logger.Info("Processing streamed audio with FFmpeg decoder", logging.Fields{
		"raw_bytes": len(audioBytes),
	})

	// Use FFmpeg decoder to properly decode MP3 data
	ad, err := d.decoder.DecodeBytes(audioBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to decode streamed audio with FFmpeg: %w", err)
	}
	audioData, ok := ad.(*common.AudioData)
	if !ok {
		return &common.AudioData{}, fmt.Errorf("failed to convert to AudioData")
	}

	logger.Info("FFmpeg decoding completed", logging.Fields{
		"decoded_samples":     len(audioData.PCM),
		"decoded_duration":    audioData.Duration.Seconds(),
		"decoded_channels":    audioData.Channels,
		"decoded_sample_rate": audioData.SampleRate,
	})

	// Convert from transcode.AudioData to common.AudioData
	commonAudioData := &common.AudioData{
		PCM:        audioData.PCM,
		SampleRate: audioData.SampleRate,
		Channels:   audioData.Channels,
		Duration:   audioData.Duration,
		Timestamp:  time.Now(),
		Metadata:   d.convertMetadata(audioData.Metadata, url),
	}

	logger.Info("Streamed audio processing completed with FFmpeg", logging.Fields{
		"final_samples":         len(commonAudioData.PCM),
		"final_duration_sec":    commonAudioData.Duration.Seconds(),
		"final_sample_rate":     commonAudioData.SampleRate,
		"final_channels":        commonAudioData.Channels,
		"normalization_applied": audioData.Metadata != nil && audioData.Metadata.Headers["normalization_applied"] == "true",
	})

	return commonAudioData, nil
}

// convertMetadata converts from transcode.StreamMetadata to common.StreamMetadata
func (d *AudioDownloader) convertMetadata(transcodeMetadata *common.StreamMetadata, url string) *common.StreamMetadata {
	if transcodeMetadata == nil {
		// Create basic metadata if none provided
		return &common.StreamMetadata{
			URL:         url,
			Type:        common.StreamTypeICEcast,
			ContentType: "audio/mpeg",
			Format:      "mp3",
			Codec:       "mp3",
			SampleRate:  44100,
			Channels:    2,
			Bitrate:     128,
			Timestamp:   time.Now(),
		}
	}

	// Convert transcode metadata to common metadata
	headers := make(map[string]string)
	if transcodeMetadata.Headers != nil {
		maps.Copy(headers, transcodeMetadata.Headers)
	}

	return &common.StreamMetadata{
		URL:         url,
		Type:        common.StreamTypeICEcast,
		Format:      transcodeMetadata.Format,
		Bitrate:     transcodeMetadata.Bitrate,
		SampleRate:  transcodeMetadata.SampleRate,
		Channels:    transcodeMetadata.Channels,
		Codec:       transcodeMetadata.Codec,
		ContentType: transcodeMetadata.ContentType,
		Title:       transcodeMetadata.Title,
		Artist:      transcodeMetadata.Artist,
		Genre:       transcodeMetadata.Genre,
		Station:     transcodeMetadata.Station,
		Headers:     headers,
		Timestamp:   time.Now(),
	}
}

// Note: convertToPCM method removed - now using FFmpeg decoder instead

// DownloadAudioSampleWithRetry downloads audio with automatic retry logic
func (d *AudioDownloader) DownloadAudioSampleWithRetry(ctx context.Context, url string, targetDuration time.Duration, maxRetries int) (*common.AudioData, error) {
	logger := logging.WithFields(logging.Fields{
		"component":       "icecast_downloader",
		"function":        "DownloadAudioSampleWithRetry",
		"max_retries":     maxRetries,
		"target_duration": targetDuration.Seconds(),
	})

	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			logger.Info("Retrying ICEcast download", logging.Fields{
				"attempt":     attempt + 1,
				"max_retries": maxRetries + 1,
				"last_error":  lastErr.Error(),
			})

			// Wait a bit before retrying
			time.Sleep(time.Duration(attempt) * time.Second)
		}

		audioData, err := d.DownloadAudioSample(ctx, url, targetDuration)
		if err == nil {
			if attempt > 0 {
				logger.Info("ICEcast download succeeded after retry", logging.Fields{
					"successful_attempt": attempt + 1,
				})
			}
			return audioData, nil
		}

		lastErr = err
		logger.Warn("ICEcast download attempt failed", logging.Fields{
			"attempt": attempt + 1,
			"error":   err.Error(),
		})
	}

	return nil, fmt.Errorf("failed after %d attempts, last error: %w", maxRetries+1, lastErr)
}
