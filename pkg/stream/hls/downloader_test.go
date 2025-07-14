package hls

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tunein/cdn-benchmark-cli/pkg/logging"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

// Mock audio decoder for testing
type mockAudioDecoder struct {
	shouldFail bool
	sampleRate int
	channels   int
	duration   time.Duration
}

func (m *mockAudioDecoder) DecodeBytes(data []byte) (*common.AudioData, error) {
	if m.shouldFail {
		return nil, common.NewStreamError(common.StreamTypeHLS, "",
			common.ErrCodeDecoding, "mock decoder failure", nil)
	}

	// Generate mock PCM data
	samples := int(float64(m.sampleRate) * m.duration.Seconds())
	pcm := make([]float64, samples*m.channels)

	// Add some non-silent data
	for i := range pcm {
		pcm[i] = 0.1 * float64(i%100) / 100.0
	}

	return &common.AudioData{
		PCM:        pcm,
		SampleRate: m.sampleRate,
		Channels:   m.channels,
		Duration:   m.duration,
		Timestamp:  time.Now(),
		Metadata: &common.StreamMetadata{
			Type:       common.StreamTypeHLS,
			Codec:      "mock",
			SampleRate: m.sampleRate,
			Channels:   m.channels,
		},
	}, nil
}

func (m *mockAudioDecoder) DecodeReader(reader io.Reader) (*common.AudioData, error) {
	// Read all data from reader and delegate to DecodeBytes
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, common.NewStreamError(common.StreamTypeHLS, "",
			common.ErrCodeDecoding, "failed to read from reader", err)
	}

	return m.DecodeBytes(data)
}

func (m *mockAudioDecoder) GetConfig() map[string]any {
	return map[string]any{
		"type":        "mock",
		"sample_rate": m.sampleRate,
		"channels":    m.channels,
		"duration":    m.duration,
		"should_fail": m.shouldFail,
	}
}

func TestNewAudioDownloader(t *testing.T) {
	client := &http.Client{}
	config := DefaultDownloadConfig()
	hlsConfig := DefaultConfig()

	downloader := NewAudioDownloader(client, config, hlsConfig)

	assert.NotNil(t, downloader)
	assert.Equal(t, client, downloader.client)
	assert.Equal(t, config, downloader.config)
	assert.Equal(t, hlsConfig, downloader.hlsConfig)
	assert.NotNil(t, downloader.segmentCache)
	assert.NotNil(t, downloader.downloadStats)
	assert.NotEmpty(t, downloader.tempDir)
}

func TestNewAudioDownloaderWithNilConfig(t *testing.T) {
	client := &http.Client{}

	downloader := NewAudioDownloader(client, nil, nil)

	assert.NotNil(t, downloader)
	assert.NotNil(t, downloader.config)
	assert.Equal(t, DefaultDownloadConfig(), downloader.config)
}

func TestDefaultDownloadConfig(t *testing.T) {
	config := DefaultDownloadConfig()

	assert.Equal(t, 10, config.MaxSegments)
	assert.Equal(t, 10*time.Second, config.SegmentTimeout)
	assert.Equal(t, 3, config.MaxRetries)
	assert.True(t, config.CacheSegments)
	assert.Equal(t, 30*time.Second, config.TargetDuration)
	assert.Equal(t, 128, config.PreferredBitrate)
	assert.Equal(t, 44100, config.OutputSampleRate)
	assert.Equal(t, 1, config.OutputChannels)
	assert.True(t, config.NormalizePCM)
	assert.Equal(t, "medium", config.ResampleQuality)
	assert.True(t, config.CleanupTempFiles)
}

func TestDownloadAudioSegment(t *testing.T) {
	t.Run("successful download with mock decoder", func(t *testing.T) {
		// Create mock AAC segment data
		mockAACData := []byte{0xFF, 0xF1, 0x50, 0x80, 0x43, 0x80, 0x00, 0x00} // AAC ADTS header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "audio/aac")
			w.WriteHeader(http.StatusOK)
			w.Write(mockAACData)
		}))
		defer server.Close()

		client := &http.Client{Timeout: 5 * time.Second}
		config := DefaultDownloadConfig()
		hlsConfig := DefaultConfig()

		// Inject mock decoder
		hlsConfig.AudioDecoder = &mockAudioDecoder{
			sampleRate: 44100,
			channels:   2,
			duration:   2 * time.Second,
		}

		downloader := NewAudioDownloader(client, config, hlsConfig)

		ctx := context.Background()
		audioData, err := downloader.DownloadAudioSegment(ctx, server.URL)

		require.NoError(t, err)
		require.NotNil(t, audioData)

		// Verify we got valid data (not exact values)
		assert.Greater(t, audioData.SampleRate, 0)
		assert.Greater(t, audioData.Channels, 0)
		assert.Greater(t, audioData.Duration, time.Duration(0))
		assert.NotEmpty(t, audioData.PCM)
		assert.NotNil(t, audioData.Metadata)

		// Check download stats were updated
		stats := downloader.GetDownloadStats()
		assert.Equal(t, 1, stats.SegmentsDownloaded)
		assert.Greater(t, stats.BytesDownloaded, int64(0))
	})

	t.Run("fallback to basic extraction", func(t *testing.T) {
		mockAACData := []byte{0xFF, 0xF1, 0x50, 0x80, 0x43, 0x80, 0x00, 0x00}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "audio/aac")
			w.WriteHeader(http.StatusOK)
			w.Write(mockAACData)
		}))
		defer server.Close()

		client := &http.Client{Timeout: 5 * time.Second}
		config := DefaultDownloadConfig()
		hlsConfig := DefaultConfig()

		// No decoder - should fallback to basic extraction
		downloader := NewAudioDownloader(client, config, hlsConfig)

		ctx := context.Background()
		audioData, err := downloader.DownloadAudioSegment(ctx, server.URL)

		require.NoError(t, err)
		require.NotNil(t, audioData)

		// Should get valid fallback data
		assert.Equal(t, config.OutputSampleRate, audioData.SampleRate)
		assert.Equal(t, config.OutputChannels, audioData.Channels)
		assert.NotEmpty(t, audioData.PCM)
		assert.NotNil(t, audioData.Metadata)
	})

	t.Run("HTTP error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
		}))
		defer server.Close()

		client := &http.Client{Timeout: 5 * time.Second}
		downloader := NewAudioDownloader(client, nil, nil)

		ctx := context.Background()
		audioData, err := downloader.DownloadAudioSegment(ctx, server.URL)

		assert.Error(t, err)
		assert.Nil(t, audioData)
	})
}

func TestBasicAudioExtraction(t *testing.T) {
	downloader := NewAudioDownloader(&http.Client{}, nil, nil)

	t.Run("AAC extraction", func(t *testing.T) {
		// Mock AAC ADTS data with sync words
		aacData := []byte{
			0xFF, 0xF1, 0x50, 0x80, 0x43, 0x80, 0x00, 0x00, // First frame
			0xFF, 0xF1, 0x50, 0x80, 0x43, 0x80, 0x00, 0x00, // Second frame
		}

		audioData, err := downloader.basicAudioExtraction(aacData, "test.aac")

		require.NoError(t, err)
		require.NotNil(t, audioData)

		// Should get valid AAC data
		assert.Greater(t, audioData.SampleRate, 0)
		assert.Greater(t, audioData.Channels, 0)
		assert.Greater(t, audioData.Duration, time.Duration(0))
		assert.Equal(t, "aac", audioData.Metadata.Codec)
		assert.NotEmpty(t, audioData.PCM)
	})

	t.Run("MP3 extraction", func(t *testing.T) {
		// Mock MP3 data with sync words
		mp3Data := []byte{
			0xFF, 0xFB, 0x90, 0x00, // MP3 sync word and header
			0xFF, 0xFB, 0x90, 0x00, // Second frame
		}

		audioData, err := downloader.basicAudioExtraction(mp3Data, "test.mp3")

		require.NoError(t, err)
		require.NotNil(t, audioData)

		// Should get valid MP3 data
		assert.Greater(t, audioData.SampleRate, 0)
		assert.Greater(t, audioData.Channels, 0)
		assert.Greater(t, audioData.Duration, time.Duration(0))
		assert.Equal(t, "mp3", audioData.Metadata.Codec)
		assert.NotEmpty(t, audioData.PCM)
	})

	t.Run("unknown format fallback", func(t *testing.T) {
		unknownData := []byte{0x00, 0x01, 0x02, 0x03}

		audioData, err := downloader.basicAudioExtraction(unknownData, "test.unknown")

		require.NoError(t, err)
		require.NotNil(t, audioData)

		// Should get fallback silence
		assert.Greater(t, audioData.SampleRate, 0)
		assert.Greater(t, audioData.Channels, 0)
		assert.Equal(t, 2*time.Second, audioData.Duration) // Should generate 2 seconds of silence
		assert.Equal(t, "unknown", audioData.Metadata.Codec)
		assert.NotEmpty(t, audioData.PCM)
	})

	t.Run("data too small", func(t *testing.T) {
		tooSmall := []byte{0x01, 0x02}

		audioData, err := downloader.basicAudioExtraction(tooSmall, "test.small")

		assert.Error(t, err)
		assert.Nil(t, audioData)
	})
}

func TestDetectAudioFormat(t *testing.T) {
	downloader := NewAudioDownloader(&http.Client{}, nil, nil)

	testCases := []struct {
		name     string
		data     []byte
		expected string
	}{
		{"AAC ADTS", []byte{0xFF, 0xF1, 0x50, 0x80}, "aac"},
		{"MP3", []byte{0xFF, 0xFB, 0x90, 0x00}, "mp3"},
		{"MP3 with ID3", []byte{'I', 'D', '3', 0x03, 0x00}, "mp3"},
		{"Transport Stream", []byte{0x47, 0x40, 0x00, 0x10}, "ts"},
		{"Unknown", []byte{0x00, 0x01, 0x02, 0x03}, "unknown"},
		{"Too small", []byte{0x01}, "unknown"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			format := downloader.detectAudioFormat(tc.data)
			assert.Equal(t, tc.expected, format)
		})
	}
}

func TestDownloadAudioSample(t *testing.T) {
	t.Run("successful multi-segment download", func(t *testing.T) {
		segmentCount := 0
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			segmentCount++
			w.Header().Set("Content-Type", "audio/aac")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte{0xFF, 0xF1, 0x50, 0x80}) // Minimal AAC data
		}))
		defer server.Close()

		// Create test playlist
		playlist := &M3U8Playlist{
			IsValid: true,
			Segments: []M3U8Segment{
				{URI: server.URL + "/segment1.aac", Duration: 10.0},
				{URI: server.URL + "/segment2.aac", Duration: 10.0},
				{URI: server.URL + "/segment3.aac", Duration: 10.0},
			},
		}

		client := &http.Client{Timeout: 5 * time.Second}
		config := DefaultDownloadConfig()
		config.MaxSegments = 2 // Limit to 2 segments
		hlsConfig := DefaultConfig()
		hlsConfig.AudioDecoder = &mockAudioDecoder{
			sampleRate: 44100,
			channels:   1,
			duration:   10 * time.Second,
		}

		downloader := NewAudioDownloader(client, config, hlsConfig)

		ctx := context.Background()
		audioData, err := downloader.DownloadAudioSample(ctx, playlist, 15*time.Second)

		require.NoError(t, err)
		require.NotNil(t, audioData)

		// Verify we got valid combined data
		assert.Equal(t, 2, segmentCount) // Should only download 2 segments
		assert.Greater(t, audioData.SampleRate, 0)
		assert.Greater(t, audioData.Channels, 0)
		assert.Greater(t, audioData.Duration, time.Duration(0))
		assert.NotEmpty(t, audioData.PCM)
	})

	t.Run("empty playlist", func(t *testing.T) {
		playlist := &M3U8Playlist{
			IsValid:  true,
			Segments: []M3U8Segment{},
		}

		downloader := NewAudioDownloader(&http.Client{}, nil, nil)

		ctx := context.Background()
		audioData, err := downloader.DownloadAudioSample(ctx, playlist, 30*time.Second)

		assert.Error(t, err)
		assert.Nil(t, audioData)
	})

	t.Run("nil playlist", func(t *testing.T) {
		downloader := NewAudioDownloader(&http.Client{}, nil, nil)

		ctx := context.Background()
		audioData, err := downloader.DownloadAudioSample(ctx, nil, 30*time.Second)

		assert.Error(t, err)
		assert.Nil(t, audioData)
	})
}

func TestAudioProcessing(t *testing.T) {
	config := DefaultDownloadConfig()
	config.NormalizePCM = true
	config.OutputChannels = 1
	config.OutputSampleRate = 22050

	downloader := NewAudioDownloader(&http.Client{}, config, nil)

	t.Run("normalization", func(t *testing.T) {
		// Create audio data with values > 1.0 (clipping)
		audioData := &common.AudioData{
			PCM:        []float64{2.0, -2.0, 1.5, -1.5, 0.5},
			SampleRate: 44100,
			Channels:   1,
			Duration:   1 * time.Second,
		}

		downloader.processAudioData(audioData)

		// Values should be normalized (within reasonable range)
		for _, sample := range audioData.PCM {
			assert.LessOrEqual(t, sample, 1.0)
			assert.GreaterOrEqual(t, sample, -1.0)
		}
	})

	t.Run("stereo to mono conversion", func(t *testing.T) {
		stereoData := &common.AudioData{
			PCM:        []float64{1.0, 0.0, 0.5, 0.5}, // L, R, L, R
			SampleRate: 44100,
			Channels:   2,
			Duration:   1 * time.Second,
		}

		monoData := downloader.convertToMono(stereoData)

		assert.Equal(t, 1, monoData.Channels)
		assert.Len(t, monoData.PCM, 2) // 2 mono samples from 4 stereo
		assert.Equal(t, stereoData.SampleRate, monoData.SampleRate)
		assert.Equal(t, stereoData.Duration, monoData.Duration)
	})
}

func TestCombineAudioSamples(t *testing.T) {
	downloader := NewAudioDownloader(&http.Client{}, nil, nil)

	t.Run("combine multiple samples", func(t *testing.T) {
		samples := []*common.AudioData{
			{
				PCM:        []float64{0.1, 0.2},
				SampleRate: 44100,
				Channels:   1,
				Duration:   1 * time.Second,
			},
			{
				PCM:        []float64{0.3, 0.4},
				SampleRate: 44100,
				Channels:   1,
				Duration:   1 * time.Second,
			},
		}

		combined, err := downloader.combineAudioSamples(samples)

		require.NoError(t, err)
		require.NotNil(t, combined)

		// Should have combined data
		assert.Len(t, combined.PCM, 4) // 2 + 2 samples
		assert.Equal(t, 44100, combined.SampleRate)
		assert.Equal(t, 1, combined.Channels)
		assert.Equal(t, 2*time.Second, combined.Duration)
	})

	t.Run("sample rate mismatch", func(t *testing.T) {
		samples := []*common.AudioData{
			{PCM: []float64{0.1}, SampleRate: 44100, Channels: 1},
			{PCM: []float64{0.2}, SampleRate: 48000, Channels: 1}, // Different rate
		}

		combined, err := downloader.combineAudioSamples(samples)

		assert.Error(t, err)
		assert.Nil(t, combined)
	})
}

func TestDownloadStats(t *testing.T) {
	client := &http.Client{Timeout: 5 * time.Second}
	downloader := NewAudioDownloader(client, nil, nil)

	// Simulate some download activity
	downloader.downloadStats.SegmentsDownloaded = 5
	downloader.downloadStats.BytesDownloaded = 50000
	downloader.downloadStats.DownloadTime = 2 * time.Second
	downloader.downloadStats.ErrorCount = 1

	stats := downloader.GetDownloadStats()

	assert.Equal(t, 5, stats.SegmentsDownloaded)
	assert.Equal(t, int64(50000), stats.BytesDownloaded)
	assert.Equal(t, 2*time.Second, stats.DownloadTime)
	assert.Equal(t, 1, stats.ErrorCount)
	assert.Greater(t, stats.AverageBitrate, 0.0) // Should calculate bitrate
}

func TestDownloaderUtilityMethods(t *testing.T) {
	downloader := NewAudioDownloader(&http.Client{}, nil, nil)

	t.Run("clear cache", func(t *testing.T) {
		downloader.segmentCache["test"] = []byte("data")
		assert.Len(t, downloader.segmentCache, 1)

		downloader.ClearCache()
		assert.Len(t, downloader.segmentCache, 0)
	})

	t.Run("update config", func(t *testing.T) {
		newConfig := &DownloadConfig{MaxSegments: 20}
		downloader.UpdateConfig(newConfig)
		assert.Equal(t, newConfig, downloader.config)
	})

	t.Run("close", func(t *testing.T) {
		err := downloader.Close()
		assert.NoError(t, err)
	})
}

func TestDownloaderWithRealStreams(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping real stream tests in short mode")
	}

	logger := logging.NewDefaultLogger().WithFields(logging.Fields{
		"test": "TestDownloaderWithRealStreams",
	})

	client := &http.Client{Timeout: 15 * time.Second}
	detector := NewDetector()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	t.Run("CDN stream download", func(t *testing.T) {
		// Get playlist first
		playlist, err := detector.DetectFromM3U8Content(ctx, TestCDNStreamURL, nil, nil)
		if err != nil {
			logger.Warn("CDN stream not accessible, skipping test", logging.Fields{
				"url":   TestCDNStreamURL,
				"error": err.Error(),
			})
			t.Skip("CDN stream not accessible")
		}

		if len(playlist.Segments) == 0 {
			t.Skip("No segments in CDN playlist")
		}

		// Resolve segment URLs if relative
		for i := range playlist.Segments {
			if !strings.HasPrefix(playlist.Segments[i].URI, "http") {
				baseURL := strings.TrimSuffix(TestCDNStreamURL, "/playlist.m3u8")
				playlist.Segments[i].URI = baseURL + "/" + playlist.Segments[i].URI
			}
		}

		config := DefaultDownloadConfig()
		config.MaxSegments = 1 // Just test one segment
		downloader := NewAudioDownloader(client, config, nil)

		logger.Info("Testing CDN stream download", logging.Fields{
			"playlist_url":  TestCDNStreamURL,
			"segment_count": len(playlist.Segments),
		})

		audioData, err := downloader.DownloadAudioSample(ctx, playlist, 10*time.Second)
		if err != nil {
			logger.Warn("CDN download failed (may be expected)", logging.Fields{"error": err.Error()})
		} else {
			require.NotNil(t, audioData)

			// Verify we got real data (not just checking exact values)
			assert.Greater(t, audioData.SampleRate, 0, "Should have valid sample rate")
			assert.Greater(t, audioData.Channels, 0, "Should have valid channel count")
			assert.Greater(t, audioData.Duration, time.Duration(0), "Should have valid duration")
			assert.NotEmpty(t, audioData.PCM, "Should have PCM data")
			assert.NotNil(t, audioData.Metadata, "Should have metadata")

			logger.Info("CDN stream download successful", logging.Fields{
				"sample_rate":      audioData.SampleRate,
				"channels":         audioData.Channels,
				"duration_seconds": audioData.Duration.Seconds(),
				"pcm_samples":      len(audioData.PCM),
				"codec":            audioData.Metadata.Codec,
			})
		}
	})

	t.Run("SRC stream download", func(t *testing.T) {
		// Get playlist first
		playlist, err := detector.DetectFromM3U8Content(ctx, TestSRCStreamURL, nil, nil)
		if err != nil {
			logger.Warn("SRC stream not accessible, skipping test", logging.Fields{
				"url":   TestSRCStreamURL,
				"error": err.Error(),
			})
			t.Skip("SRC stream not accessible")
		}

		logger.Info("SRC stream playlist info", logging.Fields{
			"is_master":     playlist.IsMaster,
			"is_valid":      playlist.IsValid,
			"segment_count": len(playlist.Segments),
			"variant_count": len(playlist.Variants),
		})

		if playlist.IsMaster && len(playlist.Variants) > 0 {
			// For master playlists, try to get a media playlist
			// Just verify we can detect it's a master playlist
			assert.True(t, playlist.IsMaster, "Should detect master playlist")
			assert.Greater(t, len(playlist.Variants), 0, "Master should have variants")

			logger.Info("SRC stream is master playlist", logging.Fields{
				"variant_count": len(playlist.Variants),
			})
		} else if len(playlist.Segments) > 0 {
			// Try to download from media playlist
			config := DefaultDownloadConfig()
			config.MaxSegments = 1 // Just test one segment
			downloader := NewAudioDownloader(client, config, nil)

			// Resolve segment URLs if needed
			for i := range playlist.Segments {
				if !strings.HasPrefix(playlist.Segments[i].URI, "http") {
					baseURL := strings.TrimSuffix(TestSRCStreamURL, "/master.m3u8")
					playlist.Segments[i].URI = baseURL + "/" + playlist.Segments[i].URI
				}
			}

			audioData, err := downloader.DownloadAudioSample(ctx, playlist, 10*time.Second)
			if err != nil {
				logger.Warn("SRC download failed (may be expected)", logging.Fields{"error": err.Error()})
			} else {
				require.NotNil(t, audioData)

				// Verify we got real data
				assert.Greater(t, audioData.SampleRate, 0)
				assert.Greater(t, audioData.Channels, 0)
				assert.Greater(t, audioData.Duration, time.Duration(0))
				assert.NotEmpty(t, audioData.PCM)

				logger.Info("SRC stream download successful", logging.Fields{
					"sample_rate":      audioData.SampleRate,
					"channels":         audioData.Channels,
					"duration_seconds": audioData.Duration.Seconds(),
					"pcm_samples":      len(audioData.PCM),
				})
			}
		}
	})
}

func BenchmarkBasicAudioExtraction(b *testing.B) {
	downloader := NewAudioDownloader(&http.Client{}, nil, nil)

	// Create mock AAC data
	aacData := make([]byte, 1024)
	for i := 0; i < len(aacData); i += 8 {
		aacData[i] = 0xFF
		aacData[i+1] = 0xF1
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := downloader.basicAudioExtraction(aacData, "test.aac")
		if err != nil {
			b.Fatal(err)
		}
	}
}
