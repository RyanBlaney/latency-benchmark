//go:build noaudio
// +build noaudio

package hls

import (
	"context"
	"fmt"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"time"
)

// Stub types for when audio processing is disabled
type AudioDownloader struct{}

type DownloadStats struct {
	SegmentsDownloaded int   `json:"segments_downloaded"`
	BytesDownloaded    int64 `json:"bytes_downloaded"`
	DownloadTime       time.Duration
	DecodeTime         time.Duration
	ErrorCount         int
	AverageBitrate     float64
}

type DownloadConfig struct{}

func NewAudioDownloader(client interface{}, config interface{}) *AudioDownloader {
	return &AudioDownloader{}
}

func (ad *AudioDownloader) DownloadAudioSample(ctx context.Context, playlist *M3U8Playlist, targetDuration time.Duration) (*common.AudioData, error) {
	return nil, fmt.Errorf("audio processing disabled at compile time")
}

func (ad *AudioDownloader) GetDownloadStats() *DownloadStats {
	return &DownloadStats{
		SegmentsDownloaded: 0,
		BytesDownloaded:    0,
		DownloadTime:       0,
		DecodeTime:         0,
		ErrorCount:         0,
		AverageBitrate:     0,
	}
}

func (ad *AudioDownloader) Close() error {
	return nil
}

func DefaultDownloadConfig() *DownloadConfig {
	return &DownloadConfig{}
}

