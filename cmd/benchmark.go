package cmd

import (
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"

	"github.com/tunein/cdn-benchmark-cli/configs"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
)

var (
	// Benchmark command flags
	benchmarkProfile    string
	benchmarkDuration   time.Duration
	benchmarkRegions    []string
	benchmarkStreams    []string
	benchmarkReference  string
	benchmarkConcurrent bool
	benchmarkOutput     string
	benchmarkTags       []string
)

// benchmarkCmd represents the benchmark command
var benchmarkCmd = &cobra.Command{
	Use:   "benchmark [flags] [stream-urls...]",
	Short: "Run CDN performance benchmarks",
	Long: `Run comprehensive CDN performance benchmarks against specified streams.

This command performs multi-dimensional analysis including:
- Audio quality comparison using spectral fingerprinting
- Latency measurement (network, processing, end-to-end)
- Stream reliability and connection statistics
- Regional performance comparison

Examples:
  # Run benchmark with default profile
  cdn-benchmark benchmark https://stream.example.com/hls/playlist.m3u8

  # Run with specific profile and regions
  cdn-benchmark benchmark --profile production --regions us-west,eu-west https://stream.example.com/

  # Compare CDN stream against reference
  cdn-benchmark benchmark --reference https://origin.example.com/stream https://cdn.example.com/stream

  # Run concurrent tests across multiple streams
  cdn-benchmark benchmark --concurrent --duration 5m stream1.m3u8 stream2.m3u8

  # Use custom output format and tags
  cdn-benchmark benchmark --output json --tags env=staging,test=nightly https://stream.example.com/`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 && benchmarkProfile == "" {
			return fmt.Errorf("requires at least one stream URL or --profile flag")
		}
		return nil
	},
	RunE: runBenchmark,
}
