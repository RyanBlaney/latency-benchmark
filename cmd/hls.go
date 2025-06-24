package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/hls"
)

var (
	hlsAnalyzeSegments bool
	hlsMaxSegments     int
	hlsFollowPlaylist  bool
	hlsShowPlaylist    bool
	hlsTimeout         time.Duration
	hlsVerbose         bool
)

var hlsCmd = &cobra.Command{
	Use:   "hls [url]",
	Short: "HLS-specific testing and analysis",
	Long: `Perform HLS-specific testing including playlist validation,
segment analysis, M3U8 parsing verification, and metadata extraction.

This command tests the complete HLS detection and parsing pipeline:
- URL pattern detection
- HTTP header analysis  
- M3U8 content parsing
- Metadata extraction
- Segment enumeration

Examples:
  # Test basic HLS detection and parsing
  cdn-benchmark hls https://playlist.fns.tunein.com/v3/news/bloomberg/aac_adts/96/media.m3u8

  # Show detailed playlist information
  cdn-benchmark hls --show-playlist --verbose https://stream.example.com/playlist.m3u8

  # Analyze specific number of segments
  cdn-benchmark hls --analyze-segments --max-segments 5 https://stream.example.com/playlist.m3u8

  # Follow live playlist updates
  cdn-benchmark hls --follow --timeout 30s https://live.example.com/playlist.m3u8`,
	Args: cobra.ExactArgs(1),
	RunE: runHLSTest,
}

func init() {
	rootCmd.AddCommand(hlsCmd)

	hlsCmd.Flags().BoolVar(&hlsAnalyzeSegments, "analyze-segments", false,
		"analyze individual HLS segments")
	hlsCmd.Flags().IntVar(&hlsMaxSegments, "max-segments", 10,
		"maximum number of segments to analyze")
	hlsCmd.Flags().BoolVar(&hlsFollowPlaylist, "follow", false,
		"follow live playlist updates")
	hlsCmd.Flags().BoolVar(&hlsShowPlaylist, "show-playlist", false,
		"show detailed playlist structure")
	hlsCmd.Flags().DurationVar(&hlsTimeout, "timeout", 30*time.Second,
		"operation timeout")
	hlsCmd.Flags().BoolVarP(&hlsVerbose, "verbose", "v", false,
		"verbose output (overrides global verbose)")
}

func runHLSTest(cmd *cobra.Command, args []string) error {
	url := args[0]

	// Use local verbose flag or global one
	verbose := hlsVerbose || viper.GetBool("verbose")

	fmt.Printf("HLS Stream Testing: %s\n", url)
	fmt.Printf("═══════════════════════════════════════════════════════════════\n\n")

	ctx, cancel := context.WithTimeout(context.Background(), hlsTimeout)
	defer cancel()

	// Test 1: URL Pattern Detection
	fmt.Printf("URL Pattern Detection\n")
	urlDetection := hls.DetectFromURL(url)
	if urlDetection == common.StreamTypeHLS {
		fmt.Printf("   ✅ URL pattern indicates HLS stream\n")
	} else {
		fmt.Printf("   ❌ URL pattern does not match HLS\n")
	}
	fmt.Printf("   Detection result: %s\n\n", urlDetection)

	// Test 2: HTTP Header Detection
	fmt.Printf("HTTP Header Detection\n")
	factory := stream.NewFactory()
	handler, err := factory.CreateHandler(common.StreamTypeHLS)
	if err != nil {
		return fmt.Errorf("failed to create HLS handler: %w", err)
	}
	defer handler.Close()

	hlsHandler, ok := handler.(*hls.Handler)
	if !ok {
		return fmt.Errorf("handler is not HLS type")
	}

	headerDetection := hls.DetectFromHeaders(ctx, hlsHandler.GetClient(), url)
	if headerDetection == common.StreamTypeHLS {
		fmt.Printf("   ✅ HTTP headers indicate HLS stream\n")
	} else {
		fmt.Printf("   ❌ HTTP headers do not indicate HLS\n")
	}
	fmt.Printf("   Detection result: %s\n\n", headerDetection)

	// Test 3: M3U8 Content Parsing
	fmt.Printf("M3U8 Content Parsing\n")
	playlist, err := hls.DetectFromM3U8Content(ctx, hlsHandler.GetClient(), url)
	if err != nil {
		fmt.Printf("   ❌ M3U8 parsing failed: %v\n\n", err)
		return fmt.Errorf("M3U8 parsing failed: %w", err)
	}

	if playlist.IsValid {
		fmt.Printf("   ✅ Valid M3U8 playlist detected\n")
		fmt.Printf("   📊 Playlist Stats:\n")
		fmt.Printf("      • Version: %d\n", playlist.Version)
		fmt.Printf("      • Type: %s\n", getPlaylistType(playlist))
		fmt.Printf("      • Target Duration: %d seconds\n", playlist.TargetDuration)
		fmt.Printf("      • Media Sequence: %d\n", playlist.MediaSequence)
		fmt.Printf("      • Segments: %d\n", len(playlist.Segments))
		if len(playlist.Variants) > 0 {
			fmt.Printf("      • Variants: %d\n", len(playlist.Variants))
		}
	} else {
		fmt.Printf("   ❌ Invalid M3U8 playlist\n")
	}
	fmt.Printf("\n")

	// Test 4: Handler Integration Test
	fmt.Printf("4️⃣  Handler Integration Test\n")

	// Test CanHandle method
	canHandle := hlsHandler.CanHandle(ctx, url)
	if canHandle {
		fmt.Printf("   ✅ Handler can process this stream\n")
	} else {
		fmt.Printf("   ❌ Handler cannot process this stream\n")
		return fmt.Errorf("handler rejected stream")
	}

	// Test Connect method
	err = hlsHandler.Connect(ctx, url)
	if err != nil {
		fmt.Printf("   ❌ Connection failed: %v\n", err)
		return fmt.Errorf("connection failed: %w", err)
	}
	fmt.Printf("   ✅ Successfully connected to stream\n")

	// Test GetMetadata method
	metadata, err := hlsHandler.GetMetadata()
	if err != nil {
		fmt.Printf("   ❌ Metadata extraction failed: %v\n", err)
	} else {
		fmt.Printf("   ✅ Metadata extracted successfully\n")
		displayMetadata(metadata, verbose)
	}
	fmt.Printf("\n")

	// Test 5: Segment Analysis (if requested)
	if hlsAnalyzeSegments && len(playlist.Segments) > 0 {
		fmt.Printf("5️⃣  Segment Analysis\n")
		analyzeSegments(playlist, hlsMaxSegments, verbose)
		fmt.Printf("\n")
	}

	// Test 6: Show Playlist Details (if requested)
	if hlsShowPlaylist {
		fmt.Printf("6️⃣  Playlist Structure\n")
		displayPlaylistDetails(playlist, verbose)
		fmt.Printf("\n")
	}

	// Test 7: Follow Live Updates (if requested)
	if hlsFollowPlaylist && playlist.IsLive {
		fmt.Printf("7️⃣  Live Playlist Following\n")
		err = followLivePlaylist(ctx, hlsHandler, verbose)
		if err != nil {
			fmt.Printf("   ❌ Live following failed: %v\n", err)
		}
		fmt.Printf("\n")
	}

	// Test Summary
	fmt.Printf("📋 Test Summary\n")
	fmt.Printf("═══════════════════════════════════════════════════════════════\n")
	fmt.Printf("URL Detection:     %s\n", getCheckmark(urlDetection == common.StreamTypeHLS))
	fmt.Printf("Header Detection:  %s\n", getCheckmark(headerDetection == common.StreamTypeHLS))
	fmt.Printf("M3U8 Parsing:      %s\n", getCheckmark(playlist.IsValid))
	fmt.Printf("Handler Integration: %s\n", getCheckmark(canHandle && err == nil))
	fmt.Printf("Metadata Extraction: %s\n", getCheckmark(metadata != nil))

	if playlist.IsValid {
		fmt.Printf("\n🎯 Stream Classification:\n")
		fmt.Printf("   Type: %s\n", getPlaylistType(playlist))
		if metadata != nil {
			fmt.Printf("   Codec: %s\n", metadata.Codec)
			fmt.Printf("   Bitrate: %d kbps\n", metadata.Bitrate)
			if metadata.Genre != "" {
				fmt.Printf("   Genre: %s\n", metadata.Genre)
			}
			if metadata.Station != "" {
				fmt.Printf("   Station: %s\n", metadata.Station)
			}
		}
	}

	return nil
}

func getPlaylistType(playlist *hls.M3U8Playlist) string {
	if playlist.IsMaster {
		return "Master Playlist"
	} else if playlist.IsLive {
		return "Live Media Playlist"
	} else {
		return "VOD Media Playlist"
	}
}

func getCheckmark(success bool) string {
	if success {
		return "✅ PASS"
	}
	return "❌ FAIL"
}

func displayMetadata(metadata *common.StreamMetadata, verbose bool) {
	fmt.Printf("   📄 Stream Metadata:\n")
	fmt.Printf("      • Type: %s\n", metadata.Type)
	fmt.Printf("      • Codec: %s\n", metadata.Codec)
	fmt.Printf("      • Bitrate: %d kbps\n", metadata.Bitrate)
	fmt.Printf("      • Sample Rate: %d Hz\n", metadata.SampleRate)
	fmt.Printf("      • Channels: %d\n", metadata.Channels)

	if metadata.Station != "" {
		fmt.Printf("      • Station: %s\n", metadata.Station)
	}
	if metadata.Genre != "" {
		fmt.Printf("      • Genre: %s\n", metadata.Genre)
	}

	if len(metadata.Headers) > 0 {
		fmt.Printf("      • Additional Properties:\n")
		for key, value := range metadata.Headers {
			if verbose || shouldShowHeader(key) {
				fmt.Printf("        - %s: %s\n", key, value)
			}
		}
	}
}

func shouldShowHeader(key string) bool {
	importantHeaders := []string{
		"content_categories", "has_ad_breaks", "discontinuity_sequence",
		"tunein_available_duration", "start_time_offset",
	}

	return slices.Contains(importantHeaders, key)
}

func analyzeSegments(playlist *hls.M3U8Playlist, maxSegments int, verbose bool) {
	segments := playlist.Segments
	if len(segments) > maxSegments {
		segments = segments[:maxSegments]
		fmt.Printf("   Analyzing first %d of %d segments:\n\n", maxSegments, len(playlist.Segments))
	} else {
		fmt.Printf("   Analyzing all %d segments:\n\n", len(segments))
	}

	totalDuration := 0.0
	categories := make(map[string]int)
	adBreaks := 0

	for i, segment := range segments {
		fmt.Printf("   Segment %d:\n", i+1)
		fmt.Printf("      Duration: %.3fs\n", segment.Duration)

		if verbose {
			fmt.Printf("      URL: %s\n", truncateURL(segment.URI, 60))
		}

		if segment.Title != "" {
			fmt.Printf("      Metadata: %s\n", segment.Title)

			// Extract categories
			if strings.Contains(segment.Title, "CATEGORY:") {
				if start := strings.Index(segment.Title, "CATEGORY:"); start != -1 {
					start += len("CATEGORY:")
					if end := strings.Index(segment.Title[start:], ","); end != -1 {
						category := segment.Title[start : start+end]
						categories[category]++
					} else {
						category := segment.Title[start:]
						categories[category]++
					}
				}
			}

			// Count ad breaks
			if strings.Contains(segment.Title, "AD_BREAK") {
				adBreaks++
			}
		}

		totalDuration += segment.Duration
		fmt.Printf("\n")
	}

	fmt.Printf("   📊 Segment Summary:\n")
	fmt.Printf("      • Total Duration: %.1fs\n", totalDuration)
	fmt.Printf("      • Average Duration: %.3fs\n", totalDuration/float64(len(segments)))

	if len(categories) > 0 {
		fmt.Printf("      • Content Categories:\n")
		for category, count := range categories {
			fmt.Printf("        - %s: %d segments\n", category, count)
		}
	}

	if adBreaks > 0 {
		fmt.Printf("      • Ad Breaks: %d\n", adBreaks)
	}
}

func displayPlaylistDetails(playlist *hls.M3U8Playlist, verbose bool) {
	// Output as JSON for detailed inspection
	if verbose {
		fmt.Printf("   Raw playlist structure (JSON):\n")
		data, err := json.MarshalIndent(playlist, "   ", "  ")
		if err != nil {
			fmt.Printf("   Error marshaling playlist: %v\n", err)
			return
		}
		fmt.Printf("   %s\n", string(data))
	} else {
		fmt.Printf("   📋 Playlist Overview:\n")
		fmt.Printf("      • Valid: %t\n", playlist.IsValid)
		fmt.Printf("      • Master Playlist: %t\n", playlist.IsMaster)
		fmt.Printf("      • Live Stream: %t\n", playlist.IsLive)
		fmt.Printf("      • Version: %d\n", playlist.Version)
		fmt.Printf("      • Target Duration: %d\n", playlist.TargetDuration)
		fmt.Printf("      • Media Sequence: %d\n", playlist.MediaSequence)
		fmt.Printf("      • Segments: %d\n", len(playlist.Segments))
		fmt.Printf("      • Variants: %d\n", len(playlist.Variants))

		if len(playlist.Headers) > 0 {
			fmt.Printf("      • Custom Headers: %d\n", len(playlist.Headers))
		}
	}
}

func followLivePlaylist(ctx context.Context, handler *hls.Handler, verbose bool) error {
	fmt.Printf("   Following live playlist updates for %v...\n", hlsTimeout)

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	lastSequence := -1

	for {
		select {
		case <-ctx.Done():
			fmt.Printf("   ⏰ Timeout reached\n")
			return nil
		case <-ticker.C:
			// Get current playlist
			playlist := handler.GetPlaylist()
			if playlist == nil {
				continue
			}

			if playlist.MediaSequence != lastSequence {
				fmt.Printf("   📱 Playlist updated - Sequence: %d, Segments: %d\n",
					playlist.MediaSequence, len(playlist.Segments))
				lastSequence = playlist.MediaSequence

				if verbose && len(playlist.Segments) > 0 {
					latest := playlist.Segments[len(playlist.Segments)-1]
					fmt.Printf("      Latest segment: %.3fs - %s\n",
						latest.Duration, truncateURL(latest.URI, 50))
				}
			}
		}
	}
}

func truncateURL(url string, maxLen int) string {
	if len(url) <= maxLen {
		return url
	}
	return url[:maxLen-3] + "..."
}
