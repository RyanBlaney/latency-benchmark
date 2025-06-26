package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/common"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream/hls"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

var (
	hlsAnalyzeSegments      bool
	hlsMaxSegments          int
	hlsFollowPlaylist       bool
	hlsShowPlaylist         bool
	hlsTimeout              time.Duration
	hlsVerbose              bool
	hlsValidateStream       bool
	hlsShowQualityAnalysis  bool
	hlsShowPlaylistAnalysis bool
	hlsShowHealthCheck      bool
	hlsCustomConfig         string
)

var titleCaser = cases.Title(language.English)

var hlsCmd = &cobra.Command{
	Use:   "hls [url]",
	Short: "HLS-specific testing and analysis",
	Long: `Perform comprehensive HLS testing including playlist validation,
segment analysis, M3U8 parsing verification, metadata extraction,
quality assessment, health checking, and performance profiling.

This command tests the complete HLS detection and parsing pipeline:
- URL pattern detection
- HTTP header analysis  
- M3U8 content parsing
- Metadata extraction
- Stream validation
- Quality assessment
- Health monitoring
- Performance profiling

Examples:
  # Test basic HLS detection and parsing
  cdn-benchmark hls https://playlist.fns.tunein.com/v3/news/bloomberg/aac_adts/96/media.m3u8

  # Full comprehensive testing
  cdn-benchmark hls --validate --quality-analysis --playlist-analysis --health-check --verbose https://stream.example.com/playlist.m3u8

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
	hlsCmd.Flags().BoolVar(&hlsValidateStream, "validate", false,
		"perform comprehensive stream validation")
	hlsCmd.Flags().BoolVar(&hlsShowQualityAnalysis, "quality-analysis", false,
		"show quality assessment and scoring")
	hlsCmd.Flags().BoolVar(&hlsShowPlaylistAnalysis, "playlist-analysis", false,
		"show comprehensive playlist analysis")
	hlsCmd.Flags().BoolVar(&hlsShowHealthCheck, "health-check", false,
		"show stream health assessment")
	hlsCmd.Flags().StringVar(&hlsCustomConfig, "config", "",
		"path to custom HLS configuration file")
}

// Simple performance timer for the command
type PerformanceTimer struct {
	startTimes map[string]time.Time
	durations  map[string]time.Duration
	totalStart time.Time
}

func NewPerformanceTimer() *PerformanceTimer {
	return &PerformanceTimer{
		startTimes: make(map[string]time.Time),
		durations:  make(map[string]time.Duration),
		totalStart: time.Now(),
	}
}

func (pt *PerformanceTimer) StartEvent(name string) {
	pt.startTimes[name] = time.Now()
}

func (pt *PerformanceTimer) EndEvent(name string) time.Duration {
	if startTime, exists := pt.startTimes[name]; exists {
		duration := time.Since(startTime)
		pt.durations[name] = duration
		delete(pt.startTimes, name)
		return duration
	}
	return 0
}

func (pt *PerformanceTimer) GetDuration(name string) time.Duration {
	return pt.durations[name]
}

func (pt *PerformanceTimer) GetTotalDuration() time.Duration {
	return time.Since(pt.totalStart)
}

func (pt *PerformanceTimer) GetAllDurations() map[string]time.Duration {
	result := make(map[string]time.Duration)

	maps.Copy(result, pt.durations)
	return result
}

func runHLSTest(cmd *cobra.Command, args []string) error {
	url := args[0]

	// Use local verbose flag or global one
	verbose := hlsVerbose || viper.GetBool("verbose")

	fmt.Printf("HLS Stream Testing: %s\n", url)
	fmt.Printf("═══════════════════════════════════════════════════════════════\n\n")

	ctx, cancel := context.WithTimeout(context.Background(), hlsTimeout)
	defer cancel()

	// Initialize performance timer
	timer := NewPerformanceTimer()
	timer.StartEvent("total_test")

	// Load custom configuration if provided
	var config *hls.Config
	if hlsCustomConfig != "" {
		// TODO: Load config from file
		config = hls.DefaultConfig()
		fmt.Printf("📁 Using custom configuration: %s\n\n", hlsCustomConfig)
	} else {
		config = hls.DefaultConfig()
	}

	// Test 1: URL Pattern Detection
	timer.StartEvent("url_detection")
	fmt.Printf("1️⃣  URL Pattern Detection\n")
	urlDetection := hls.DetectFromURL(url)
	if urlDetection == common.StreamTypeHLS {
		fmt.Printf("   ✅ URL pattern indicates HLS stream\n")
	} else {
		fmt.Printf("   ❌ URL pattern does not match HLS\n")
	}
	fmt.Printf("   Detection result: %s\n", urlDetection)
	timer.EndEvent("url_detection")
	fmt.Printf("\n")

	// Test 2: HTTP Header Detection
	timer.StartEvent("header_detection")
	fmt.Printf("2️⃣  HTTP Header Detection\n")
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
	fmt.Printf("   Detection result: %s\n", headerDetection)
	timer.EndEvent("header_detection")
	fmt.Printf("\n")

	// Test 3: M3U8 Content Parsing
	timer.StartEvent("m3u8_parsing")
	fmt.Printf("3️⃣  M3U8 Content Parsing\n")
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
	timer.EndEvent("m3u8_parsing")
	fmt.Printf("\n")

	// Test 4: Handler Integration Test
	timer.StartEvent("handler_integration")
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
	timer.EndEvent("handler_integration")
	fmt.Printf("\n")

	// Test 5: Stream Validation (if requested)
	var validationPassed bool
	if hlsValidateStream {
		timer.StartEvent("stream_validation")
		fmt.Printf("5️⃣  Stream Validation\n")
		validator := hls.NewValidatorWithConfig(config)

		// Validate URL
		if err := validator.ValidateURL(ctx, url); err != nil {
			fmt.Printf("   ❌ URL validation failed: %v\n", err)
		} else {
			fmt.Printf("   ✅ URL validation passed\n")
		}

		// Validate stream
		if err := validator.ValidateStream(ctx, hlsHandler); err != nil {
			fmt.Printf("   ❌ Stream validation failed: %v\n", err)
		} else {
			fmt.Printf("   ✅ Stream validation passed\n")
			validationPassed = true
		}

		timer.EndEvent("stream_validation")
		fmt.Printf("\n")
	}

	// Test 6: Quality Assessment (if requested)
	var qualityMetrics *hls.QualityMetrics
	if hlsShowQualityAnalysis && metadata != nil {
		timer.StartEvent("quality_assessment")
		fmt.Printf("6️⃣  Quality Assessment\n")
		qualityAnalyzer := hls.NewQualityAnalyzer()
		qualityMetrics = qualityAnalyzer.AnalyzeQuality(metadata)
		displayQualityMetrics(qualityMetrics, verbose)
		timer.EndEvent("quality_assessment")
		fmt.Printf("\n")
	}

	// Test 7: Comprehensive Analysis (if requested)
	var playlistAnalysis *hls.PlaylistAnalysis
	if hlsShowPlaylistAnalysis {
		timer.StartEvent("playlist_analysis")
		fmt.Printf("7️⃣  Comprehensive Analysis\n")
		analyzer := hls.NewPlaylistAnalyzer()
		playlistAnalysis = analyzer.AnalyzePlaylist(playlist)
		displayPlaylistAnalysis(playlistAnalysis, verbose)
		timer.EndEvent("playlist_analysis")
		fmt.Printf("\n")
	}

	// Test 8: Health Assessment (if requested)
	var healthStatus *hls.HealthStatus
	if hlsShowHealthCheck && metadata != nil {
		timer.StartEvent("health_assessment")
		fmt.Printf("8️⃣  Health Assessment\n")
		healthChecker := hls.NewStreamHealthChecker(config)
		responseTime := timer.GetDuration("m3u8_parsing") + timer.GetDuration("header_detection")
		healthStatus = healthChecker.CheckStreamHealth(playlist, metadata, responseTime)
		displayHealthStatus(healthStatus, verbose)
		timer.EndEvent("health_assessment")
		fmt.Printf("\n")
	}

	// Test 9: Segment Analysis (if requested)
	if hlsAnalyzeSegments && len(playlist.Segments) > 0 {
		timer.StartEvent("segment_analysis")
		fmt.Printf("9️⃣  Segment Analysis\n")
		analyzeSegments(playlist, hlsMaxSegments, verbose)
		timer.EndEvent("segment_analysis")
		fmt.Printf("\n")
	}

	// Test 10: Show Playlist Details (if requested)
	if hlsShowPlaylist {
		fmt.Printf("🔟 Playlist Structure\n")
		displayPlaylistDetails(playlist, verbose)
		fmt.Printf("\n")
	}

	// Test 11: Follow Live Updates (if requested)
	if hlsFollowPlaylist && playlist.IsLive {
		fmt.Printf("1️⃣1️⃣ Live Playlist Following\n")
		err = followLivePlaylist(ctx, hlsHandler, verbose)
		if err != nil {
			fmt.Printf("   ❌ Live following failed: %v\n", err)
		}
		fmt.Printf("\n")
	}

	// Performance Summary
	timer.EndEvent("total_test")
	if verbose {
		fmt.Printf("⚡ Performance Summary\n")
		displayPerformanceSummary(timer)
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

	if hlsValidateStream {
		fmt.Printf("Stream Validation: %s\n", getCheckmark(validationPassed))
	}

	if hlsShowQualityAnalysis && qualityMetrics != nil {
		fmt.Printf("Quality Score:     %.1f/100 (%s)\n", qualityMetrics.QualityScore, qualityMetrics.OverallQuality)
	}

	if hlsShowHealthCheck && healthStatus != nil {
		fmt.Printf("Health Score:      %.1f/100 (%s)\n", healthStatus.Score, getHealthStatus(healthStatus.IsHealthy))
	}

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

func displayQualityMetrics(metrics *hls.QualityMetrics, verbose bool) {
	fmt.Printf("   🎯 Quality Assessment:\n")
	fmt.Printf("      • Overall Quality: %s\n", metrics.OverallQuality)
	fmt.Printf("      • Quality Score: %.1f/100\n", metrics.QualityScore)
	fmt.Printf("      • Audio Quality: %s\n", metrics.AudioQuality)

	if metrics.VideoQuality != "" {
		fmt.Printf("      • Video Quality: %s\n", metrics.VideoQuality)
	}

	if metrics.Resolution != "" {
		fmt.Printf("      • Resolution: %s\n", metrics.Resolution)
	}

	if metrics.FrameRate > 0 {
		fmt.Printf("      • Frame Rate: %.1f fps\n", metrics.FrameRate)
	}

	if verbose {
		fmt.Printf("      • Detailed Metrics:\n")
		fmt.Printf("        - Bitrate: %d kbps\n", metrics.Bitrate)
		fmt.Printf("        - Sample Rate: %d Hz\n", metrics.SampleRate)
		fmt.Printf("        - Channels: %d\n", metrics.Channels)
	}
}

func displayPlaylistAnalysis(analysis *hls.PlaylistAnalysis, verbose bool) {
	fmt.Printf("   📊 Playlist Analysis:\n")
	fmt.Printf("      • Type: %s\n", func() string {
		if analysis.IsMaster {
			return "Master Playlist"
		}
		return "Media Playlist"
	}())

	fmt.Printf("      • Live Stream: %t\n", analysis.IsLive)
	fmt.Printf("      • Version: %d\n", analysis.Version)
	fmt.Printf("      • Total Segments: %d\n", analysis.TotalSegments)
	fmt.Printf("      • Total Variants: %d\n", analysis.TotalVariants)

	if analysis.EstimatedDuration > 0 {
		fmt.Printf("      • Estimated Duration: %v\n", analysis.EstimatedDuration)
	}

	// Segment analysis
	if analysis.SegmentAnalysis != nil {
		sa := analysis.SegmentAnalysis
		fmt.Printf("      • Segment Analysis:\n")
		fmt.Printf("        - Average Duration: %.2fs\n", sa.AverageDuration)
		fmt.Printf("        - Duration Range: %.2fs - %.2fs\n", sa.MinDuration, sa.MaxDuration)
		fmt.Printf("        - Has Ad Breaks: %t\n", sa.HasAdBreaks)
		fmt.Printf("        - Has Discontinuities: %t\n", sa.HasDiscontinuities)
		if sa.CategoryCount > 0 {
			fmt.Printf("        - Content Categories: %d\n", sa.CategoryCount)
		}
	}

	// Variant analysis
	if analysis.VariantAnalysis != nil {
		va := analysis.VariantAnalysis
		fmt.Printf("      • Variant Analysis:\n")
		fmt.Printf("        - Bandwidth Range: %d - %d kbps\n", va.MinBandwidth/1000, va.MaxBandwidth/1000)
		fmt.Printf("        - Average Bandwidth: %d kbps\n", va.AvgBandwidth/1000)

		if len(va.CodecDistribution) > 0 && verbose {
			fmt.Printf("        - Codec Distribution:\n")
			for codec, count := range va.CodecDistribution {
				fmt.Printf("          • %s: %d variants\n", codec, count)
			}
		}

		if len(va.ResolutionDistribution) > 0 && verbose {
			fmt.Printf("        - Resolution Distribution:\n")
			for resolution, count := range va.ResolutionDistribution {
				fmt.Printf("          • %s: %d variants\n", resolution, count)
			}
		}
	}

	// Content patterns
	if analysis.ContentPatterns != nil {
		cp := analysis.ContentPatterns
		if cp.PrimaryGenre != "" {
			fmt.Printf("      • Primary Genre: %s\n", cp.PrimaryGenre)
		}

		if len(cp.Categories) > 0 && verbose {
			fmt.Printf("      • All Categories: %v\n", cp.Categories)
		}

		if len(cp.AdPatterns) > 0 {
			fmt.Printf("      • Ad Patterns: %v\n", cp.AdPatterns)
		}

		if len(cp.TimePatterns) > 0 && verbose {
			fmt.Printf("      • Time Patterns: %v\n", cp.TimePatterns)
		}
	}
}

func displayHealthStatus(status *hls.HealthStatus, verbose bool) {
	fmt.Printf("   🏥 Health Assessment:\n")
	fmt.Printf("      • Overall Health: %s\n", getHealthStatus(status.IsHealthy))
	fmt.Printf("      • Health Score: %.1f/100\n", status.Score)
	fmt.Printf("      • Response Time: %v\n", status.ResponseTime)
	fmt.Printf("      • Playlist Size: %d elements\n", status.PlaylistSize)

	if status.EstimatedBandwidth > 0 {
		fmt.Printf("      • Estimated Bandwidth: %d kbps\n", status.EstimatedBandwidth/1000)
	}

	if len(status.Issues) > 0 {
		fmt.Printf("      • Issues Found:\n")
		for _, issue := range status.Issues {
			var severity string
			switch issue.Severity {
			case "warning":
				severity = "⚠️"
			case "critical":
				severity = "❌"
			default:
				severity = "ℹ️"
			}
			fmt.Printf("        %s %s: %s\n", severity, issue.Category, issue.Description)
			if verbose && issue.Impact != "" {
				fmt.Printf("          Impact: %s\n", issue.Impact)
			}
		}
	}

	if len(status.Recommendations) > 0 && verbose {
		fmt.Printf("      • Recommendations:\n")
		for _, rec := range status.Recommendations {
			fmt.Printf("        • %s\n", rec)
		}
	}
}

func displayPerformanceSummary(timer *PerformanceTimer) {
	durations := timer.GetAllDurations()
	total := timer.GetTotalDuration()

	fmt.Printf("   ⏱️ Performance Timing:\n")
	fmt.Printf("      • Total Duration: %v\n", total)

	for event, duration := range durations {
		if event != "total_test" {
			percentage := float64(duration) / float64(total) * 100
			fmt.Printf("      • %s: %v (%.1f%%)\n",
				strings.ReplaceAll(titleCaser.String(strings.ReplaceAll(event, "_", " ")), " ", " "),
				duration, percentage)
		}
	}
}

func getHealthStatus(isHealthy bool) string {
	if isHealthy {
		return "Healthy"
	}
	return "Unhealthy"
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

	if metadata.Format != "" {
		fmt.Printf("      • Format: %s\n", metadata.Format)
	}

	if metadata.Bitrate > 0 {
		fmt.Printf("      • Bitrate: %d kbps\n", metadata.Bitrate)
	}

	if metadata.SampleRate > 0 {
		fmt.Printf("      • Sample Rate: %d Hz\n", metadata.SampleRate)
	}

	if metadata.Channels > 0 {
		fmt.Printf("      • Channels: %d\n", metadata.Channels)
	}

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
		"content-type", "server", "x-tunein-playlist-available-duration",
		"x-cache", "via", "is_live", "is_master", "content_categories",
		"has_ad_breaks", "discontinuity_sequence", "tunein_available_duration",
		"start_time_offset",
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
			// Refresh playlist
			if err := handler.RefreshPlaylist(ctx); err != nil {
				fmt.Printf("   ❌ Failed to refresh playlist: %v\n", err)
				continue
			}

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
