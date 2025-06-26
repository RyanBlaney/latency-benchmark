// Add this to your cmd/ directory as config_test.go

package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"github.com/tunein/cdn-benchmark-cli/configs"
)

// configTestCmd represents the config test command
var configTestCmd = &cobra.Command{
	Use:   "config-test",
	Short: "Test and display all configuration values",
	Long: `Test configuration loading and display all values to verify proper parsing.

This command loads the configuration and displays all values in a structured format
to help verify that your YAML configuration is being parsed correctly.

Examples:
  # Test with default config file
  cdn-benchmark config-test

  # Test with specific config file
  cdn-benchmark --config /path/to/config.yaml config-test`,
	RunE: runConfigTest,
}

func init() {
	rootCmd.AddCommand(configTestCmd)
}

func runConfigTest(cmd *cobra.Command, args []string) error {
	fmt.Println("CDN BENCHMARK CONFIGURATION TEST")
	fmt.Println(strings.Repeat("=", 80))

	// Load configuration
	config, err := configs.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load configuration: %w", err)
	}

	printSection("APPLICATION SETTINGS")
	printKeyValue("Verbose", fmt.Sprintf("%t", config.Verbose))
	printKeyValue("Log Level", config.LogLevel)
	printKeyValue("Output Format", config.OutputFormat)
	printKeyValue("Config Directory", config.ConfigDir)
	printKeyValue("Data Directory", config.DataDir)

	printSection("TEST CONFIGURATION")
	printKeyValue("Timeout", config.Test.Timeout.String())
	printKeyValue("Retry Attempts", fmt.Sprintf("%d", config.Test.RetryAttempts))
	printKeyValue("Retry Delay", config.Test.RetryDelay.String())
	printKeyValue("Concurrent", fmt.Sprintf("%t", config.Test.Concurrent))
	printKeyValue("Max Concurrency", fmt.Sprintf("%d", config.Test.MaxConcurrency))

	printSection("GLOBAL STREAM CONFIGURATION")
	printKeyValue("Connection Timeout", config.Stream.ConnectionTimeout.String())
	printKeyValue("Read Timeout", config.Stream.ReadTimeout.String())
	printKeyValue("Buffer Size", fmt.Sprintf("%d bytes", config.Stream.BufferSize))
	printKeyValue("Max Redirects", fmt.Sprintf("%d", config.Stream.MaxRedirects))
	printKeyValue("User Agent", config.Stream.UserAgent)
	if len(config.Stream.Headers) > 0 {
		printSubsection(fmt.Sprintf("Headers (%d)", len(config.Stream.Headers)))
		for key, value := range config.Stream.Headers {
			printKeyValue("  "+key, value)
		}
	}

	printSection("AUDIO CONFIGURATION")
	printKeyValue("Sample Rate", fmt.Sprintf("%d Hz", config.Audio.SampleRate))
	printKeyValue("Channels", fmt.Sprintf("%d", config.Audio.Channels))
	printKeyValue("Buffer Duration", config.Audio.BufferDuration.String())
	printKeyValue("Window Size", fmt.Sprintf("%d", config.Audio.WindowSize))
	printKeyValue("Overlap", fmt.Sprintf("%.2f", config.Audio.Overlap))
	printKeyValue("Window Function", config.Audio.WindowFunction)
	printKeyValue("FFT Size", fmt.Sprintf("%d", config.Audio.FFTSize))
	printKeyValue("Mel Bins", fmt.Sprintf("%d", config.Audio.MelBins))

	printSection("QUALITY CONFIGURATION")
	printKeyValue("Min Similarity", fmt.Sprintf("%.3f", config.Quality.MinSimilarity))
	printKeyValue("Max Latency", config.Quality.MaxLatency.String())
	printKeyValue("Min Bitrate", fmt.Sprintf("%d kbps", config.Quality.MinBitrate))
	printKeyValue("Max Dropouts", fmt.Sprintf("%d", config.Quality.MaxDropouts))
	printKeyValue("Buffer Health", fmt.Sprintf("%.2f", config.Quality.BufferHealth))

	printSection("OUTPUT CONFIGURATION")
	printKeyValue("Precision", fmt.Sprintf("%d", config.Output.Precision))
	printKeyValue("Include Metadata", fmt.Sprintf("%t", config.Output.IncludeMetadata))
	printKeyValue("Timestamps", fmt.Sprintf("%t", config.Output.Timestamps))
	printKeyValue("Colors", fmt.Sprintf("%t", config.Output.Colors))
	printKeyValue("Pager", fmt.Sprintf("%t", config.Output.Pager))

	printSection("HLS CONFIGURATION")

	printSubsection("Parser")
	printKeyValue("  Strict Mode", fmt.Sprintf("%t", config.HLS.Parser.StrictMode))
	printKeyValue("  Max Segment Analysis", fmt.Sprintf("%d", config.HLS.Parser.MaxSegmentAnalysis))
	printKeyValue("  Ignore Unknown Tags", fmt.Sprintf("%t", config.HLS.Parser.IgnoreUnknownTags))
	printKeyValue("  Validate URIs", fmt.Sprintf("%t", config.HLS.Parser.ValidateURIs))
	if len(config.HLS.Parser.CustomTagHandlers) > 0 {
		printKeyValue("  Custom Tag Handlers", fmt.Sprintf("(%d)", len(config.HLS.Parser.CustomTagHandlers)))
		for tag, handler := range config.HLS.Parser.CustomTagHandlers {
			printKeyValue("    "+tag, handler)
		}
	}

	printSubsection("Detection")
	printKeyValue("  Timeout Seconds", fmt.Sprintf("%d", config.HLS.Detection.TimeoutSeconds))
	printKeyValue("  URL Patterns", fmt.Sprintf("(%d) %v", len(config.HLS.Detection.URLPatterns), config.HLS.Detection.URLPatterns))
	printKeyValue("  Content Types", fmt.Sprintf("(%d) %v", len(config.HLS.Detection.ContentTypes), config.HLS.Detection.ContentTypes))
	printKeyValue("  Required Headers", fmt.Sprintf("(%d) %v", len(config.HLS.Detection.RequiredHeaders), config.HLS.Detection.RequiredHeaders))

	printSubsection("HTTP")
	printKeyValue("  User Agent", config.HLS.HTTP.UserAgent)
	printKeyValue("  Accept Header", config.HLS.HTTP.AcceptHeader)
	printKeyValue("  Connection Timeout", config.HLS.HTTP.ConnectionTimeout.String())
	printKeyValue("  Read Timeout", config.HLS.HTTP.ReadTimeout.String())
	printKeyValue("  Max Redirects", fmt.Sprintf("%d", config.HLS.HTTP.MaxRedirects))
	printKeyValue("  Buffer Size", fmt.Sprintf("%d bytes", config.HLS.HTTP.BufferSize))
	if len(config.HLS.HTTP.CustomHeaders) > 0 {
		printKeyValue("  Custom Headers", fmt.Sprintf("(%d)", len(config.HLS.HTTP.CustomHeaders)))
		for key, value := range config.HLS.HTTP.CustomHeaders {
			printKeyValue("    "+key, value)
		}
	}

	printSubsection("Audio")
	printKeyValue("  Sample Duration", config.HLS.Audio.SampleDuration.String())
	printKeyValue("  Buffer Duration", config.HLS.Audio.BufferDuration.String())
	printKeyValue("  Max Segments", fmt.Sprintf("%d", config.HLS.Audio.MaxSegments))
	printKeyValue("  Follow Live", fmt.Sprintf("%t", config.HLS.Audio.FollowLive))
	printKeyValue("  Analyze Segments", fmt.Sprintf("%t", config.HLS.Audio.AnalyzeSegments))

	printSubsection("Metadata Extractor")
	printKeyValue("  Enable URL Patterns", fmt.Sprintf("%t", config.HLS.MetadataExtractor.EnableURLPatterns))
	printKeyValue("  Enable Header Mappings", fmt.Sprintf("%t", config.HLS.MetadataExtractor.EnableHeaderMappings))
	printKeyValue("  Enable Segment Analysis", fmt.Sprintf("%t", config.HLS.MetadataExtractor.EnableSegmentAnalysis))
	if len(config.HLS.MetadataExtractor.DefaultValues) > 0 {
		printKeyValue("  Default Values", fmt.Sprintf("(%d)", len(config.HLS.MetadataExtractor.DefaultValues)))
		for key, value := range config.HLS.MetadataExtractor.DefaultValues {
			printKeyValue("    "+key, fmt.Sprintf("%v", value))
		}
	}

	printSection("ICECAST CONFIGURATION")

	printSubsection("Detection")
	printKeyValue("  Timeout Seconds", fmt.Sprintf("%d", config.ICEcast.Detection.TimeoutSeconds))
	printKeyValue("  URL Patterns", fmt.Sprintf("(%d) %v", len(config.ICEcast.Detection.URLPatterns), config.ICEcast.Detection.URLPatterns))
	printKeyValue("  Content Types", fmt.Sprintf("(%d) %v", len(config.ICEcast.Detection.ContentTypes), config.ICEcast.Detection.ContentTypes))
	printKeyValue("  Required Headers", fmt.Sprintf("(%d) %v", len(config.ICEcast.Detection.RequiredHeaders), config.ICEcast.Detection.RequiredHeaders))
	printKeyValue("  Common Ports", fmt.Sprintf("(%d) %v", len(config.ICEcast.Detection.CommonPorts), config.ICEcast.Detection.CommonPorts))

	printSubsection("HTTP")
	printKeyValue("  User Agent", config.ICEcast.HTTP.UserAgent)
	printKeyValue("  Accept Header", config.ICEcast.HTTP.AcceptHeader)
	printKeyValue("  Connection Timeout", config.ICEcast.HTTP.ConnectionTimeout.String())
	printKeyValue("  Read Timeout", config.ICEcast.HTTP.ReadTimeout.String())
	printKeyValue("  Max Redirects", fmt.Sprintf("%d", config.ICEcast.HTTP.MaxRedirects))
	printKeyValue("  Request ICY Meta", fmt.Sprintf("%t", config.ICEcast.HTTP.RequestICYMeta))
	if len(config.ICEcast.HTTP.CustomHeaders) > 0 {
		printKeyValue("  Custom Headers", fmt.Sprintf("(%d)", len(config.ICEcast.HTTP.CustomHeaders)))
		for key, value := range config.ICEcast.HTTP.CustomHeaders {
			printKeyValue("    "+key, value)
		}
	}

	printSubsection("Audio")
	printKeyValue("  Buffer Duration", config.ICEcast.Audio.BufferDuration.String())
	printKeyValue("  Sample Duration", config.ICEcast.Audio.SampleDuration.String())
	printKeyValue("  Max Read Attempts", fmt.Sprintf("%d", config.ICEcast.Audio.MaxReadAttempts))
	printKeyValue("  Read Timeout", config.ICEcast.Audio.ReadTimeout.String())
	printKeyValue("  Handle ICY Meta", fmt.Sprintf("%t", config.ICEcast.Audio.HandleICYMeta))
	printKeyValue("  Metadata Interval", fmt.Sprintf("%d", config.ICEcast.Audio.MetadataInterval))

	printSubsection("Metadata Extractor")
	printKeyValue("  Enable Header Mappings", fmt.Sprintf("%t", config.ICEcast.MetadataExtractor.EnableHeaderMappings))
	printKeyValue("  Enable ICY Metadata", fmt.Sprintf("%t", config.ICEcast.MetadataExtractor.EnableICYMetadata))
	printKeyValue("  ICY Metadata Timeout", config.ICEcast.MetadataExtractor.ICYMetadataTimeout.String())
	if len(config.ICEcast.MetadataExtractor.DefaultValues) > 0 {
		printKeyValue("  Default Values", fmt.Sprintf("(%d)", len(config.ICEcast.MetadataExtractor.DefaultValues)))
		for key, value := range config.ICEcast.MetadataExtractor.DefaultValues {
			printKeyValue("    "+key, fmt.Sprintf("%v", value))
		}
	}

	printSection("REGIONS")
	for name, region := range config.Regions {
		printSubsection(strings.ToUpper(name))
		printKeyValue("  Name", region.Name)
		printKeyValue("  Endpoint", region.Endpoint)
		printKeyValue("  Location", region.Location)
		printKeyValue("  Enabled", fmt.Sprintf("%t", region.Enabled))
		if len(region.Headers) > 0 {
			printKeyValue("  Headers", fmt.Sprintf("(%d)", len(region.Headers)))
			for key, value := range region.Headers {
				printKeyValue("    "+key, value)
			}
		}
	}

	printSection("PROFILES")
	for name, profile := range config.Profiles {
		printSubsection(strings.ToUpper(name))
		printKeyValue("  Name", profile.Name)
		printKeyValue("  Description", profile.Description)
		printKeyValue("  Duration", profile.Duration.String())
		printKeyValue("  Regions", fmt.Sprintf("(%d) %v", len(profile.Regions), profile.Regions))
		printKeyValue("  Metrics", fmt.Sprintf("(%d) %v", len(profile.Metrics), profile.Metrics))
		if len(profile.Streams) > 0 {
			printKeyValue("  Streams", fmt.Sprintf("(%d)", len(profile.Streams)))
			for i, stream := range profile.Streams {
				printKeyValue(fmt.Sprintf("    %d. %s (%s)", i+1, stream.Name, stream.Type), stream.URL)
			}
		}
		printKeyValue("  Thresholds", "")
		printKeyValue("    Min Similarity", fmt.Sprintf("%.3f", profile.Thresholds.MinSimilarity))
		printKeyValue("    Max Latency", profile.Thresholds.MaxLatency.String())
		printKeyValue("    Min Bitrate", fmt.Sprintf("%d", profile.Thresholds.MinBitrate))
		if len(profile.Tags) > 0 {
			printKeyValue("  Tags", fmt.Sprintf("(%d)", len(profile.Tags)))
			for key, value := range profile.Tags {
				printKeyValue("    "+key, value)
			}
		}
	}

	fmt.Println()
	fmt.Println(ColorGreen + strings.Repeat("-", 80))
	fmt.Println("CONFIGURATION TEST COMPLETED SUCCESSFULLY")
	fmt.Printf("Config file: %s\n", getConfigFilePath())
	fmt.Println(strings.Repeat("=", 80) + ColorReset)

	return nil
}

func printSection(title string) {
	fmt.Printf("\n%s\n", title)
	fmt.Println(strings.Repeat("-", len(title)))
}

func printSubsection(title string) {
	fmt.Printf("\n  %s\n", title)
}

func printKeyValue(key, value string) {
	if value == "" {
		fmt.Printf("%-35s\n", key)
	} else {
		fmt.Printf("%-35s %s\n", key+":", value)
	}
}

func getConfigFilePath() string {
	homeDir, _ := os.UserHomeDir()
	return fmt.Sprintf("%s/.config/cdn-benchmark/config.yaml", homeDir)
}
