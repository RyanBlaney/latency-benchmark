package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/tunein/cdn-benchmark-cli/configs"
	"github.com/tunein/cdn-benchmark-cli/pkg/stream"
)

var (
	fingerprintVerbose         bool
	fingerprintDebug           bool
	fingerprintSegmentDuration time.Duration
	fingerprintTimeout         time.Duration
)

var fingerprintCmd = &cobra.Command{
	Use:   "fingerprint [url1] [url2]",
	Short: "Validate streams via audio fingerprinting and comparison",
	Long: `Perform comprehensive audio fingerprinting and comparison through
	downloading short segments of audio and comparing the fingerprints through
	various methods.`,
	Args: cobra.ExactArgs(2),
	RunE: runFingerprint,
}

func init() {
	rootCmd.AddCommand(fingerprintCmd)

	fingerprintCmd.Flags().BoolVarP(&fingerprintVerbose, "verbose", "v", false,
		"verbose output (overrides global verbose)")
	fingerprintCmd.Flags().BoolVarP(&fingerprintDebug, "debug", "d", false,
		"debug logging mode")
	fingerprintCmd.Flags().DurationVarP(&fingerprintSegmentDuration, "segment-duration", "t", time.Second*15,
		"the downloaded length of each stream")
	fingerprintCmd.Flags().DurationVarP(&fingerprintTimeout, "timeout", "", time.Second*30,
		"timeout for stream operations")
}

func runFingerprint(cmd *cobra.Command, args []string) error {
	url1 := args[0]
	//url2 := args[1]

	verbose := fingerprintVerbose || viper.GetBool("verbose")

	fmt.Printf("Stream Fingerprint Comparison\n")
	fmt.Printf("=============================\n\n")

	ctx, cancel := context.WithTimeout(context.Background(), fingerprintTimeout)
	defer cancel()

	timer := NewPerformanceTimer()
	timer.StartEvent("overall")

	timer.StartEvent("config_loading")
	fmt.Printf("   Configuration Loading...\n")

	appConfig, err := configs.LoadConfig()
	if err != nil {
		return fmt.Errorf("   %sFailed to load config: %v%s", ColorRed, err, ColorReset)
	}
	fmt.Printf("   %sApplication configuration loaded%s\n\n", ColorGreen, ColorReset)

	timer.StartEvent("stream_detection")
	fmt.Printf("   Detecting stream types...\n")

	streamFactory := stream.NewFactory()

	handler1, err := streamFactory.DetectAndCreate(ctx, url1)
	if err != nil {
		return fmt.Errorf("   %sFailed to detect type and create handler: %v%s", ColorRed, err, ColorReset)
	}

	err = handler1.Connect(ctx, url1)
	if err != nil {
		return fmt.Errorf("   %sFailed to connect to stream: %v%s", ColorRed, err, ColorReset)
	}

	metadata1, err := handler1.GetMetadata()
	if err != nil {
		return fmt.Errorf("   %sFailed to extract metadata from stream: %v%s", ColorRed, err, ColorReset)
	}
	formattedMetadata1, err := json.Marshal(metadata1)
	if err != nil {
		return fmt.Errorf("   %sFailed to marshal json for audio data 1: %v%s", ColorRed, err, ColorReset)
	}
	fmt.Printf("Metadata for Stream 1: %s", string(formattedMetadata1))

	audioData1, err := handler1.ReadAudio(ctx)
	if err != nil {
		return fmt.Errorf("   %sFailed to extract audio data from stream: %v%s", ColorRed, err, ColorReset)
	}
	formattedData1, err := json.Marshal(audioData1)
	if err != nil {
		return fmt.Errorf("   %sFailed to marshal json for audio data 1: %v%s", ColorRed, err, ColorReset)
	}
	fmt.Printf("Audio Data for Stream 1: %s", string(formattedData1))

	if verbose {
		fmt.Printf("%s", appConfig.LogLevel)
	}

	return nil
}
