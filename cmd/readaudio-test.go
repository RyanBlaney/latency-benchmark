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

var ( // rat => readaudio-test ¬‿¬
	ratVerbose         bool
	ratDebug           bool
	ratSegmentDuration time.Duration
	ratTimeout         time.Duration
)

var ratCmd = &cobra.Command{
	Use:   "readaudio-test [url1] [url2]",
	Short: "Validate streams via audio fingerprinting and comparison",
	Long: `Perform comprehensive audio fingerprinting and comparison through
	downloading short segments of audio and comparing the fingerprints through
	various methods.`,
	Args: cobra.ExactArgs(2),
	RunE: runReadAudioTest,
}

func init() {
	rootCmd.AddCommand(ratCmd)

	ratCmd.Flags().BoolVarP(&ratVerbose, "verbose", "v", false,
		"verbose output (overrides global verbose)")
	ratCmd.Flags().BoolVarP(&ratDebug, "debug", "d", false,
		"debug logging mode")
	ratCmd.Flags().DurationVarP(&ratSegmentDuration, "segment-duration", "t", time.Second*15,
		"the downloaded length of each stream")
	ratCmd.Flags().DurationVarP(&ratTimeout, "timeout", "T", time.Second*40,
		"timeout for stream operations")
}

func runReadAudioTest(cmd *cobra.Command, args []string) error {
	url1 := args[0]
	url2 := args[1]

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

	audioData1, err := handler1.ReadAudioWithDuration(ctx, time.Second*10)
	if err != nil {
		return fmt.Errorf("   %sFailed to extract audio data from stream: %v%s", ColorRed, err, ColorReset)
	}
	formattedData1, err := json.Marshal(audioData1)
	if err != nil {
		return fmt.Errorf("   %sFailed to marshal json for audio data 1: %v%s", ColorRed, err, ColorReset)
	}
	fmt.Printf("\nAudio Data for Stream 1: %s", string(formattedData1))

	// URL 2: ICEcast
	fmt.Printf("\n\nStream 2: ICEcast: %s\n", url2)
	handler2, err := streamFactory.DetectAndCreate(ctx, url2)
	if err != nil {
		return fmt.Errorf("   %sFailed to detect type and create handler: %v%s", ColorRed, err, ColorReset)
	}

	err = handler2.Connect(ctx, url2)
	if err != nil {
		return fmt.Errorf("   %sFailed to connect to stream: %v%s", ColorRed, err, ColorReset)
	}

	metadata2, err := handler2.GetMetadata()
	if err != nil {
		return fmt.Errorf("   %sFailed to extract metadata from stream: %v%s", ColorRed, err, ColorReset)
	}
	formattedMetadata2, err := json.Marshal(metadata2)
	if err != nil {
		return fmt.Errorf("   %sFailed to marshal json for audio data 2: %v%s", ColorRed, err, ColorReset)
	}
	fmt.Printf("\nMetadata for Stream 2: %s\n\n", string(formattedMetadata2))

	audioData2, err := handler2.ReadAudioWithDuration(ctx, time.Second*10)
	if err != nil {
		return fmt.Errorf("   %sFailed to extract audio data from stream: %v%s", ColorRed, err, ColorReset)
	}
	formattedData2, err := json.Marshal(audioData2)
	if err != nil {
		return fmt.Errorf("   %sFailed to marshal json for audio data 2: %v%s", ColorRed, err, ColorReset)
	}
	fmt.Printf("Audio Data for Stream 2: %s", string(formattedData2))

	if verbose {
		fmt.Printf("%s", appConfig.LogLevel)
	}

	return nil
}
