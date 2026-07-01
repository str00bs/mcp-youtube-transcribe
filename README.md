# YouTube Transcription MCP Server

A production-ready [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/spec) server for transcribing YouTube videos using local Whisper models. This server can be connected to AI assistants like [Vibe Code CLI](https://github.com/mistralai/vibe), [Cursor](https://www.cursor.com/), or any other MCP-compatible client.

## Features

- **Video Metadata**: Get title, description, duration, channel info, views, likes, and thumbnails
- **Caption Extraction**: List and download existing YouTube captions/subtitles
- **Audio Download**: Extract audio from YouTube videos in various formats (MP3, WAV, etc.)
- **Local Transcription**: On-device transcription using Whisper models (faster-whisper)
- **Word-Level Timestamps**: Precise timing for each word or segment
- **Multi-Language Support**: Transcribe videos in multiple languages
- **Automatic Cleanup**: Optional automatic cleanup of temporary files
- **GPU Support**: Automatic GPU detection and usage for faster transcription

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd mcp-youtube-transcribe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Optional: Install for GPU support (faster transcription)
pip install faster-whisper[cuda]

# Optional: Install cloud transcription services
pip install assemblyai  # For AssemblyAI
# pip install google-cloud-speech  # For Google Cloud
# pip install boto3  # For AWS Transcribe
```

### 2. Configure Environment Variables

Create a `.env` file or set environment variables:

```bash
# Transcription model (tiny, base, small, medium, large)
export WHISPER_MODEL=small

# Data directories
export DOWNLOAD_DIR=./data/downloads
export TEMP_DIR=./data/temp

# Maximum audio length in seconds (default: 3600 = 1 hour)
export MAX_AUDIO_LENGTH=3600

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
export LOG_LEVEL=INFO

# Enable/disable automatic cleanup of temp files
export CLEANUP_TEMP_FILES=true

# API Keys (optional, for cloud services)
export ASSEMBLYAI_API_KEY=your_assemblyai_key
# export GOOGLE_CLOUD_API_KEY=your_google_key
# export AWS_ACCESS_KEY_ID=your_aws_key
# export AWS_SECRET_ACCESS_KEY=your_aws_secret
# export AWS_REGION=us-east-1
```

Or use a `config.json` file:

```json
{
  "whisper_model": "small",
  "download_dir": "./data/downloads",
  "temp_dir": "./data/temp",
  "max_audio_length": 3600,
  "log_level": "INFO",
  "cleanup_temp_files": true,
  "assemblyai_api_key": "your_key"
}
```

### 3. Run the Server

#### Using stdio (recommended for MCP clients):

```bash
python main.py --stdio
```

#### Using HTTP:

```bash
python main.py --host 0.0.0.0 --port 8080
```

### 4. Connect to Your MCP Client

#### For Vibe Code CLI:

Add to your `~/.vibe/config.toml`:

```toml
[mcp_servers.youtube_transcription]
command = "python"
args = ["/path/to/mcp-youtube-transcribe/main.py", "--stdio"]
```

#### For Cursor:

Configure in Cursor's settings to add an MCP server pointing to the stdio command.

## Usage Examples

Once connected to an MCP client, you can use the following tools:

### Get Video Metadata

```python
metadata = await tools.youtube_transcription.get_video_metadata_tool(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
)
print(f"Title: {metadata.title}")
print(f"Duration: {metadata.duration_human}")
print(f"Channel: {metadata.channel}")
print(f"Views: {metadata.views}")
```

### List Available Captions

```python
captions = await tools.youtube_transcription.list_captions_tool(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
)
for cap in captions.captions:
    print(f"{cap.language} ({cap.language_code}): {cap.format}")
```

### Download Audio

```python
audio = await tools.youtube_transcription.download_audio_tool(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    format="mp3"
)
print(f"Downloaded to: {audio.audio_path}")
```

### Transcribe a YouTube Video

```python
transcript = await tools.youtube_transcription.transcribe_video(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    language="en",
    model="small",
    word_timestamps=True
)
print(f"Full text: {transcript.full_text}")
for segment in transcript.segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")
```

### Transcribe an Audio File

```python
transcript = await tools.youtube_transcription.transcribe_audio(
    audio_path="/path/to/audio.mp3",
    language="en",
    word_timestamps=True
)
print(transcript.full_text)
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `WHISPER_MODEL` | `small` | Whisper model size (tiny, base, small, medium, large) |
| `DOWNLOAD_DIR` | `./data/downloads` | Directory for downloaded audio files |
| `TEMP_DIR` | `./data/temp` | Directory for temporary files |
| `MAX_AUDIO_LENGTH` | `3600` | Maximum audio length in seconds |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `CLEANUP_TEMP_FILES` | `true` | Automatically clean up temporary files |


## Transcription Method

The server uses **local Whisper** (faster-whisper) for all transcription:

- **Pros**: Free, offline, no API limits, no external dependencies, privacy-friendly
- **Cons**: Slower on CPU, requires GPU for best performance
- **Models**: tiny, base, small, medium, large
- **Languages**: 99+ languages
- **Hardware**: Automatic GPU detection if CUDA is available

## Performance Tips

1. **Use GPU**: For local transcription, use a GPU with CUDA support:
   ```bash
   pip install faster-whisper[cuda]
   export CUDA_AVAILABLE=true
   ```

2. **Model Selection**:
   - `tiny`: Fastest, lowest accuracy
   - `base`: Good balance
   - `small`: Better accuracy, reasonable speed (default)
   - `medium`: High accuracy, requires more memory
   - `large`: Best accuracy, requires GPU

3. **Audio Format**: MP3 is recommended for compatibility. WAV provides best quality.

4. **Resource Limits**: Increase `MAX_AUDIO_LENGTH` for longer videos (be aware of memory usage).

## Supported Video URL Formats

- Standard: `https://www.youtube.com/watch?v=VIDEO_ID`
- Short: `https://youtu.be/VIDEO_ID`
- Embed: `https://www.youtube.com/embed/VIDEO_ID`
- Direct: `https://www.youtube.com/v/VIDEO_ID`
- Shorts: `https://www.youtube.com/shorts/VIDEO_ID`
- Plain ID: `VIDEO_ID` (11 characters)

## Error Handling

All operations include proper error handling and validation:

- Invalid video URLs are rejected early
- Missing dependencies raise clear error messages
- File operations are validated before execution
- Network errors are caught and reported
- Resource limits are enforced

## Security

- Filenames are sanitized to prevent path traversal
- Video IDs are validated before use
- Temporary files are cleaned up automatically (configurable)
- No sensitive data is logged

## Project Structure

```
mcp-youtube-transcribe/
├── main.py                 # Entry point
├── src/
│   └── server.py           # Main server implementation
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues or questions:
- Check the [MCP Documentation](https://github.com/modelcontextprotocol/spec)
- Open an issue in the repository
- Ask in the community discussions
