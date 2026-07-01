# YouTube Transcription MCP Server - Documentation

A Model Context Protocol (MCP) server that provides YouTube video transcription capabilities for AI agents and tools.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install mcp faster-whisper yt-dlp pydantic

# Optional: Install for GPU support (CUDA)
pip install faster-whisper[cuda]

# Optional: For cloud transcription services
pip install assemblyai google-cloud-speech boto3
```

### 2. Configure Environment Variables

Create a `.env` file or set environment variables:

```bash
# Transcription model (tiny, base, small, medium, large)
export WHISPER_MODEL=small

# Download directories
export DOWNLOAD_DIR=./downloads
export TEMP_DIR=./temp

# Maximum audio length in seconds (default: 3600 = 1 hour)
export MAX_AUDIO_LENGTH=3600

# API Keys (optional, for cloud services)
export ASSEMBLYAI_API_KEY=your_assemblyai_key
export GOOGLE_CLOUD_API_KEY=your_google_key
export AWS_ACCESS_KEY_ID=your_aws_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret
export AWS_REGION=us-east-1
```

Or create a `config.json` file:

```json
{
  "whisper_model": "small",
  "download_dir": "./downloads",
  "temp_dir": "./temp",
  "max_audio_length": 3600,
  "assemblyai_api_key": "your_assemblyai_key"
}
```

### 3. Run the Server

#### Using stdio (recommended for Vibe CLI):

```bash
python mcp_server.py --stdio
```

#### Using HTTP:

```bash
python mcp_server.py --host 0.0.0.0 --port 8080
```

### 4. Connect to Vibe Code CLI

Add to your `~/.vibe/config.toml`:

```toml
[mcp_servers.youtube_transcription]
command = "python"
args = ["/path/to/mcp_server.py", "--stdio"]
```

Or run with environment variables:

```toml
[mcp_servers.youtube_transcription]
command = "bash"
args = ["-c", "WHISPER_MODEL=small python /path/to/mcp_server.py --stdio"]
```

## ✨ Features

- ✅ **Video Metadata**: Get title, description, duration, channel info, and more
- ✅ **Caption Extraction**: Download existing YouTube captions when available
- ✅ **Audio Download**: Extract audio from YouTube videos in various formats
- ✅ **Local Transcription**: Use Whisper models (faster-whisper) for on-device transcription
- ✅ **Cloud Transcription**: Support for AssemblyAI, Google Cloud, and AWS Transcribe
- ✅ **Timestamped Transcripts**: Word-level and segment-level timestamps
- ✅ **Speaker Diarization**: Identify different speakers (with supported services)
- ✅ **Multi-language Support**: Transcribe in multiple languages

## 🛠️ Usage Examples

### Get Video Metadata

```python
metadata = await tools.youtube_transcription.get_video_metadata(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
)
print(metadata.title)
print(metadata.duration_human)
```

### List Available Captions

```python
captions = await tools.youtube_transcription.list_captions(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
)
for caption in captions.captions:
    print(f"{caption.language} ({caption.language_code}): {caption.format}")
```

### Download Audio

```python
audio = await tools.youtube_transcription.download_audio(
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
    word_timestamps=True,
    speaker_diarization=False
)
print(transcript.full_text)
for segment in transcript.segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")
```

### Transcribe an Audio File

```python
transcript = await tools.youtube_transcription.transcribe_audio(
    audio_path="/path/to/audio.mp3",
    language="en",
    model="small"
)
print(transcript.full_text)
```

### Get Existing Caption Text

```python
caption_text = await tools.youtube_transcription.get_caption_text(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    language="en"
)
print(caption_text["text"])
```

## 📋 Available Tools


| Tool                 | Description                                        | Parameters                                                                               |
| -------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `get_video_metadata` | Get YouTube video metadata (title, duration, etc.) | `video_url`                                                                              |
| `list_captions`      | List available caption tracks for a video          | `video_url`                                                                              |
| `download_audio`     | Download audio from a YouTube video                | `video_url`, `format`                                                                    |
| `transcribe_video`   | Transcribe a YouTube video directly                | `video_url`, `language`, `model`, `word_timestamps`, `speaker_diarization`, `translate`  |
| `transcribe_audio`   | Transcribe an audio file                           | `audio_path`, `language`, `model`, `word_timestamps`, `speaker_diarization`, `translate` |
| `get_caption_text`   | Get text from YouTube's built-in captions          | `video_url`, `language`                                                                  |


## 🎯 Transcription Methods

The server supports multiple transcription backends:

### 1. Local Whisper (faster-whisper)

- **Pros**: Free, offline, no API limits
- **Cons**: Slower on CPU, requires GPU for best performance
- **Models**: tiny, base, small, medium, large
- **Languages**: 99+ languages

### 2. AssemblyAI (Cloud)

- **Pros**: Fast, accurate, speaker diarization
- **Cons**: Requires API key, has costs
- **Setup**: Set `ASSEMBLYAI_API_KEY` environment variable

### 3. Google Cloud Speech-to-Text

- **Pros**: High accuracy, many languages
- **Cons**: Requires API key, has costs
- **Setup**: Set `GOOGLE_CLOUD_API_KEY` environment variable

### 4. AWS Transcribe

- **Pros**: Scalable, enterprise-ready
- **Cons**: Requires AWS credentials, has costs
- **Setup**: Set AWS environment variables

**Priority**: The server will try cloud services first (if configured), then fall back to local Whisper.

## ⚡ Performance Tips

1. **Use GPU**: For local transcription, use a GPU with CUDA support
  ```bash
   pip install faster-whisper[cuda]
   export WHISPER_MODEL=medium  # or large for better accuracy
  ```
2. **Model Selection**:
  - `tiny`: Fastest, lowest accuracy
  - `base`: Good balance
  - `small`: Better accuracy, reasonable speed
  - `medium`: High accuracy, requires more memory
  - `large`: Best accuracy, requires GPU
3. **Batch Processing**: For multiple videos, process them sequentially to avoid memory issues
4. **Audio Format**: MP3 is recommended for compatibility. WAV provides best quality for transcription.

## 🔧 Configuration Options


| Option                  | Default       | Description                     |
| ----------------------- | ------------- | ------------------------------- |
| `WHISPER_MODEL`         | `small`       | Whisper model size              |
| `DOWNLOAD_DIR`          | `./downloads` | Directory for downloaded audio  |
| `TEMP_DIR`              | `./temp`      | Directory for temporary files   |
| `MAX_AUDIO_LENGTH`      | `3600`        | Maximum audio length in seconds |
| `ASSEMBLYAI_API_KEY`    | -             | AssemblyAI API key              |
| `GOOGLE_CLOUD_API_KEY`  | -             | Google Cloud API key            |
| `AWS_ACCESS_KEY_ID`     | -             | AWS access key                  |
| `AWS_SECRET_ACCESS_KEY` | -             | AWS secret key                  |
| `AWS_REGION`            | `us-east-1`   | AWS region                      |


## 🐛 Troubleshooting

### "faster-whisper not installed"

```bash
pip install faster-whisper
```

### "No module named 'mcp'"

```bash
pip install mcp
```

### "FFmpeg not found"

Install FFmpeg on your system:

- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **Mac**: `brew install ffmpeg`
- **Windows**: Download from [https://ffmpeg.org](https://ffmpeg.org)

### "CUDA not available"

Install CUDA drivers and cuDNN, then:

```bash
pip install faster-whisper[cuda]
```

### "Video not found"

Ensure the YouTube URL is valid and the video is publicly accessible.

### "Audio too long"

Increase `MAX_AUDIO_LENGTH` environment variable or split long videos.

## 📚 Advanced Usage

### Custom Whisper Model

You can use custom Whisper models from Hugging Face:

```bash
export WHISPER_MODEL=/path/to/custom/model
```

### Multiple Servers

Run multiple instances with different configurations:

```toml
[mcp_servers.youtube_transcription_en]
command = "bash"
args = ["-c", "WHISPER_MODEL=medium LANGUAGE=en python mcp_server.py --stdio"]

[mcp_servers.youtube_transcription_fr]
command = "bash"
args = ["-c", "WHISPER_MODEL=medium LANGUAGE=fr python mcp_server.py --stdio"]
```

### Using with Other MCP Clients

The server can be connected to any MCP-compatible client, not just Vibe Code CLI. Refer to your client's documentation for connecting MCP servers.

## 📝 Changelog

- **1.0.0**: Initial release with Whisper and AssemblyAI support
- **1.1.0**: Added Google Cloud and AWS Transcribe support
- **1.2.0**: Added caption extraction and audio download features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License

## 🆘 Support

For issues or questions:

- Check the [MCP Documentation](https://docs.mistral.ai/vibe/code/cli/mcp-servers)
- Open an issue in the repository
- Ask in the Mistral AI community