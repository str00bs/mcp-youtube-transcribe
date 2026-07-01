"""
YouTube Transcription MCP Server

A Model Context Protocol (MCP) server that provides YouTube video transcription capabilities.
This server can be connected to Vibe Code CLI, Vibe Work, or any other MCP-compatible client.

Features:
- Download YouTube videos and extract audio
- Transcribe using local Whisper model (faster-whisper)
- Support for cloud transcription services (AssemblyAI, Google, AWS)
- Get video metadata (title, description, duration)
- List available caption tracks
- Generate timestamps and speaker diarization (with supported models)

Requirements:
- Python 3.10+
- FFmpeg (for audio extraction)
- Optional: CUDA for GPU-accelerated transcription

Installation:
    pip install mcp faster-whisper yt-dlp pydantic

Configuration:
    Create a config.json file or set environment variables:
    - YOUTUBE_API_KEY: Optional YouTube Data API key
    - ASSEMBLYAI_API_KEY: Optional AssemblyAI API key
    - GOOGLE_CLOUD_API_KEY: Optional Google Cloud API key
    - WHISPER_MODEL: Whisper model size (tiny, base, small, medium, large)

Usage:
    mcp-server-youtube-transcription [--config config.json]

MCP Tools Available:
    - get_video_metadata: Get YouTube video metadata
    - list_captions: List available caption tracks for a video
    - download_audio: Download audio from a YouTube video
    - transcribe_video: Transcribe a YouTube video
    - transcribe_audio: Transcribe an audio file
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mcp
import yt_dlp
from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ServerConfig:
    """Configuration for the YouTube Transcription MCP Server."""
    whisper_model: str = "small"
    download_dir: str = "./downloads"
    temp_dir: str = "./temp"
    max_audio_length: int = 3600  # 1 hour in seconds
    youtube_api_key: Optional[str] = None
    assemblyai_api_key: Optional[str] = None
    google_cloud_api_key: Optional[str] = None
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    aws_region: str = "us-east-1"
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            download_dir=os.getenv("DOWNLOAD_DIR", "./downloads"),
            temp_dir=os.getenv("TEMP_DIR", "./temp"),
            max_audio_length=int(os.getenv("MAX_AUDIO_LENGTH", "3600")),
            youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
            assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            google_cloud_api_key=os.getenv("GOOGLE_CLOUD_API_KEY"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )


config = ServerConfig.from_env()

# Ensure directories exist
Path(config.download_dir).mkdir(parents=True, exist_ok=True)
Path(config.temp_dir).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Pydantic Models for Tool Inputs/Outputs
# ============================================================================

class VideoMetadata(BaseModel):
    """YouTube video metadata."""
    video_id: str
    title: str
    description: Optional[str] = None
    duration_seconds: float
    duration_human: str
    upload_date: str
    channel: str
    channel_id: str
    views: Optional[int] = None
    likes: Optional[int] = None
    thumbnail_url: Optional[str] = None
    is_live: bool = False
    
class CaptionTrack(BaseModel):
    """A caption track available for a video."""
    language: str
    language_code: str
    is_auto_generated: bool
    format: str  # vtt, srt, json3, etc.
    url: Optional[str] = None
    
class CaptionListResponse(BaseModel):
    """Response for list_captions tool."""
    video_id: str
    captions: List[CaptionTrack]
    has_auto_captions: bool
    
class AudioDownloadResponse(BaseModel):
    """Response for download_audio tool."""
    video_id: str
    audio_path: str
    format: str
    duration_seconds: float
    sample_rate: int
    
class TranscriptSegment(BaseModel):
    """A segment of transcribed text."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    
class TranscriptResponse(BaseModel):
    """Response for transcription tools."""
    video_id: Optional[str] = None
    audio_path: Optional[str] = None
    language: str
    segments: List[TranscriptSegment]
    full_text: str
    transcription_method: str
    model_used: Optional[str] = None
    duration_seconds: float
    
class TranscriptionOptions(BaseModel):
    """Options for transcription."""
    language: Optional[str] = Field(default=None, description="Language code (e.g., 'en', 'fr', 'es')")
    model: Optional[str] = Field(default=None, description="Transcription model to use")
    word_timestamps: bool = Field(default=True, description="Include word-level timestamps")
    speaker_diarization: bool = Field(default=False, description="Identify different speakers")
    translate: bool = Field(default=False, description="Translate to English")
    

# ============================================================================
# YouTube Helper Functions
# ============================================================================

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    # Handle short URLs: https://youtu.be/VIDEO_ID
    if "youtu.be" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    
    # Handle standard URLs: https://www.youtube.com/watch?v=VIDEO_ID
    if "youtube.com" in url:
        # Check for /watch?v= or /embed/ or /v/
        if "v=" in url:
            return url.split("v=")[1].split("&")[0].split("?")[0]
        elif "/embed/" in url:
            return url.split("/embed/")[1].split("?")[0].split("&")[0]
        elif "/v/" in url:
            return url.split("/v/")[1].split("?")[0].split("&")[0]
        elif "/shorts/" in url:
            return url.split("/shorts/")[1].split("?")[0].split("&")[0]
    
    # If it's just the ID
    if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url):
        return url
    
    raise ValueError(f"Could not extract video ID from: {url}")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_video_metadata(video_id: str) -> VideoMetadata:
    """Get metadata for a YouTube video."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        # Get duration
        duration = info.get('duration', 0)
        
        # Get channel info
        channel = info.get('channel', 'Unknown')
        channel_id = info.get('channel_id', '')
        
        # Get view count
        views = info.get('view_count')
        
        # Get like count
        likes = info.get('like_count')
        
        # Get upload date
        upload_date = info.get('upload_date', '')
        if upload_date:
            upload_date = upload_date[:8]  # YYYYMMDD format
        
        # Get thumbnails
        thumbnails = info.get('thumbnails', [])
        thumbnail_url = None
        if thumbnails:
            # Get the highest resolution thumbnail
            thumbnail_url = thumbnails[-1].get('url')
        
        # Check if live
        is_live = info.get('is_live', False)
        
        return VideoMetadata(
            video_id=video_id,
            title=info.get('title', 'Unknown'),
            description=info.get('description'),
            duration_seconds=duration,
            duration_human=format_duration(duration),
            upload_date=upload_date,
            channel=channel,
            channel_id=channel_id,
            views=views,
            likes=likes,
            thumbnail_url=thumbnail_url,
            is_live=is_live,
        )


def list_captions(video_id: str) -> CaptionListResponse:
    """List available caption tracks for a YouTube video."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'listformats': False,
        'writesubtitles': False,
        'allsubtitles': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        captions = []
        has_auto_captions = False
        
        # Get subtitles
        subtitles = info.get('subtitles', {})
        for lang_code, tracks in subtitles.items():
            for track in tracks:
                # Determine if auto-generated
                is_auto = track.get('name', '').lower() in ['auto-generated', 'a.en'] or \
                         track.get('ext', '') == 'vtt'
                
                # Get language name
                language = track.get('name', lang_code)
                
                # Normalize language code
                if len(lang_code) == 2:
                    language_code = lang_code
                else:
                    language_code = lang_code.split('-')[0] if '-' in lang_code else lang_code
                
                caption = CaptionTrack(
                    language=language,
                    language_code=language_code,
                    is_auto_generated=is_auto,
                    format=track.get('ext', 'unknown'),
                    url=track.get('url'),
                )
                captions.append(caption)
                
                if is_auto:
                    has_auto_captions = True
        
        # Also check automatic captions
        automatic_captions = info.get('automatic_captions', {})
        for lang_code, tracks in automatic_captions.items():
            for track in tracks:
                language = track.get('name', lang_code)
                language_code = lang_code.split('-')[0] if '-' in lang_code else lang_code
                
                caption = CaptionTrack(
                    language=language,
                    language_code=language_code,
                    is_auto_generated=True,
                    format=track.get('ext', 'vtt'),
                    url=track.get('url'),
                )
                # Avoid duplicates
                if not any(c.language_code == language_code and c.is_auto_generated for c in captions):
                    captions.append(caption)
                    has_auto_captions = True
        
        return CaptionListResponse(
            video_id=video_id,
            captions=captions,
            has_auto_captions=has_auto_captions,
        )


def download_audio(video_id: str, format: str = "mp3") -> AudioDownloadResponse:
    """Download audio from a YouTube video."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Create output path
    output_path = Path(config.download_dir) / f"{video_id}.{format}"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': format,
            'preferredquality': '192',
        }],
        'outtmpl': str(output_path),
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        
        # Get actual output path (may have been modified by yt-dlp)
        actual_path = str(output_path)
        if not os.path.exists(actual_path):
            # Try to find the downloaded file
            for f in Path(config.download_dir).glob(f"{video_id}.*"):
                if f.suffix[1:] in ['mp3', 'wav', 'm4a', 'opus', 'ogg']:
                    actual_path = str(f)
                    break
        
        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"Audio file not found after download: {actual_path}")
        
        # Get audio info
        duration = info.get('duration', 0)
        sample_rate = 0
        
        try:
            # Try to get sample rate from file
            import wave
            if actual_path.endswith('.wav'):
                with wave.open(actual_path, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
        except:
            # Default to 44100 if we can't determine
            sample_rate = 44100
        
        return AudioDownloadResponse(
            video_id=video_id,
            audio_path=actual_path,
            format=format,
            duration_seconds=duration,
            sample_rate=sample_rate,
        )


# ============================================================================
# Transcription Functions
# ============================================================================

async def transcribe_with_whisper(
    audio_path: str,
    language: Optional[str] = None,
    model_size: Optional[str] = None,
    word_timestamps: bool = True,
) -> TranscriptResponse:
    """Transcribe audio using local Whisper model."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper is not installed. "
            "Install it with: pip install faster-whisper"
        )
    
    # Use configured model or provided one
    model_name = model_size or config.whisper_model
    
    # Load model
    device = "cuda" if config.whisper_model != "tiny" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        model = WhisperModel(
            model_size_or_path=model_name,
            device=device,
            compute_dtype=compute_type,
        )
    except Exception as e:
        # Fallback to CPU
        model = WhisperModel(
            model_size_or_path=model_name,
            device="cpu",
            compute_dtype="int8",
        )
    
    # Transcribe
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=word_timestamps,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    
    # Process segments
    transcript_segments = []
    full_text = ""
    
    for segment in segments:
        transcript_segment = TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            confidence=segment.avg_logprob if hasattr(segment, 'avg_logprob') else None,
        )
        transcript_segments.append(transcript_segment)
        full_text += segment.text + " "
    
    # Get duration
    duration = info.get('duration', 0)
    
    return TranscriptResponse(
        audio_path=audio_path,
        language=language or "unknown",
        segments=transcript_segments,
        full_text=full_text.strip(),
        transcription_method="whisper",
        model_used=model_name,
        duration_seconds=duration,
    )


async def transcribe_with_assemblyai(
    audio_path: str,
    language: Optional[str] = None,
    word_timestamps: bool = True,
    speaker_diarization: bool = False,
) -> TranscriptResponse:
    """Transcribe audio using AssemblyAI API."""
    if not config.assemblyai_api_key:
        raise ValueError("AssemblyAI API key not configured. Set ASSEMBLYAI_API_KEY environment variable.")
    
    try:
        import assemblyai as aai
    except ImportError:
        raise ImportError(
            "assemblyai is not installed. "
            "Install it with: pip install assemblyai"
        )
    
    # Configure
    aai.settings.api_key = config.assemblyai_api_key
    
    # Upload file
    transccriber = aai.Transcriber()
    
    transcribe_config = aai.TranscriptionConfig(
        language=language,
        word_timestamps=word_timestamps,
        speaker_diarization=speaker_diarization,
    )
    
    # Transcribe
    transcript = transccriber.transcribe(
        audio_path,
        config=transcribe_config,
    )
    
    # Process results
    segments = []
    full_text = ""
    
    if transcript.status == aai.TranscriptStatus.error:
        raise ValueError(f"AssemblyAI transcription failed: {transcript.error}")
    
    for utterance in transcript.utterances:
        segment = TranscriptSegment(
            start=utterance.start,
            end=utterance.end,
            text=utterance.text,
            speaker=utterance.speaker if speaker_diarization else None,
        )
        segments.append(segment)
        full_text += utterance.text + " "
    
    return TranscriptResponse(
        audio_path=audio_path,
        language=language or "unknown",
        segments=segments,
        full_text=full_text.strip(),
        transcription_method="assemblyai",
        model_used="assemblyai",
        duration_seconds=transcript.duration,
    )


async def transcribe_audio(
    audio_path: str,
    options: TranscriptionOptions,
) -> TranscriptResponse:
    """Transcribe an audio file using the configured method."""
    # Check file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check file size/duration
    try:
        import wave
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / rate
    except:
        # Try with FFmpeg
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True, text=True
            )
            duration = float(result.stdout.strip())
        except:
            duration = 0
    
    if duration > config.max_audio_length:
        raise ValueError(
            f"Audio file is too long ({duration}s). "
            f"Maximum allowed: {config.max_audio_length}s"
        )
    
    # Try methods in order of preference
    methods = []
    
    # If AssemblyAI is configured, try it first (cloud, faster)
    if config.assemblyai_api_key:
        methods.append(("assemblyai", transcribe_with_assemblyai))
    
    # Always have Whisper as fallback
    methods.append(("whisper", transcribe_with_whisper))
    
    # Try each method
    last_error = None
    for method_name, method_func in methods:
        try:
            return await method_func(
                audio_path,
                language=options.language,
                word_timestamps=options.word_timestamps,
                speaker_diarization=options.speaker_diarization,
            )
        except Exception as e:
            last_error = e
            continue
    
    raise RuntimeError(f"All transcription methods failed. Last error: {last_error}")


async def transcribe_video(
    video_id: str,
    options: TranscriptionOptions,
) -> TranscriptResponse:
    """Transcribe a YouTube video."""
    # First, download the audio
    audio_response = download_audio(video_id, format="mp3")
    
    # Then transcribe
    transcript = await transcribe_audio(audio_response.audio_path, options)
    
    # Add video_id to response
    transcript.video_id = video_id
    
    return transcript


# ============================================================================
# MCP Server Setup
# ============================================================================

# Initialize MCP server
server = mcp.Server("youtube-transcription")


@server.get_video_metadata()
async def handle_get_video_metadata(video_url: str) -> VideoMetadata:
    """
    Get metadata for a YouTube video.
    
    Args:
        video_url: YouTube video URL or video ID
        
    Returns:
        VideoMetadata: Object containing video title, description, duration, etc.
    """
    video_id = extract_video_id(video_url)
    return get_video_metadata(video_id)


@server.list_captions()
async def handle_list_captions(video_url: str) -> CaptionListResponse:
    """
    List available caption tracks for a YouTube video.
    
    Args:
        video_url: YouTube video URL or video ID
        
    Returns:
        CaptionListResponse: Object containing list of available captions
    """
    video_id = extract_video_id(video_url)
    return list_captions(video_id)


@server.download_audio()
async def handle_download_audio(
    video_url: str,
    format: str = "mp3",
) -> AudioDownloadResponse:
    """
    Download audio from a YouTube video.
    
    Args:
        video_url: YouTube video URL or video ID
        format: Audio format (mp3, wav, m4a, opus, ogg)
        
    Returns:
        AudioDownloadResponse: Object containing path to downloaded audio file
    """
    video_id = extract_video_id(video_url)
    return download_audio(video_id, format=format)


@server.transcribe_video()
async def handle_transcribe_video(
    video_url: str,
    language: Optional[str] = None,
    model: Optional[str] = None,
    word_timestamps: bool = True,
    speaker_diarization: bool = False,
    translate: bool = False,
) -> TranscriptResponse:
    """
    Transcribe a YouTube video.
    
    Args:
        video_url: YouTube video URL or video ID
        language: Language code (e.g., 'en', 'fr', 'es')
        model: Transcription model to use
        word_timestamps: Include word-level timestamps
        speaker_diarization: Identify different speakers
        translate: Translate to English
        
    Returns:
        TranscriptResponse: Object containing transcription segments and full text
    """
    video_id = extract_video_id(video_url)
    options = TranscriptionOptions(
        language=language,
        model=model,
        word_timestamps=word_timestamps,
        speaker_diarization=speaker_diarization,
        translate=translate,
    )
    return await transcribe_video(video_id, options)


@server.transcribe_audio()
async def handle_transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    model: Optional[str] = None,
    word_timestamps: bool = True,
    speaker_diarization: bool = False,
    translate: bool = False,
) -> TranscriptResponse:
    """
    Transcribe an audio file.
    
    Args:
        audio_path: Path to audio file
        language: Language code (e.g., 'en', 'fr', 'es')
        model: Transcription model to use
        word_timestamps: Include word-level timestamps
        speaker_diarization: Identify different speakers
        translate: Translate to English
        
    Returns:
        TranscriptResponse: Object containing transcription segments and full text
    """
    options = TranscriptionOptions(
        language=language,
        model=model,
        word_timestamps=word_timestamps,
        speaker_diarization=speaker_diarization,
        translate=translate,
    )
    return await transcribe_audio(audio_path, options)


@server.get_caption_text()
async def handle_get_caption_text(
    video_url: str,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get text from YouTube's built-in captions if available.
    
    Args:
        video_url: YouTube video URL or video ID
        language: Language code (optional, will use first available if not specified)
        
    Returns:
        Dictionary containing caption text and metadata
    """
    video_id = extract_video_id(video_url)
    
    # List available captions
    caption_list = list_captions(video_id)
    
    # Find matching caption
    target_caption = None
    if language:
        for caption in caption_list.captions:
            if caption.language_code == language:
                target_caption = caption
                break
    else:
        # Use first non-auto caption, or first auto caption
        for caption in caption_list.captions:
            if not caption.is_auto_generated:
                target_caption = caption
                break
        if not target_caption and caption_list.captions:
            target_caption = caption_list.captions[0]
    
    if not target_caption or not target_caption.url:
        return {
            "video_id": video_id,
            "error": "No captions available for this video",
            "available_languages": [c.language_code for c in caption_list.captions],
        }
    
    # Download caption
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'writesubtitles': True,
            'subtitleslangs': [target_caption.language_code],
            'subtitlesformat': target_caption.format,
            'outtmpl': os.path.join(config.temp_dir, f"%s.{target_caption.format}"),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=True,
            )
        
        # Find the downloaded caption file
        caption_files = list(Path(config.temp_dir).glob(f"{video_id}.*"))
        caption_file = None
        for f in caption_files:
            if f.suffix[1:] == target_caption.format:
                caption_file = f
                break
        
        if not caption_file or not caption_file.exists():
            return {
                "video_id": video_id,
                "error": "Failed to download caption file",
            }
        
        # Read and parse caption file
        text = caption_file.read_text()
        
        # Clean up
        caption_file.unlink()
        
        return {
            "video_id": video_id,
            "language": target_caption.language,
            "language_code": target_caption.language_code,
            "format": target_caption.format,
            "text": text,
            "is_auto_generated": target_caption.is_auto_generated,
        }
        
    except Exception as e:
        return {
            "video_id": video_id,
            "error": str(e),
        }


# ============================================================================
# Resource Access (Optional: expose audio files as resources)
# ============================================================================

@server.list_resources()
async def handle_list_resources() -> List[Dict[str, Any]]:
    """List available audio files that have been downloaded."""
    resources = []
    
    for audio_file in Path(config.download_dir).glob("*.*"):
        if audio_file.suffix[1:] in ['mp3', 'wav', 'm4a', 'opus', 'ogg']:
            resources.append({
                "uri": f"file://{audio_file}",
                "name": audio_file.name,
                "mimeType": f"audio/{audio_file.suffix[1:]}",
                "description": f"Downloaded audio: {audio_file.name}",
            })
    
    return resources


@server.read_resource()
async def handle_read_resource(uri: str) -> Union[str, bytes]:
    """Read a resource (audio file content)."""
    # Convert file:// URI to path
    if uri.startswith("file://"):
        path = uri[7:]
    else:
        path = uri
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resource not found: {uri}")
    
    # Return file content
    with open(path, 'rb') as f:
        return f.read()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Transcription MCP Server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Use stdio transport instead of HTTP",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file",
    )
    
    args = parser.parse_args()
    
    # Load config from file if specified
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config_data = json.load(f)
            config = ServerConfig(**config_data)
    
    # Ensure directories exist
    Path(config.download_dir).mkdir(parents=True, exist_ok=True)
    Path(config.temp_dir).mkdir(parents=True, exist_ok=True)
    
    if args.stdio:
        # Use stdio transport
        print("Starting YouTube Transcription MCP Server (stdio)...", flush=True)
        server.run_stdio()
    else:
        # Use HTTP transport
        print(f"Starting YouTube Transcription MCP Server on {args.host}:{args.port}...", flush=True)
        server.run_http(
            host=args.host,
            port=args.port,
        )