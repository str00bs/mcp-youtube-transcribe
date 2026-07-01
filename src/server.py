"""
YouTube Transcription MCP Server

A production-ready MCP server for transcribing YouTube videos.
Uses fastmcp for the MCP framework, yt-dlp for YouTube operations,
and faster-whisper for local transcription.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ServerConfig:
    """Server configuration loaded from environment variables."""
    whisper_model: str = "small"
    download_dir: Path = field(default_factory=lambda: Path("./data/downloads"))
    temp_dir: Path = field(default_factory=lambda: Path("./data/temp"))
    max_audio_length: int = 3600  # 1 hour
    log_level: str = "INFO"
    cleanup_temp_files: bool = True
    
    # API keys (optional)
    assemblyai_api_key: Optional[str] = None
    google_cloud_api_key: Optional[str] = None
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    aws_region: str = "us-east-1"
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        return cls(
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            download_dir=Path(os.getenv("DOWNLOAD_DIR", "./data/downloads")),
            temp_dir=Path(os.getenv("TEMP_DIR", "./data/temp")),
            max_audio_length=int(os.getenv("MAX_AUDIO_LENGTH", "3600")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            cleanup_temp_files=os.getenv("CLEANUP_TEMP_FILES", "true").lower() == "true",
            assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            google_cloud_api_key=os.getenv("GOOGLE_CLOUD_API_KEY"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if self.whisper_model not in valid_models:
            raise ValueError(f"Invalid WHISPER_MODEL: {self.whisper_model}. Must be one of {valid_models}")
        if self.max_audio_length <= 0:
            raise ValueError("MAX_AUDIO_LENGTH must be positive")


config = ServerConfig.from_env()
config.validate()

# Setup logging
logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("youtube_mcp")

# Ensure directories exist
config.download_dir.mkdir(parents=True, exist_ok=True)
config.temp_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Custom Exceptions
# ============================================================================

class YouTubeError(Exception):
    """Base exception for YouTube operations."""
    pass


class InvalidVideoURLError(YouTubeError):
    """Invalid YouTube video URL."""
    pass


class VideoNotFoundError(YouTubeError):
    """YouTube video not found."""
    pass


class DownloadError(YouTubeError):
    """Failed to download video/audio."""
    pass


class TranscriptionError(YouTubeError):
    """Failed to transcribe audio."""
    pass


class DependencyError(YouTubeError):
    """Required dependency not installed."""
    pass


# ============================================================================
# Pydantic Models
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
    """A caption track for a video."""
    language: str
    language_code: str
    is_auto_generated: bool
    format: str
    url: Optional[str] = None


class CaptionListResponse(BaseModel):
    """List of caption tracks."""
    video_id: str
    captions: List[CaptionTrack]
    has_auto_captions: bool


class AudioDownloadResponse(BaseModel):
    """Audio download response."""
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
    """Transcription response."""
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
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en', 'fr', 'es')"
    )
    model: Optional[str] = Field(
        default=None,
        description="Transcription model to use"
    )
    word_timestamps: bool = Field(
        default=True,
        description="Include word-level timestamps"
    )
    speaker_diarization: bool = Field(
        default=False,
        description="Identify different speakers"
    )
    translate: bool = Field(
        default=False,
        description="Translate to English"
    )
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        if v and not re.match(r'^[a-zA-Z]{2,3}(-[a-zA-Z]{2,3})?$', v):
            logger.warning(f"Invalid language code: {v}")
        return v


# ============================================================================
# Utility Functions
# ============================================================================

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    if not url or not isinstance(url, str):
        raise InvalidVideoURLError("Video URL cannot be empty")
    
    url = url.strip()
    
    # Short URL: youtu.be/VIDEO_ID
    if "youtu.be" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0].split("#")[0]
    
    # Standard URL: youtube.com/watch?v=VIDEO_ID
    if "youtube.com" in url:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0].split("?")[0].split("#")[0]
        for prefix in ["/embed/", "/v/", "/shorts/"]:
            if prefix in url:
                return url.split(prefix)[1].split("?")[0].split("&")[0].split("#")[0]
    
    # Plain ID
    if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url):
        return url
    
    raise InvalidVideoURLError(f"Cannot extract video ID from: {url}")


def validate_video_id(video_id: str) -> bool:
    """Validate YouTube video ID format."""
    return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    return sanitized.replace('..', '_')


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def get_audio_duration(audio_path: str) -> float:
    """Get audio file duration in seconds."""
    try:
        import wave
        with wave.open(audio_path, 'rb') as f:
            return f.getnframes() / f.getframerate()
    except Exception:
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
    return 0.0


# ============================================================================
# Resource Management
# ============================================================================

class TempFileManager:
    """Manage temporary files with automatic cleanup."""
    
    def __init__(self):
        self.tracked_files: List[Path] = []
    
    def track(self, file_path: Path) -> None:
        """Track a file for cleanup."""
        self.tracked_files.append(file_path)
        logger.debug(f"Tracking temp file: {file_path}")
    
    def cleanup(self, file_path: Optional[Path] = None) -> None:
        """Clean up tracked files."""
        if file_path:
            self._cleanup_file(file_path)
            if file_path in self.tracked_files:
                self.tracked_files.remove(file_path)
        elif config.cleanup_temp_files:
            for f in self.tracked_files[:]:
                self._cleanup_file(f)
                self.tracked_files.remove(f)
    
    def _cleanup_file(self, file_path: Path) -> None:
        """Clean up a single file."""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all tracked files."""
        for f in self.tracked_files[:]:
            self._cleanup_file(f)
        self.tracked_files.clear()


temp_files = TempFileManager()


# ============================================================================
# YouTube Functions
# ============================================================================

def get_video_metadata(video_id: str) -> VideoMetadata:
    """Get metadata for a YouTube video."""
    if not validate_video_id(video_id):
        raise InvalidVideoURLError(f"Invalid video ID: {video_id}")
    
    import yt_dlp
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': True}
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise VideoNotFoundError(f"Video not found: {video_id}")
            
            thumbnails = info.get('thumbnails', [])
            thumbnail_url = thumbnails[-1].get('url') if thumbnails else None
            
            return VideoMetadata(
                video_id=video_id,
                title=info.get('title', 'Unknown'),
                description=info.get('description'),
                duration_seconds=info.get('duration', 0),
                duration_human=format_duration(info.get('duration', 0)),
                upload_date=info.get('upload_date', '')[:8],
                channel=info.get('channel', 'Unknown'),
                channel_id=info.get('channel_id', ''),
                views=info.get('view_count'),
                likes=info.get('like_count'),
                thumbnail_url=thumbnail_url,
                is_live=info.get('is_live', False),
            )
    except Exception as e:
        raise YouTubeError(f"Failed to get metadata: {e}") from e


def list_captions(video_id: str) -> CaptionListResponse:
    """List available captions for a video."""
    if not validate_video_id(video_id):
        raise InvalidVideoURLError(f"Invalid video ID: {video_id}")
    
    import yt_dlp
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {'quiet': True, 'no_warnings': True, 'writesubtitles': False, 'allsubtitles': False}
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise VideoNotFoundError(f"Video not found: {video_id}")
            
            captions: List[CaptionTrack] = []
            has_auto = False
            
            # Process subtitles
            for lang_code, tracks in info.get('subtitles', {}).items():
                for track in tracks:
                    is_auto = track.get('name', '').lower() in ['auto-generated', 'a.en'] or track.get('ext') == 'vtt'
                    lang_norm = lang_code.split('-')[0] if '-' in lang_code else lang_code
                    captions.append(CaptionTrack(
                        language=track.get('name', lang_code),
                        language_code=lang_norm,
                        is_auto_generated=is_auto,
                        format=track.get('ext', 'unknown'),
                        url=track.get('url'),
                    ))
                    if is_auto:
                        has_auto = True
            
            # Process automatic captions
            for lang_code, tracks in info.get('automatic_captions', {}).items():
                for track in tracks:
                    lang_norm = lang_code.split('-')[0] if '-' in lang_code else lang_code
                    cap = CaptionTrack(
                        language=track.get('name', lang_code),
                        language_code=lang_norm,
                        is_auto_generated=True,
                        format=track.get('ext', 'vtt'),
                        url=track.get('url'),
                    )
                    if not any(c.language_code == lang_norm and c.is_auto_generated for c in captions):
                        captions.append(cap)
                        has_auto = True
            
            return CaptionListResponse(
                video_id=video_id,
                captions=captions,
                has_auto_captions=has_auto
            )
    except Exception as e:
        raise YouTubeError(f"Failed to list captions: {e}") from e


def _download_audio(video_id: str, output_format: str = "mp3") -> AudioDownloadResponse:
    """Download audio from a YouTube video."""
    if not validate_video_id(video_id):
        raise InvalidVideoURLError(f"Invalid video ID: {video_id}")
    
    if output_format not in ['mp3', 'wav', 'm4a', 'opus', 'ogg']:
        raise ValueError(f"Unsupported format: {output_format}")
    
    import yt_dlp
    
    safe_id = sanitize_filename(video_id)
    output_path = config.download_dir / f"{safe_id}.{output_format}"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': output_format,
            'preferredquality': '192',
        }],
        'outtmpl': str(output_path),
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        
        actual_path = str(output_path)
        if not os.path.exists(actual_path):
            # Find the actual downloaded file
            for f in config.download_dir.glob(f"{safe_id}.*"):
                if f.suffix[1:] in ['mp3', 'wav', 'm4a', 'opus', 'ogg']:
                    actual_path = str(f)
                    break
        
        if not os.path.exists(actual_path):
            raise DownloadError(f"Audio file not found: {actual_path}")
        
        # Get sample rate
        sample_rate = 44100
        try:
            if actual_path.endswith('.wav'):
                import wave
                with wave.open(actual_path, 'rb') as f:
                    sample_rate = f.getframerate()
        except Exception:
            pass
        
        temp_files.track(output_path) if output_path.exists() else temp_files.track(Path(actual_path))
        
        return AudioDownloadResponse(
            video_id=video_id,
            audio_path=actual_path,
            format=output_format,
            duration_seconds=info.get('duration', 0),
            sample_rate=sample_rate,
        )
    except Exception as e:
        raise DownloadError(f"Failed to download audio: {e}") from e


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
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise DependencyError("faster-whisper not installed. Run: pip install faster-whisper")
    
    # Check duration limit
    duration = get_audio_duration(audio_path)
    if duration > config.max_audio_length:
        raise TranscriptionError(
            f"Audio too long ({duration:.0f}s). Max: {config.max_audio_length}s"
        )
    
    model_name = model_size or config.whisper_model
    device = "cuda" if model_name != "tiny" and os.getenv("CUDA_AVAILABLE", "").lower() == "true" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    logger.info(f"Loading Whisper model: {model_name} (device: {device})")
    
    try:
        model = WhisperModel(
            model_size_or_path=model_name,
            device=device,
            compute_dtype=compute_type,
        )
    except Exception as e:
        logger.warning(f"Failed to load on {device}, falling back to CPU: {e}")
        model = WhisperModel(
            model_size_or_path=model_name,
            device="cpu",
            compute_dtype="int8",
        )
    
    logger.info("Starting Whisper transcription")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=word_timestamps,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    
    transcript_segments = []
    full_text = ""
    for seg in segments:
        transcript_segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else None,
        ))
        full_text += seg.text + " "
    
    return TranscriptResponse(
        audio_path=audio_path,
        language=language or "unknown",
        segments=transcript_segments,
        full_text=full_text.strip(),
        transcription_method="whisper",
        model_used=model_name,
        duration_seconds=info.get('duration', 0),
    )


async def transcribe_with_assemblyai(
    audio_path: str,
    language: Optional[str] = None,
    word_timestamps: bool = True,
    speaker_diarization: bool = False,
) -> TranscriptResponse:
    """Transcribe audio using AssemblyAI."""
    if not config.assemblyai_api_key:
        raise DependencyError("ASSEMBLYAI_API_KEY not configured")
    
    try:
        import assemblyai as aai
    except ImportError:
        raise DependencyError("assemblyai not installed. Run: pip install assemblyai")
    
    aai.settings.api_key = config.assemblyai_api_key
    transcriber = aai.Transcriber()
    
    transcribe_config = aai.TranscriptionConfig(
        language=language,
        word_timestamps=word_timestamps,
        speaker_diarization=speaker_diarization,
    )
    
    logger.info("Starting AssemblyAI transcription")
    transcript = transcriber.transcribe(audio_path, config=transcribe_config)
    
    if transcript.status == aai.TranscriptStatus.error:
        raise TranscriptionError(f"AssemblyAI failed: {transcript.error}")
    
    segments = []
    full_text = ""
    for utterance in transcript.utterances:
        segments.append(TranscriptSegment(
            start=utterance.start,
            end=utterance.end,
            text=utterance.text,
            speaker=utterance.speaker if speaker_diarization else None,
        ))
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


async def _transcribe_audio(
    audio_path: str,
    options: TranscriptionOptions,
) -> TranscriptResponse:
    """Transcribe an audio file using available methods."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check duration
    duration = get_audio_duration(audio_path)
    if duration > config.max_audio_length:
        raise TranscriptionError(
            f"Audio too long ({duration:.0f}s). Max: {config.max_audio_length}s"
        )
    
    # Try methods in order: AssemblyAI first, then Whisper
    methods = []
    if config.assemblyai_api_key:
        methods.append(("assemblyai", transcribe_with_assemblyai))
    methods.append(("whisper", transcribe_with_whisper))
    
    last_error = None
    for method_name, method_func in methods:
        try:
            return await method_func(
                audio_path,
                language=options.language,
                word_timestamps=options.word_timestamps,
                speaker_diarization=options.speaker_diarization,
                model_size=options.model,
            )
        except Exception as e:
            last_error = e
            logger.warning(f"Method {method_name} failed: {e}")
    
    raise TranscriptionError(f"All transcription methods failed: {last_error}")


async def _transcribe_video(
    video_id: str,
    options: TranscriptionOptions,
) -> TranscriptResponse:
    """Transcribe a YouTube video."""
    if not validate_video_id(video_id):
        raise InvalidVideoURLError(f"Invalid video ID: {video_id}")
    
    # Download audio
    audio_resp = _download_audio(video_id, output_format="mp3")
    
    try:
        # Transcribe
        transcript = await _transcribe_audio(audio_resp.audio_path, options)
        transcript.video_id = video_id
        return transcript
    finally:
        # Cleanup audio file
        if config.cleanup_temp_files:
            temp_files.cleanup(Path(audio_resp.audio_path))


# ============================================================================
# MCP Server
# ============================================================================

mcp = FastMCP(
    name="youtube-transcription",
    version="1.0.0",
    instructions="YouTube video transcription MCP server. Provides tools for extracting metadata, captions, audio, and transcriptions from YouTube videos.",
)


@mcp.tool(description="Get metadata for a YouTube video (title, description, duration, channel info, etc.)")
async def get_video_metadata(video_url: str) -> VideoMetadata:
    """Get metadata for a YouTube video."""
    video_id = extract_video_id(video_url)
    return get_video_metadata(video_id)


@mcp.tool(description="List available caption tracks for a YouTube video")
async def list_captions(video_url: str) -> CaptionListResponse:
    """List available caption tracks for a YouTube video."""
    video_id = extract_video_id(video_url)
    return list_captions(video_id)


@mcp.tool(description="Download audio from a YouTube video")
async def download_audio(
    video_url: str,
    format: str = "mp3",
) -> AudioDownloadResponse:
    """Download audio from a YouTube video.
    
    Args:
        video_url: YouTube video URL or video ID
        format: Audio format (mp3, wav, m4a, opus, ogg)
    """
    video_id = extract_video_id(video_url)
    return _download_audio(video_id, output_format=format)


@mcp.tool(description="Transcribe a YouTube video")
async def transcribe_video(
    video_url: str,
    language: Optional[str] = None,
    model: Optional[str] = None,
    word_timestamps: bool = True,
    speaker_diarization: bool = False,
    translate: bool = False,
) -> TranscriptResponse:
    """Transcribe a YouTube video.
    
    Args:
        video_url: YouTube video URL or video ID
        language: Language code (e.g., 'en', 'fr', 'es')
        model: Transcription model to use
        word_timestamps: Include word-level timestamps
        speaker_diarization: Identify different speakers
        translate: Translate to English
    """
    video_id = extract_video_id(video_url)
    options = TranscriptionOptions(
        language=language,
        model=model,
        word_timestamps=word_timestamps,
        speaker_diarization=speaker_diarization,
        translate=translate,
    )
    return await _transcribe_video(video_id, options)


@mcp.tool(description="Transcribe an audio file")
async def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    model: Optional[str] = None,
    word_timestamps: bool = True,
    speaker_diarization: bool = False,
    translate: bool = False,
) -> TranscriptResponse:
    """Transcribe an audio file.
    
    Args:
        audio_path: Path to audio file
        language: Language code (e.g., 'en', 'fr', 'es')
        model: Transcription model to use
        word_timestamps: Include word-level timestamps
        speaker_diarization: Identify different speakers
        translate: Translate to English
    """
    options = TranscriptionOptions(
        language=language,
        model=model,
        word_timestamps=word_timestamps,
        speaker_diarization=speaker_diarization,
        translate=translate,
    )
    return await _transcribe_audio(audio_path, options)


# ============================================================================
# Main Entry Point
# ============================================================================

async def main_async():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YouTube Transcription MCP Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)"
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Use stdio transport"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting YouTube Transcription MCP Server v1.0.0")
    logger.info(f"Whisper model: {config.whisper_model}")
    logger.info(f"Max audio length: {config.max_audio_length}s")
    
    if args.stdio:
        logger.info("Running in stdio mode...")
        await mcp.run_stdio_async()
    else:
        logger.info(f"Running on http://{args.host}:{args.port}")
        await mcp.run_http_async(host=args.host, port=args.port)


def main():
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
