"""Tests for YouTube Transcription MCP Server."""

import pytest
import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from server import (
    extract_video_id,
    validate_video_id,
    sanitize_filename,
    format_duration,
    InvalidVideoURLError,
    YouTubeError,
    ServerConfig,
    VideoMetadata,
    CaptionTrack,
    CaptionListResponse,
    AudioDownloadResponse,
    TranscriptSegment,
    TranscriptResponse,
    TranscriptionOptions,
    mcp,
)


class TestExtractVideoId:
    """Test video ID extraction from various URL formats."""
    
    def test_standard_url(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    
    def test_short_url(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    
    def test_embed_url(self):
        assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    
    def test_v_url(self):
        assert extract_video_id("https://www.youtube.com/v/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    
    def test_shorts_url(self):
        assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    
    def test_plain_id(self):
        assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    
    def test_url_with_params(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s") == "dQw4w9WgXcQ"
    
    def test_url_with_hash(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ#t=10") == "dQw4w9WgXcQ"
    
    def test_invalid_url(self):
        with pytest.raises(InvalidVideoURLError):
            extract_video_id("invalid")
    
    def test_empty_url(self):
        with pytest.raises(InvalidVideoURLError):
            extract_video_id("")
    
    def test_none_url(self):
        with pytest.raises(InvalidVideoURLError):
            extract_video_id(None)


class TestValidateVideoId:
    """Test video ID validation."""
    
    def test_valid_id(self):
        assert validate_video_id("dQw4w9WgXcQ") is True
    
    def test_invalid_length(self):
        assert validate_video_id("abc") is False
        assert validate_video_id("a" * 12) is False
    
    def test_invalid_characters(self):
        assert validate_video_id("abc!@#") is False
        assert validate_video_id("abc def") is False


class TestSanitizeFilename:
    """Test filename sanitization."""
    
    def test_special_characters(self):
        assert sanitize_filename("test<>:\"/|?*") == "test________"
    
    def test_directory_traversal(self):
        assert sanitize_filename("../test") == "__test"
        # "test/../file" becomes "test___file" because each "/" is replaced with "_" and ".." becomes "__"
        assert sanitize_filename("test/../file") == "test___file"
    
    def test_normal_filename(self):
        assert sanitize_filename("test.mp3") == "test.mp3"


class TestFormatDuration:
    """Test duration formatting."""
    
    def test_hours_minutes_seconds(self):
        assert format_duration(3661) == "1h 1m 1s"
    
    def test_minutes_seconds(self):
        assert format_duration(125) == "2m 5s"
    
    def test_seconds_only(self):
        assert format_duration(45) == "45s"
    
    def test_zero(self):
        assert format_duration(0) == "0s"


class TestServerConfig:
    """Test server configuration."""
    
    def test_default_config(self):
        config = ServerConfig()
        assert config.whisper_model == "small"
        assert config.max_audio_length == 3600
        assert config.log_level == "INFO"
        assert config.cleanup_temp_files is True
    
    def test_validation(self):
        config = ServerConfig()
        config.validate()  # Should not raise
    
    def test_invalid_model(self):
        config = ServerConfig(whisper_model="invalid")
        with pytest.raises(ValueError):
            config.validate()
    
    def test_invalid_max_length(self):
        config = ServerConfig(max_audio_length=-1)
        with pytest.raises(ValueError):
            config.validate()


class TestPydanticModels:
    """Test Pydantic models."""
    
    def test_video_metadata(self):
        metadata = VideoMetadata(
            video_id="test123",
            title="Test Video",
            duration_seconds=120.5,
            duration_human="2m 0s",
            upload_date="20240101",
            channel="Test Channel",
            channel_id="UC123",
        )
        assert metadata.video_id == "test123"
        assert metadata.title == "Test Video"
        assert metadata.description is None
    
    def test_caption_track(self):
        track = CaptionTrack(
            language="English",
            language_code="en",
            is_auto_generated=False,
            format="vtt",
        )
        assert track.language == "English"
        assert track.is_auto_generated is False
    
    def test_transcript_segment(self):
        seg = TranscriptSegment(
            start=0.0,
            end=5.0,
            text="Hello",
            confidence=0.95,
        )
        assert seg.start == 0.0
        assert seg.text == "Hello"
    
    def test_transcript_response(self):
        response = TranscriptResponse(
            video_id="test123",
            language="en",
            segments=[],
            full_text="Hello World",
            transcription_method="whisper",
            duration_seconds=5.0,
        )
        assert response.video_id == "test123"
        assert response.full_text == "Hello World"
    
    def test_transcription_options(self):
        options = TranscriptionOptions(
            language="fr",
            model="small",
            word_timestamps=False,
        )
        assert options.language == "fr"
        assert options.word_timestamps is False


@pytest.mark.asyncio
class TestMCPServer:
    """Test MCP server functionality."""
    
    async def test_list_tools(self):
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        
        assert "get_video_metadata" in tool_names
        assert "list_captions" in tool_names
        assert "download_audio" in tool_names
        assert "transcribe_video" in tool_names
        assert "transcribe_audio" in tool_names
    
    async def test_server_metadata(self):
        assert mcp.name == "youtube-transcription"
        assert mcp.version == "1.0.0"


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_video_url_error(self):
        with pytest.raises(InvalidVideoURLError):
            extract_video_id("not-a-valid-url")
    
    def test_youtube_error_hierarchy(self):
        error = InvalidVideoURLError("test")
        assert isinstance(error, YouTubeError)
        assert isinstance(error, Exception)


def test_import_main():
    """Test that main module can be imported."""
    import main
    assert hasattr(main, 'main')
