from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, Response
import uvicorn
import tempfile
import os

from parse_pdf import parsePdf, groupDualDialogues, groupSections, sortLines, cleanPage, getTopTrends, stitchSeperateWordsIntoLines, processInitialPages
from utils import cleanScript

from pydantic import BaseModel
from typing import List, Optional
import subprocess
import base64
import json
import asyncio
from pathlib import Path
import shutil

# Single FastAPI app instance
app = FastAPI(
    title="Screenplay PDF to JSON & Audio Mixing Service",
    description="API to convert screenplay PDFs into structured JSON format and mix audio with sound effects"
)

# Root route for Railway health check
@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {"message": "Screenplay PDF & Audio Mixing Service is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    ffmpeg_available = check_ffmpeg_available()
    return {
        "status": "healthy",
        "ffmpeg_available": ffmpeg_available,
        "message": "All services are running",
        "services": {
            "pdf_conversion": True,
            "audio_mixing": ffmpeg_available
        }
    }

# PDF Conversion Endpoints
@app.post("/convert/", response_class=JSONResponse)
async def convert_pdf_to_json(
    pdf_file: UploadFile = File(...),
    page_start: int = None
):
    """
    Convert a screenplay PDF file to JSON format.
    
    - **pdf_file**: The screenplay PDF file to convert
    - **page_start**: Optional page number to start parsing from
    """
    # Validate file type
    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        # Write the uploaded file content to the temporary file
        temp_pdf.write(await pdf_file.read())
        temp_pdf_path = temp_pdf.name
    
    try:
        # Open the temporary PDF file for processing
        with open(temp_pdf_path, 'rb') as script_file:
            # Process the PDF file
            result = convert(script_file, page_start)
        
        # Delete the temporary file after processing
        os.unlink(temp_pdf_path)
        
        # Return the JSON result
        return result
    
    except Exception as e:
        # Delete the temporary file in case of error
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
        
        # Raise an HTTP exception with the error details
        raise HTTPException(status_code=500, detail=f"Failed to convert PDF: {str(e)}")

def convert(script_file, page_start):
    """
    Convert a screenplay PDF file to JSON format.
    
    Args:
        script_file: The PDF file object
        page_start: Optional page number to start parsing from
        
    Returns:
        A list of screenplay elements in JSON format
    """
    # Parse script based on pdfminer.six
    new_script = parsePdf(script_file)["pdf"]

    # Process initial pages
    first_pages_dict = processInitialPages(new_script)
    first_pages = first_pages_dict["firstPages"]
    skip_page = page_start if page_start else first_pages_dict["pageStart"]

    # Remove any useless line (page number, empty line, special symbols)
    new_script = cleanPage(new_script, skip_page)

    # Sort lines by y. If y is the same, then sort by x
    new_script = sortLines(new_script, skip_page)

    # Group dual dialogues into the same segments
    new_script = groupDualDialogues(new_script, skip_page)

    # Stitch words into what's supposed to be part of the same line
    new_script = stitchSeperateWordsIntoLines(new_script, skip_page)
    
    # Get top trends
    top_trends = getTopTrends(new_script)

    # Group into sections based on type
    new_script = groupSections(top_trends, new_script, skip_page, False)

    # Clean the script
    new_script = cleanScript(new_script, False)

    # Add first pages
    new_script = first_pages + new_script
    
    return new_script

# Audio Mixing Models

class SoundEffect(BaseModel):
    audio_base64: str
    start_time: float
    volume: Optional[float] = 0.3
    description: str
    fade_in_duration: Optional[float] = 0.3
    fade_out_duration: Optional[float] = 0.3
    duration: Optional[float] = None

class NarratorTimestamp(BaseModel):
    start_time: float
    end_time: float

class AudioMixRequest(BaseModel):
    speech_audio_base64: str
    sound_effects: List[SoundEffect]
    total_duration: float
    normalize_volume: Optional[bool] = True
    target_lufs: Optional[float] = -16.0
    peak_limit: Optional[float] = -1.0
    narrator_timestamps: Optional[List[NarratorTimestamp]] = []
    background_music_base64: Optional[str] = None
    narrator_bg_volume: Optional[float] = 0.4  # Volume when narrator is speaking
    character_bg_volume: Optional[float] = 0.15  # Volume when characters are speaking

class AudioMixResponse(BaseModel):
    mixed_audio_base64: str
    success: bool
    message: str
    processing_info: dict

# Audio Mixing Utility Functions

def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available in the system"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def safe_cleanup(temp_files: List[str]) -> None:
    """Safely clean up temporary files"""
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Failed to cleanup {file_path}: {e}")

async def normalize_audio_levels(
    input_path: str, 
    output_path: str, 
    target_lufs: float = -16.0,
    peak_limit: float = -1.0
) -> bool:
    """
    Normalize audio levels using FFmpeg loudnorm filter
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output normalized audio file
        target_lufs: Target loudness in LUFS (default: -16.0 for streaming)
        peak_limit: Peak limiter in dB (default: -1.0)
    
    Returns:
        bool: True if normalization successful, False otherwise
    """
    try:
        # First pass: analyze audio to get loudness statistics
        analyze_cmd = [
            "ffmpeg",
            "-i", input_path,
            "-af", f"loudnorm=I={target_lufs}:TP={peak_limit}:LRA=11:print_format=json",
            "-f", "null",
            "-"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *analyze_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"Audio analysis failed: {stderr.decode()}")
            return False
        
        # Extract loudness measurements from stderr (FFmpeg outputs to stderr)
        stderr_text = stderr.decode()
        
        # Look for the JSON output in stderr
        json_start = stderr_text.find('{')
        json_end = stderr_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                loudness_data = json.loads(stderr_text[json_start:json_end])
                
                # Second pass: apply normalization with measured values
                normalize_cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite output
                    "-i", input_path,
                    "-af", (
                        f"loudnorm=I={target_lufs}:TP={peak_limit}:LRA=11:"
                        f"measured_I={loudness_data.get('input_i', target_lufs)}:"
                        f"measured_LRA={loudness_data.get('input_lra', 11)}:"
                        f"measured_TP={loudness_data.get('input_tp', peak_limit)}:"
                        f"measured_thresh={loudness_data.get('input_thresh', -26)}:"
                        f"offset={loudness_data.get('target_offset', 0)}"
                    ),
                    "-c:a", "mp3",
                    "-b:a", "128k",
                    output_path
                ]
                
            except json.JSONDecodeError:
                # Fallback to single-pass normalization
                normalize_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", input_path,
                    "-af", f"loudnorm=I={target_lufs}:TP={peak_limit}:LRA=11",
                    "-c:a", "mp3",
                    "-b:a", "128k",
                    output_path
                ]
        else:
            # Fallback to single-pass normalization
            normalize_cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-af", f"loudnorm=I={target_lufs}:TP={peak_limit}:LRA=11",
                "-c:a", "mp3",
                "-b:a", "128k",
                output_path
            ]
        
        # Execute normalization
        process = await asyncio.create_subprocess_exec(
            *normalize_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"Audio normalization failed: {stderr.decode()}")
            return False
        
        return os.path.exists(output_path)
        
    except Exception as e:
        print(f"Error during audio normalization: {e}")
        return False

async def normalize_single_audio(
    audio_data: bytes,
    target_lufs: float = -16.0,
    peak_limit: float = -1.0
) -> bytes:
    """Normalize a single audio file"""
    temp_dir = tempfile.mkdtemp(prefix="audio_norm_")
    temp_files = []
    
    try:
        # Write input audio
        input_path = os.path.join(temp_dir, "input.mp3")
        with open(input_path, "wb") as f:
            f.write(audio_data)
        temp_files.append(input_path)
        
        # Output path
        output_path = os.path.join(temp_dir, "output.mp3")
        temp_files.append(output_path)
        
        # Normalize
        success = await normalize_audio_levels(input_path, output_path, target_lufs, peak_limit)
        
        if not success:
            # Return original if normalization failed
            return audio_data
        
        # Read normalized audio
        with open(output_path, "rb") as f:
            return f.read()
            
    finally:
        safe_cleanup(temp_files)
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

async def mix_audio_with_ffmpeg(
    speech_audio: bytes, 
    sound_effects: List[dict], 
    total_duration: float,
    normalize_volume: bool = True,
    target_lufs: float = -16.0,
    peak_limit: float = -1.0,
    narrator_timestamps: List[dict] = None,
    background_music: bytes = None,
    narrator_bg_volume: float = 0.1,
    character_bg_volume: float = 0.05
) -> bytes:
    """
    Mix speech audio with sound effects and background music with dynamic volume based on narrator timestamps
    
    Args:
        speech_audio: Main speech audio as bytes
        sound_effects: List of sound effect dictionaries
        total_duration: Total duration of the mixed audio
        normalize_volume: Whether to normalize the final output
        target_lufs: Target loudness for normalization
        peak_limit: Peak limiter for normalization
        narrator_timestamps: List of timestamp dictionaries for narrator sections
        background_music: Background music as bytes (optional)
        narrator_bg_volume: Background music volume when narrator is speaking
        character_bg_volume: Background music volume when characters are speaking
    
    Returns:
        Mixed audio as bytes
    """
    temp_dir = tempfile.mkdtemp(prefix="audio_mix_")
    temp_files = []
    
    try:
        # Write speech audio to temp file
        speech_path = os.path.join(temp_dir, "speech.mp3")
        with open(speech_path, "wb") as f:
            f.write(speech_audio)
        temp_files.append(speech_path)
        
        input_paths = [speech_path]
        filter_parts = []
        mix_inputs = ["[0:a]"]  # Main speech audio
        
        # Handle background music if provided
        if background_music:
            bg_music_path = os.path.join(temp_dir, "bg_music.mp3")
            with open(bg_music_path, "wb") as f:
                f.write(background_music)
            temp_files.append(bg_music_path)
            input_paths.append(bg_music_path)
            
            bg_input_index = len(input_paths) - 1
            
            # Create dynamic volume filter for background music based on narrator timestamps
            volume_filter = f"[{bg_input_index}:a]aloop=loop=-1:size=2e+09,atrim=end={total_duration:.3f}"
            
            if narrator_timestamps:
                # Start with character volume (lower) as default
                volume_filter += f",volume={character_bg_volume}"
                
                # Sort narrator timestamps by start time
                sorted_timestamps = sorted(narrator_timestamps, key=lambda x: x["start_time"])
                
                # Add volume increases for narrator sections
                for timestamp in sorted_timestamps:
                    start_time = timestamp["start_time"]
                    end_time = timestamp["end_time"]
                    # Increase the volume during narrator speech
                    volume_filter += f",volume='if(between(t,{start_time},{end_time}),{narrator_bg_volume}/{character_bg_volume},1)':eval=frame"
                
            else:
                # No narrator timestamps, use constant narrator volume with looping
                volume_filter = f"[{bg_input_index}:a]aloop=loop=-1:size=2e+09,atrim=end={total_duration:.3f},volume={narrator_bg_volume}"
            
            filter_parts.append(f"{volume_filter}[bgmusic]")
            mix_inputs.append("[bgmusic]")
            
            print(f"Background music filter: {volume_filter}")
        
        # Handle sound effects
        for i, sfx in enumerate(sound_effects):
            sfx_path = os.path.join(temp_dir, f"sfx_{i}.mp3")
            with open(sfx_path, "wb") as f:
                f.write(sfx["audio"])
            temp_files.append(sfx_path)
            input_paths.append(sfx_path)
            
            input_index = len(input_paths) - 1
            delay_ms = int(sfx["start_time"] * 1000)
            output_label = f"sfx{i}"
            
            # Build the filter chain for this sound effect
            filter_chain = f"[{input_index}:a]"
            
            # Add delay
            filter_chain += f"adelay={delay_ms}|{delay_ms}"
            
            # Add volume adjustment
            filter_chain += f",volume={sfx['volume']}"
            
            # Add fade effects if specified
            fade_in_duration = sfx.get("fade_in_duration", 0.3)
            if fade_in_duration > 0:
                filter_chain += f",afade=t=in:st={sfx['start_time']:.3f}:d={fade_in_duration:.3f}"
            
            fade_out_duration = sfx.get("fade_out_duration", 0.3)
            if fade_out_duration > 0:
                if sfx.get("duration"):
                    fade_out_start = sfx["start_time"] + sfx["duration"] - fade_out_duration
                else:
                    fade_out_start = total_duration - fade_out_duration
                
                if fade_out_start > sfx["start_time"]:
                    filter_chain += f",afade=t=out:st={fade_out_start:.3f}:d={fade_out_duration:.3f}"
            
            filter_parts.append(f"{filter_chain}[{output_label}]")
            mix_inputs.append(f"[{output_label}]")
        
        # Final mix command with optional normalization
        if normalize_volume:
            filter_parts.append(
                f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest,"
                f"loudnorm=I={target_lufs}:TP={peak_limit}:LRA=11"
            )
        else:
            filter_parts.append(
                f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest"
            )
        
        filter_complex = "; ".join(filter_parts)
        
        # Output file
        output_path = os.path.join(temp_dir, "output.mp3")
        temp_files.append(output_path)
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
        ]
        
        # Add input files
        for input_path in input_paths:
            cmd.extend(["-i", input_path])
        
        # Add filter complex and output options
        cmd.extend([
            "-filter_complex", filter_complex,
            "-c:a", "mp3",
            "-b:a", "128k",
            "-t", str(total_duration),
            output_path
        ])
        
        print(f"FFmpeg command: {' '.join(cmd)}")
        print(f"Filter complex: {filter_complex}")
        
        # Execute FFmpeg command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
            print(f"FFmpeg stderr: {error_msg}")
            raise RuntimeError(f"FFmpeg failed: {error_msg}")
        
        # Check if output file was created
        if not os.path.exists(output_path):
            raise RuntimeError("Output file was not created")
        
        # Read the mixed audio
        with open(output_path, "rb") as f:
            mixed_audio = f.read()
        
        return mixed_audio
        
    finally:
        # Clean up temp files
        safe_cleanup(temp_files)
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to remove temp directory {temp_dir}: {e}")

# Audio Mixing Endpoints

@app.post("/mix-audio", response_model=AudioMixResponse)
async def mix_audio_endpoint(request: AudioMixRequest):
    """
    Mix speech audio with sound effects and background music with narrator-based volume control
    
    This endpoint supports:
    - Dynamic background music volume based on narrator timestamps
    - Sound effects with fade in/out capabilities
    - Volume normalization
    - Base64 encoded audio input/output
    """
    try:
        print("Narrator timestamps:", request.narrator_timestamps)
        
        # Check if FFmpeg is available
        if not check_ffmpeg_available():
            raise HTTPException(
                status_code=500, 
                detail="FFmpeg is not available on this system"
            )
        
        # Validate input
        if not request.speech_audio_base64:
            raise HTTPException(status_code=400, detail="Speech audio is required")
        
        # Handle case with no sound effects but possible background music
        if not request.sound_effects and not request.background_music_base64:
            # No sound effects or background music, just normalize if requested
            if request.normalize_volume:
                try:
                    speech_audio = base64.b64decode(request.speech_audio_base64)
                    normalized_audio = await normalize_single_audio(
                        speech_audio, 
                        request.target_lufs or -16.0,
                        request.peak_limit or -1.0
                    )
                    normalized_base64 = base64.b64encode(normalized_audio).decode()
                    
                    return AudioMixResponse(
                        mixed_audio_base64=normalized_base64,
                        success=True,
                        message="No sound effects or background music to mix, returned normalized speech audio",
                        processing_info={
                            "sound_effects_count": 0,
                            "total_duration": request.total_duration,
                            "mixing_performed": False,
                            "normalization_performed": True,
                            "background_music_added": False,
                            "narrator_timestamps_count": 0,
                            "target_lufs": request.target_lufs or -16.0
                        }
                    )
                except Exception as e:
                    return AudioMixResponse(
                        mixed_audio_base64=request.speech_audio_base64,
                        success=True,
                        message=f"No sound effects to mix, normalization failed: {e}",
                        processing_info={
                            "sound_effects_count": 0,
                            "total_duration": request.total_duration,
                            "mixing_performed": False,
                            "normalization_performed": False,
                            "background_music_added": False,
                            "narrator_timestamps_count": 0,
                            "normalization_error": str(e)
                        }
                    )
            else:
                return AudioMixResponse(
                    mixed_audio_base64=request.speech_audio_base64,
                    success=True,
                    message="No sound effects or background music to mix, returned original audio",
                    processing_info={
                        "sound_effects_count": 0,
                        "total_duration": request.total_duration,
                        "mixing_performed": False,
                        "normalization_performed": False,
                        "background_music_added": False,
                        "narrator_timestamps_count": 0
                    }
                )
        
        # Decode speech audio
        try:
            speech_audio = base64.b64decode(request.speech_audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid speech audio base64: {e}")
        
        # Decode background music if provided
        background_music = None
        if request.background_music_base64:
            try:
                background_music = base64.b64decode(request.background_music_base64)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid background music base64: {e}")
        
        # Prepare sound effects
        sound_effects = []
        for i, sfx in enumerate(request.sound_effects):
            try:
                sfx_audio = base64.b64decode(sfx.audio_base64)
                sound_effects.append({
                    "audio": sfx_audio,
                    "start_time": sfx.start_time,
                    "volume": sfx.volume or 0.3,
                    "description": sfx.description,
                    "fade_in_duration": sfx.fade_in_duration or 0.3,
                    "fade_out_duration": sfx.fade_out_duration or 0.3,
                    "duration": sfx.duration
                })
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid sound effect {i} base64: {e}"
                )
        
        # Convert narrator timestamps to dict format
        narrator_timestamps_dict = []
        if request.narrator_timestamps:
            narrator_timestamps_dict = [
                {
                    "start_time": ts.start_time,
                    "end_time": ts.end_time
                }
                for ts in request.narrator_timestamps
            ]
        
        # Perform audio mixing
        mixed_audio = await mix_audio_with_ffmpeg(
            speech_audio, 
            sound_effects, 
            request.total_duration,
            request.normalize_volume or True,
            request.target_lufs or -16.0,
            request.peak_limit or -1.0,
            narrator_timestamps_dict,
            background_music,
            request.narrator_bg_volume or 0.4,
            request.character_bg_volume or 0.15
        )
        
        # Encode result to base64
        mixed_audio_base64 = base64.b64encode(mixed_audio).decode()
        
        return AudioMixResponse(
            mixed_audio_base64=mixed_audio_base64,
            success=True,
            message="Audio mixing with background music and narrator-based volume control completed successfully",
            processing_info={
                "sound_effects_count": len(sound_effects),
                "total_duration": request.total_duration,
                "mixing_performed": True,
                "normalization_performed": request.normalize_volume or True,
                "background_music_added": background_music is not None,
                "narrator_timestamps_count": len(narrator_timestamps_dict),
                "target_lufs": request.target_lufs or -16.0,
                "peak_limit": request.peak_limit or -1.0,
                "output_size_bytes": len(mixed_audio)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Audio mixing failed: {str(e)}"
        )

@app.post("/mix-audio-simple")
async def mix_audio_simple_endpoint(
    speech_audio: UploadFile = File(...),
    sound_effects_data: str = Form(...),
    total_duration: float = Form(...),
    normalize_volume: bool = Form(True),
    target_lufs: float = Form(-16.0),
    peak_limit: float = Form(-1.0)
):
    """
    Alternative endpoint that accepts file uploads directly with fade and normalization support
    
    Args:
        speech_audio: Main speech audio file
        sound_effects_data: JSON string with sound effect metadata including fade parameters
        total_duration: Total duration of the mixed audio
        normalize_volume: Whether to apply volume normalization (default: True)
        target_lufs: Target loudness in LUFS (default: -16.0)
        peak_limit: Peak limiter in dB (default: -1.0)
    """
    try:
        # Check FFmpeg availability
        if not check_ffmpeg_available():
            raise HTTPException(
                status_code=500, 
                detail="FFmpeg is not available on this system"
            )
        
        # Read speech audio
        speech_content = await speech_audio.read()
        
        # Parse sound effects metadata
        try:
            sfx_metadata = json.loads(sound_effects_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid sound_effects_data JSON")
        
        # Process sound effects with fade parameters
        sound_effects = []
        for sfx in sfx_metadata:
            if "audio_base64" not in sfx:
                continue
            try:
                sfx_audio = base64.b64decode(sfx["audio_base64"])
                sound_effects.append({
                    "audio": sfx_audio,
                    "start_time": sfx.get("start_time", 0),
                    "volume": sfx.get("volume", 0.3),
                    "description": sfx.get("description", ""),
                    "fade_in_duration": sfx.get("fade_in_duration", 0.3),
                    "fade_out_duration": sfx.get("fade_out_duration", 0.3),
                    "duration": sfx.get("duration")
                })
            except Exception:
                continue  # Skip invalid sound effects
        
        # Perform mixing with normalization
        mixed_audio = await mix_audio_with_ffmpeg(
            speech_content, 
            sound_effects, 
            total_duration,
            normalize_volume,
            target_lufs,
            peak_limit
        )
        
        # Return as audio file response
        filename = f"mixed_audio_normalized_{target_lufs}LUFS.mp3" if normalize_volume else "mixed_audio.mp3"
        return Response(
            content=mixed_audio,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio mixing failed: {str(e)}")

@app.get("/debug-ffmpeg")
async def debug_ffmpeg():
    """Debug endpoint to check FFmpeg installation and capabilities"""
    try:
        # Check if FFmpeg is available
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        version_info = result.stdout.split('\n')[0] if result.stdout else "Unknown"
        
        # Check available codecs
        codecs_result = subprocess.run(
            ["ffmpeg", "-codecs"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        has_mp3 = "mp3" in codecs_result.stdout.lower() if codecs_result.stdout else False
        
        # Check for loudnorm filter
        filters_result = subprocess.run(
            ["ffmpeg", "-filters"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        has_loudnorm = "loudnorm" in filters_result.stdout.lower() if filters_result.stdout else False
        
        return {
            "ffmpeg_available": result.returncode == 0,
            "version": version_info,
            "mp3_codec_available": has_mp3,
            "loudnorm_filter_available": has_loudnorm,
            "normalization_supported": has_loudnorm,
            "return_code": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            "ffmpeg_available": False,
            "error": "FFmpeg command timed out"
        }
    except FileNotFoundError:
        return {
            "ffmpeg_available": False,
            "error": "FFmpeg not found in system PATH"
        }
    except Exception as e:
        return {
            "ffmpeg_available": False,
            "error": str(e)
        }

# Fixed startup for Railway deployment
if __name__ == "__main__":
    # Use PORT environment variable from Railway, fallback to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)