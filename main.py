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

from pydantic import BaseModel
from typing import List, Optional
import subprocess
import base64
import json
import asyncio
import tempfile
import os
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

# Audio Mixing Models and Functions

class SoundEffect(BaseModel):
    audio_base64: str
    start_time: float
    volume: Optional[float] = 0.3
    description: str
    fade_in_duration: Optional[float] = 1.5  # Fade in duration in seconds
    fade_out_duration: Optional[float] = 1.5  # Fade out duration in seconds
    duration: Optional[float] = None  # Optional: specify sound effect duration

class AudioMixRequest(BaseModel):
    speech_audio_base64: str
    sound_effects: List[SoundEffect]
    total_duration: float

class AudioMixResponse(BaseModel):
    mixed_audio_base64: str
    success: bool
    message: str
    processing_info: dict

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

async def mix_audio_with_ffmpeg(
    speech_audio: bytes, 
    sound_effects: List[dict], 
    total_duration: float
) -> bytes:
    """Mix speech audio with sound effects using FFmpeg with fade in/out effects"""
    
    if not sound_effects:
        return speech_audio
    
    # Create temporary directory for this session
    temp_dir = tempfile.mkdtemp(prefix="audio_mix_")
    temp_files = []
    
    try:
        # Write speech audio to temp file
        speech_path = os.path.join(temp_dir, "speech.mp3")
        with open(speech_path, "wb") as f:
            f.write(speech_audio)
        temp_files.append(speech_path)
        
        # Write sound effects to temp files
        sfx_paths = []
        input_paths = [speech_path]
        
        for i, sfx in enumerate(sound_effects):
            sfx_path = os.path.join(temp_dir, f"sfx_{i}.mp3")
            with open(sfx_path, "wb") as f:
                f.write(sfx["audio"])
            temp_files.append(sfx_path)
            input_paths.append(sfx_path)
            
            sfx_paths.append({
                "path": sfx_path,
                "start_time": sfx["start_time"],
                "volume": sfx["volume"],
                "fade_in_duration": sfx.get("fade_in_duration", 0.5),
                "fade_out_duration": sfx.get("fade_out_duration", 0.5),
                "duration": sfx.get("duration")
            })
        
        # Build FFmpeg filter complex with fade effects
        filter_parts = []
        mix_inputs = ["[0:a]"]  # Main speech audio
        
        for i, sfx in enumerate(sfx_paths):
            input_index = i + 1  # +1 because 0 is speech
            delay_ms = int(sfx["start_time"] * 1000)
            output_label = f"sfx{i}"
            
            # Build the filter chain for this sound effect
            filter_chain = f"[{input_index}:a]"
            
            # Add delay
            filter_chain += f"adelay={delay_ms}|{delay_ms}"
            
            # Add volume adjustment
            filter_chain += f",volume={sfx['volume']}"
            
            # Add fade in effect
            fade_in_duration = sfx["fade_in_duration"]
            if fade_in_duration > 0:
                filter_chain += f",afade=t=in:st={sfx['start_time']:.3f}:d={fade_in_duration:.3f}"
            
            # Add fade out effect
            fade_out_duration = sfx["fade_out_duration"]
            if fade_out_duration > 0:
                # Calculate fade out start time
                if sfx["duration"]:
                    # If duration is specified, fade out before the end
                    fade_out_start = sfx["start_time"] + sfx["duration"] - fade_out_duration
                else:
                    # If no duration specified, fade out near the end of total duration
                    fade_out_start = total_duration - fade_out_duration
                
                if fade_out_start > sfx["start_time"]:  # Only add fade out if it makes sense
                    filter_chain += f",afade=t=out:st={fade_out_start:.3f}:d={fade_out_duration:.3f}"
            
            # Complete the filter for this sound effect
            filter_parts.append(f"{filter_chain}[{output_label}]")
            mix_inputs.append(f"[{output_label}]")
        
        # Final mix command
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
        
        # Execute FFmpeg command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
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
        # Remove temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to remove temp directory {temp_dir}: {e}")

# Updated audio mixing endpoint
@app.post("/mix-audio", response_model=AudioMixResponse)
async def mix_audio_endpoint(request: AudioMixRequest):
    """
    Mix speech audio with sound effects with fade in/out capabilities
    
    This endpoint takes base64-encoded speech audio and sound effects,
    mixes them using FFmpeg with fade effects, and returns the mixed audio as base64.
    """
    try:
        # Check if FFmpeg is available
        if not check_ffmpeg_available():
            raise HTTPException(
                status_code=500, 
                detail="FFmpeg is not available on this system"
            )
        
        # Validate input
        if not request.speech_audio_base64:
            raise HTTPException(status_code=400, detail="Speech audio is required")
        
        if not request.sound_effects:
            # No sound effects, just return the original audio
            return AudioMixResponse(
                mixed_audio_base64=request.speech_audio_base64,
                success=True,
                message="No sound effects to mix, returned original audio",
                processing_info={
                    "sound_effects_count": 0,
                    "total_duration": request.total_duration,
                    "mixing_performed": False
                }
            )
        
        # Decode speech audio
        try:
            speech_audio = base64.b64decode(request.speech_audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid speech audio base64: {e}")
        
        # Prepare sound effects with fade parameters
        sound_effects = []
        for i, sfx in enumerate(request.sound_effects):
            try:
                sfx_audio = base64.b64decode(sfx.audio_base64)
                sound_effects.append({
                    "audio": sfx_audio,
                    "start_time": sfx.start_time,
                    "volume": sfx.volume or 0.3,
                    "description": sfx.description,
                    "fade_in_duration": sfx.fade_in_duration or 0.5,
                    "fade_out_duration": sfx.fade_out_duration or 0.5,
                    "duration": sfx.duration
                })
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid sound effect {i} base64: {e}"
                )
        
        # Perform audio mixing
        mixed_audio = await mix_audio_with_ffmpeg(
            speech_audio, 
            sound_effects, 
            request.total_duration
        )
        
        # Encode result to base64
        mixed_audio_base64 = base64.b64encode(mixed_audio).decode()
        
        return AudioMixResponse(
            mixed_audio_base64=mixed_audio_base64,
            success=True,
            message="Audio mixing with fade effects completed successfully",
            processing_info={
                "sound_effects_count": len(sound_effects),
                "total_duration": request.total_duration,
                "mixing_performed": True,
                "output_size_bytes": len(mixed_audio),
                "fade_effects_applied": True
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Audio mixing failed: {str(e)}"
        )

# Updated simple endpoint with fade support
@app.post("/mix-audio-simple")
async def mix_audio_simple_endpoint(
    speech_audio: UploadFile = File(...),
    sound_effects_data: str = Form(...),
    total_duration: float = Form(...)
):
    """
    Alternative endpoint that accepts file uploads directly with fade support
    sound_effects_data should be JSON string with sound effect metadata including fade parameters
    
    Example sound_effects_data JSON:
    [
        {
            "audio_base64": "base64_encoded_audio_data",
            "start_time": 5.0,
            "volume": 0.4,
            "fade_in_duration": 1.0,
            "fade_out_duration": 2.0,
            "duration": 10.0,
            "description": "Background music"
        }
    ]
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
                    "fade_in_duration": sfx.get("fade_in_duration", 0.5),
                    "fade_out_duration": sfx.get("fade_out_duration", 0.5),
                    "duration": sfx.get("duration")
                })
            except Exception:
                continue  # Skip invalid sound effects
        
        # Perform mixing
        mixed_audio = await mix_audio_with_ffmpeg(speech_content, sound_effects, total_duration)
        
        # Return as audio file response
        return Response(
            content=mixed_audio,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=mixed_audio_with_fades.mp3"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio mixing failed: {str(e)}")
# Audio Mixing Endpoints

@app.post("/mix-audio-simple")
async def mix_audio_simple_endpoint(
    speech_audio: UploadFile = File(...),
    sound_effects_data: str = Form(...),
    total_duration: float = Form(...)
):
    """
    Alternative endpoint that accepts file uploads directly
    sound_effects_data should be JSON string with sound effect metadata
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
        
        # For this simple version, we assume sound effects are provided as base64 in metadata
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
                    "description": sfx.get("description", "")
                })
            except Exception:
                continue  # Skip invalid sound effects
        
        # Perform mixing
        mixed_audio = await mix_audio_with_ffmpeg(speech_content, sound_effects, total_duration)
        
        # Return as audio file response
        return Response(
            content=mixed_audio,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=mixed_audio.mp3"}
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
        
        return {
            "ffmpeg_available": result.returncode == 0,
            "version": version_info,
            "mp3_codec_available": has_mp3,
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