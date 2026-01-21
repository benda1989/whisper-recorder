from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Request, Response, Depends, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import asyncio
import numpy as np
import threading
import time
import tempfile
import soundfile as sf
from typing import Dict, List
from dataclasses import dataclass, field
import json
import socket
import datetime
from faster_whisper import WhisperModel
from  config import *
import secrets
whisper_model = WhisperModel("large-v3", device="cuda")
RECORDINGS = "records"
os.makedirs(RECORDINGS, exist_ok=True)

# Add global stop flag
stop_flag = False

# Define lifespan function first
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Execute on startup
    # Load recording schedules
    load_schedules()

    # Start TCP server
    threading.Thread(target=start_tcp_server, daemon=True) .start()
    
    # Start device timeout cleanup task
    threading.Thread(target=clean_devices, daemon=True) .start()
    
    yield
    
    # Execute on shutdown
    global stop_flag
    stop_flag = True
    # Save recording schedules
    save_schedules()

# Create FastAPI application
app = FastAPI(title="Multi-device Voice Recording Service", version="1.5.0", lifespan=lifespan)


sessions = {}  # Store session information

# Request model
class LoginRequest(BaseModel):
    password: str

SESSION_SECRET = secrets.token_hex(32)
# Generate session token
def generate_session_token():
    return secrets.token_urlsafe(32)

# Verify session
def verify_session(request: Request):
    session_token = request.cookies.get("session_token")
    if not session_token or session_token not in sessions:
        return False
    return True

# Authentication required dependency
def require_auth(request: Request):
    if not verify_session(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login required for access"
        )
    return True


@dataclass
class DeviceSession:
    device_id: str
    # Connection information
    connection_type: str  # "tcp" or "websocket" 
    connection_obj: object
    
    # Audio session data
    raw_audio: List[int] = field(default_factory=list)
    last_chunk_time: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
# Global variables - unified device session management
device_sessions: Dict[str, DeviceSession] = {}
sessions_lock = threading.Lock()


def save_device_audio(device_id: str, raw_audio: List[int]):
    """Simplified audio saving - directly save 3-minute audio files"""
    try:
        if len(raw_audio) < 1600:  # Ignore if less than 0.1s
            return
            
        # Convert audio data
        audio_array = np.array(raw_audio, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Create date folder
        date_dir = os.path.join(RECORDINGS, device_id, time.strftime("%Y%m%d"))
        os.makedirs(date_dir, exist_ok=True)
        
        # Save audio file directly, using timestamp naming
        filepath = os.path.join(date_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.wav")
        sf.write(filepath, audio_float, samplerate=16000, subtype='PCM_16')
                
        # Asynchronous transcription (non-blocking save)
        threading.Thread(
            target=transcribe_audio_file, 
            args=(device_id, filepath, date_dir), 
            daemon=True
        ).start()
        
    except Exception as e:
        logger.error(f"❌ Device {device_id} audio save failed: {str(e)}")

def transcribe_audio_file(device_id: str, filepath: str, date_dir: str):
    """Asynchronously transcribe audio file"""
    try:
        timestamp = time.strftime('%H:%M:%S')
        # Read audio file for transcription
        segments, _ = whisper_model.transcribe(filepath, language="zh",beam_size=5, vad_filter=True)
        text = ''.join(segment.text for segment in segments).strip()
        
        if text:
            line = f"[{timestamp}] {text}"
            # Save to hourly file
            txt_file = os.path.join(date_dir, f"{time.strftime('%Y-%m-%d-%H')}.txt")
            with open(txt_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            logger.debug(f"📝 Device {device_id} transcription completed: {len(text)} characters")
        else:
            os.remove(filepath)
    except Exception as e:
        logger.error(f"❌ Device {device_id} transcription failed: {str(e)}")


# Recording schedule configuration
@dataclass
class RecordingSchedule:
    """Simplified recording schedule"""
    start_at: int  # Start timestamp (unix timestamp)
    duration: int  # Recording duration (seconds)
    stop_at:  int
    created_time: float = field(default_factory=time.time)
    
    def is_active(self) -> bool:
        """Check if currently within recording time""" 
        return self.start_at <= time.time() <= self.stop_at
    
    def __post_init__(self):
        """Validate parameters"""
        if self.start_at < 0:
            raise ValueError("start_at must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
    
    
# Recording schedule management
recording_schedules: Dict[str, List[RecordingSchedule]] = {}
schedules_lock = threading.Lock() 

def load_schedules():
    """Load recording schedules from JSON file"""
    global recording_schedules
    try:
        if os.path.exists(SCHEDULES_FILE):
            with open(SCHEDULES_FILE, 'r', encoding='utf-8') as f:
                datas = json.load(f)
                recording_schedules = {}
                for device_id, data in datas.items():
                    recording_schedules[device_id] = [ RecordingSchedule(**i) for i in data]
    except Exception as e:
        logger.error(f"❌ Failed to load recording schedules: {e}")
        recording_schedules = {}

def save_schedules():
    """Save recording schedules to JSON file"""
    try:
        # Convert to serializable format
        data = {}
        today_midnight = datetime.datetime.combine(datetime.date.today(), datetime.time.min).timestamp()
        for device_id, schedules in recording_schedules.items():
            data[device_id] = []
            
            for schedule in schedules:
                schedule_dict = {
                    "start_at": schedule.start_at,
                    "stop_at": schedule.stop_at,
                    "duration": schedule.duration,
                    "created_time": schedule.created_time
                }
                # Only keep schedules from today midnight onwards
                if schedule.start_at >= today_midnight:
                    data[device_id].append(schedule_dict)
        
        with open(SCHEDULES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("💾 Recording schedules saved")
    except Exception as e:
        logger.error(f"❌ Failed to save recording schedules: {e}")


def clean_connection(old: DeviceSession, new_obj: object):
    """Safely clean up old connections"""
    if old.connection_type == "tcp" and old.connection_obj != new_obj:
        try:
            if hasattr(old.connection_obj, 'fileno') and old.connection_obj.fileno() != -1:
                old.connection_obj.close()
            logger.info(f"Cleaned old TCP connection for device {old.device_id}")
        except (OSError, AttributeError):
            pass
        except Exception as e:
            logger.warning(f"Failed to clean connection for device {old.device_id}: {e}")
    elif old.connection_type == "websocket" and old.connection_obj != new_obj:
        try:
            logger.info(f"WebSocket connection {old.device_id} will auto cleanup")
        except Exception as e:
            logger.warning(f"WebSocket {old.device_id} cleanup warning: {e}")

def register_client(device_id: str, connection_type: str, connection_obj: object ) -> Dict:
    ""
    current_time = time.time()
    duration = 0
    device_schedules = recording_schedules.get(device_id, []) 
    if not device_schedules:
        os.makedirs(os.path.join(RECORDINGS, device_id), exist_ok=True)
        return 0
    for schedule in device_schedules:
        if schedule.is_active():
            duration= int(schedule.stop_at - current_time)
            break
    if duration <= 0:
        return 0
    with sessions_lock:
            # Check if device exists and handle connection type
        if device_id in device_sessions:
            existing_session = device_sessions[device_id]
            if existing_session.connection_type == connection_type:
                # Same connection type, clean old connection
                clean_connection(existing_session, connection_obj)
                logger.warning(f"Device {device_id} duplicate {connection_type} connection, cleaned old connection")
            else:
                # Different connection type, log warning but allow coexistence
                logger.warning(f"Device {device_id} multiple connection types: existing {existing_session.connection_type}, new {connection_type}")
        
        # Create or update device session
        device_sessions[device_id] = DeviceSession(
            device_id=device_id,
            connection_type=connection_type,
            connection_obj=connection_obj,
            start_time=current_time,
        )
        logger.info(f"Device Connect: {device_id} via {connection_type}")
    return duration

def unregister_client(device_id: str) -> bool:
    """Unregister client connection"""
    with sessions_lock:
        if device_id in device_sessions:
            try:
                session = device_sessions[device_id]
                if session.connection_type == "tcp" and hasattr(session.connection_obj, 'fileno'):
                    try:
                        if session.connection_obj.fileno() != -1:
                            session.connection_obj.close()
                    except (OSError, AttributeError):
                        pass
                del device_sessions[device_id]
                logger.info(f"Device discount: {device_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to unregister device {device_id}: {e}")
                return False
        return False

def clean_devices():
    """Background task to periodically clean inactive devices"""
    while not stop_flag:
        try:
            current_time = time.time()
            inactive_devices = []
            with sessions_lock:
                for device_id, session in device_sessions.items():
                    if current_time - session.last_chunk_time > DEVICE_TIMEOUT:
                        inactive_devices.append(device_id)
            for device_id in inactive_devices:
                logger.warning(f"🕒 Device timeout, auto cleanup: {device_id}")
                unregister_client(device_id)
        except Exception as e:
            logger.error(f"❌ Device cleanup task error: {str(e)}")
        finally:
            time.sleep(DEVICE_TIMEOUT)

def check_device_status(session: str,device_id: str) -> dict:
    """Unified device status check function"""
    if session:
        return {
                "device_id": device_id,
                "is_online": time.time() - session.last_chunk_time < DEVICE_TIMEOUT,
                "last_activity": session.last_chunk_time,
                "start_time": session.start_time,
                "device_name": DeviceName.get(device_id,  device_id),
                }
    return {
            "device_id": device_id,
            "device_name": DeviceName.get(device_id,  device_id),
            "is_online": False,
            "last_activity": 0,
            "start_time": 0,
            }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            segments, info = whisper_model.transcribe(temp_file.name, beam_size=5,vad_filter=True)
            os.unlink(temp_file.name)
            return {
                "language": info.language,
                "segments":  segments#["[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text) for segment in segments]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/devices")
async def get_active_devices(auth: bool = Depends(require_auth)):
    """Get device list (including online and offline devices)"""
    devices_info = []
    # Get active device information
    for  device_id, session in device_sessions.items():
        devices_info.append(check_device_status(session, device_id))
    for device_id in os.listdir(RECORDINGS):
        if device_id not in device_sessions.keys():
            devices_info.append(check_device_status(None, device_id) )
    return {
        "devices": devices_info,
        "total": len(device_sessions.keys())
    }

@app.get("/devices/{device_id}/live_status")
async def get_live_status(device_id: str, auth: bool = Depends(require_auth)):
    """Get device real-time status"""
    return check_device_status(device_sessions.get(device_id), device_id)


@app.get("/devices/{device_id}/recordings")
async def get_device_recordings(device_id: str, auth: bool = Depends(require_auth)):
    """Get recording file list for specified device (organized by date folders)"""
    device_dir = os.path.join(RECORDINGS, device_id)
    
    if not os.path.exists(device_dir):
        raise HTTPException(status_code=404, detail=f"Device {device_id} does not exist")
    
    recordings_by_date = {}
    # Traverse date folders under device directory
    for date_folder in os.listdir(device_dir):
        date_path = os.path.join(device_dir, date_folder)
        if not os.path.isdir(date_path):
            continue
        
        recordings = []
        for file in os.listdir(date_path):
            if file.endswith('.wav'):
                filepath = os.path.join(date_path, file)
                stat = os.stat(filepath)
                recordings.append({
                    "filename": file,
                    "size": stat.st_size,
                    "created_time": stat.st_ctime,
                    "date_folder": date_folder,
                    "relative_path": f"{date_folder}/{file}"
                })
        
        if recordings:
            recordings_by_date[date_folder] = sorted(recordings, key=lambda x: x["created_time"], reverse=True)
    
    return {
        "device_id": device_id,
        "recordings_by_date": recordings_by_date,
        "total_dates": len(recordings_by_date)
    }

@app.get("/devices/{device_id}/playlist/{date}")
async def get_device_playlist(device_id: str, date: str, auth: bool = Depends(require_auth)):
    """Get audio playlist for specified device and date"""
    date_dir = os.path.join(RECORDINGS, device_id, date)
    
    if not os.path.exists(date_dir):
        raise HTTPException(status_code=404, detail=f"Date {date} for device {device_id} does not exist")
    
    audio_files = []
    for file in os.listdir(date_dir):
        if file.endswith('.wav'):
            filepath = os.path.join(date_dir, file)
            stat = os.stat(filepath)
            
            # Read audio information
            try:
                audio_data, samplerate = sf.read(filepath)
                duration = len(audio_data) / samplerate
            except:
                duration = 0
            
            audio_files.append({
                "filename": file,
                "url": f"/audio/{device_id}/{date}/{file}",
                "size": stat.st_size,
                "duration": round(duration, 2),
                "created_time": stat.st_ctime,
            })
    
    # Sort by creation time
    audio_files = sorted(audio_files, key=lambda x: x["created_time"])
    
    return {
        "device_id": device_id,
        "date": date,
        "total_files": len(audio_files),
        "playlist": audio_files,
        "total_duration": round(sum(f["duration"] for f in audio_files), 2)
    }

@app.get("/audio/{device_id}/{date}/{filename}")
async def get_audio_file(device_id: str, date: str, filename: str, auth: bool = Depends(require_auth)):
    """Get audio file"""
    filepath = os.path.join(RECORDINGS, device_id, date, filename)
    
    if not os.path.exists(filepath) or not filename.endswith('.wav'):
        raise HTTPException(status_code=404, detail="Audio file does not exist")
    
    return FileResponse(
        filepath,
        media_type="audio/wav",
        filename=filename,
        headers={"Accept-Ranges": "bytes"}
    )

@app.post("/login")
async def login(request: LoginRequest, response: Response):
    """User login verification"""
    if request.password == SYSTEM_PASSWORD:
        # Create session
        session_token = generate_session_token()
        sessions[session_token] = {
            "created_at": time.time(),
            "last_activity": time.time()
        }
        
        # Set session cookie
        response.set_cookie(
            "session_token", 
            session_token, 
            max_age=86400,  # 24 hours
            httponly=True,
            secure=False  # Should be set to True in production environment
        )
        
        return {"success": True, "message": "Login successful"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password"
        )

@app.get("/check-auth")
async def check_auth(request: Request):
    """Check authentication status"""
    authenticated = verify_session(request)
    return {"authenticated": authenticated}

@app.post("/logout")
async def logout(request: Request, response: Response):
    """User logout"""
    session_token = request.cookies.get("session_token")
    if session_token and session_token in sessions:
        del sessions[session_token]
    
    response.delete_cookie("session_token")
    return {"success": True, "message": "Logged out successfully"}

@app.get("/login", response_class=HTMLResponse)
async def get_login_page():
    """Get login page"""
    with open("login.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/", response_class=HTMLResponse)
async def get_playlist_page(request: Request):
    """Get audio playlist page (authentication required)"""
    if not verify_session(request):
        return RedirectResponse(url="/login")
    
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Store real-time listening connections
live_listeners: Dict[str, List[WebSocket]] = {}
live_listeners_lock = threading.Lock()

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """WebSocket audio transmission endpoint (optimized non-blocking version)"""
    global stop_flag
    await websocket.accept()
    device_id = None
    
    try:       
        while not stop_flag:
            # Receive message - could be text (device ID) or binary (audio data)
            try:
                # Add timeout to avoid permanent blocking
                message = await asyncio.wait_for(websocket.receive(), timeout=5.0)
                # logger.info(f"🔍 WebSocket received message: {message}")
                
                if "text" in message:
                    # Handle text message (device ID registration)
                    try:
                        data = json.loads(message["text"])
                        # logger.info(f"🔍 JSON parsing successful: {data}")
                        if data.get("type") == "device_id":
                            device_id = data.get("id")
                            recording_duration = register_client(device_id, "websocket", websocket )
                            try:
                                await websocket.send_text(recording_duration)
                                logger.info(f"✅ WebSocket device registration successful: {device_id}")
                            except Exception as e:
                                logger.error(f"❌ WebSocket failed to send recording configuration: {e}")
                        else:
                            logger.warning(f"❌ Unknown message type: {data.get('type')}")
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ WebSocket received invalid JSON message: {e}")
                        
                elif "bytes" in message:
                    if not device_id:
                        logger.warning("❌ Received audio data but device not registered")
                        continue
                    raw_data = message["bytes"]
                    if raw_data:
                        try:
                            # Use asyncio.create_task to prevent audio processing from blocking WebSocket reception
                            asyncio.create_task(
                                process_audio_async(device_id, raw_data)
                            )
                        except Exception as e:
                            logger.error(f"❌ Failed to process WebSocket audio data: {str(e)}")
                            
            except asyncio.TimeoutError:
                # Timeout is normal, continue loop
                continue
            except WebSocketDisconnect:
                logger.info(f"📱 WebSocket device disconnected: {device_id}")
                break
                
    except Exception as e:
        logger.error(f"❌ WebSocket connection error: {str(e)}")
    finally:
        if device_id:
            unregister_client(device_id)

def process_audio(device_id: str, raw_data: bytes):
    """Unified audio data processing function"""
    try:
        session=device_sessions.get(device_id)
        if not session:
            logger.warning(f"⚠️ Device {device_id} session does not exist, skipping audio processing")
            return None
        current_time = time.time()
        with sessions_lock:
            session.last_chunk_time = current_time
            session.raw_audio.extend(np.frombuffer(raw_data, dtype=np.int16))
            
            # Check if saving is needed (quick check)
            if current_time - session.start_time >= 180.0 and len(session.raw_audio) > 0:
                audio_to_save = session.raw_audio.copy()
                session.raw_audio.clear()
                session.start_time = current_time
                return audio_to_save
            
    except Exception as e:
        logger.error(f"❌ Audio processing failed {device_id}: {str(e)}")

    return None

async def process_audio_async(device_id: str, raw_data: bytes):
    # Forward audio data to real-time listeners (performed outside lock)
    await audio_to_listeners(device_id, raw_data)
    """Asynchronous audio data processing wrapper"""
    audio_to_save = process_audio(device_id, raw_data)
    if audio_to_save:
        # logger.info(f"💾 Device {device_id} 3-minute file save")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_device_audio, device_id, audio_to_save)

# Forward audio data to real-time listeners
async def audio_to_listeners(device_id: str, raw_data: bytes):
    """Forward audio data to real-time listening WebSocket connections (optimized version)"""
    if device_id not in live_listeners:
        return
    
    try:
        int16_data = np.frombuffer(raw_data, dtype=np.int16)
    
        # Optimization: use memory operations instead of temporary files
        import io
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, int16_data, 16000, format='WAV')
        audio_wav_data = audio_buffer.getvalue()
        audio_buffer.close()
        
        # Forward to all listeners
        disconnected_listeners = []
        with live_listeners_lock:
            listeners = live_listeners.get(device_id, []).copy()
        
        for listener_ws in listeners:
            try:
                await listener_ws.send_bytes(audio_wav_data)
            except Exception as e:
                logger.warning(f"Failed to forward audio data: {e}")
                disconnected_listeners.append(listener_ws)
        
        # Clean up disconnected connections
        if disconnected_listeners:
            with live_listeners_lock:
                if device_id in live_listeners:
                    for ws in disconnected_listeners:
                        if ws in live_listeners[device_id]:
                            live_listeners[device_id].remove(ws)
                    if not live_listeners[device_id]:
                        del live_listeners[device_id]
                        
    except Exception as e:
        logger.error(f"Failed to process audio data forwarding: {e}")

@app.get("/devices/{device_id}/daily_transcript/{date}")
async def get_daily_transcript(device_id: str, date: str, auth: bool = Depends(require_auth)):
    """Get complete transcript text for specified device and date"""
    date_dir = os.path.join(RECORDINGS, device_id, date)
    
    if not os.path.exists(date_dir):
        raise HTTPException(status_code=404, detail=f"Date {date} for device {device_id} does not exist")
    
    # Find all txt files
    txt_files = []
    for file in os.listdir(date_dir):
        if file.endswith('.txt'):
            filepath = os.path.join(date_dir, file)
            txt_files.append((filepath, os.path.getctime(filepath)))
    
    if not txt_files:
        return {
            "device_id": device_id,
            "date": date,
            "transcript": "",
            "total_lines": 0,
            "message": "No transcript text for this date"
        }
    
    # Sort by creation time
    txt_files.sort(key=lambda x: x[1])
    
    # Concatenate all text content
    all_text = []
    total_lines = 0
    
    for filepath, _ in txt_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                all_text.extend(lines)
                total_lines += len(lines)
        except Exception as e:
            logger.warning(f"⚠️ Failed to read text file: {filepath}, {e}")
    
    # Concatenate into complete text
    full_transcript = "".join(all_text).strip()
    
    return {
        "device_id": device_id,
        "date": date,
        "transcript": full_transcript,
        "total_lines": total_lines,
        "files_count": len(txt_files)
    }

# Simplified recording schedule management API
@app.get("/schedules/{device_id}")
async def get_device_schedules(device_id: str, auth: bool = Depends(require_auth)):
    """Get recording schedule for specified device"""
    with schedules_lock:
        device_schedules = recording_schedules.get(device_id, [])
        schedules_list = []
        for schedule in device_schedules:
            schedule_dict = {
                "start_at": schedule.start_at,
                "duration": schedule.duration,
                "created_time": schedule.created_time,
                "is_active": schedule.is_active()
            }
            schedules_list.append(schedule_dict)
        
        return {
            "device_id": device_id,
            "schedules": schedules_list
        }

@app.post("/schedules/{device_id}")
async def add_schedule(device_id: str, schedule_data: dict, auth: bool = Depends(require_auth)):
    """Add recording schedule"""
    try:
        schedule = RecordingSchedule(
            start_at=schedule_data["start_at"],
            duration=schedule_data["duration"],
            stop_at=schedule_data["start_at"]+schedule_data["duration"],
            created_time=time.time(),
        )
        
        with schedules_lock:
            if device_id not in recording_schedules:
                recording_schedules[device_id] = []
            recording_schedules[device_id].append(schedule)
            save_schedules()
            load_schedules()
        logger.info(f"Add Schedular: {device_id}")
        return {"success": True, "message": "Recording schedule added successfully"}
        
    except Exception as e:
        logger.error(f"❌ Failed to add recording schedule: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to add recording schedule: {str(e)}")

def handle_tcp_client(client_socket, client_address):
    """Handle single TCP client connection"""
    device_id = None
    buffer = b""
    
    try:
        while not stop_flag:
            try:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    # logger.info(f"📱 TCP client actively disconnected: {client_address}")
                    break
                
                buffer += data
                # If device ID is not yet available, look for device ID first
                if device_id is None:
                    # Look for simplified device ID message format: device_id:xxxxx
                    try:
                        buffer_str = buffer.decode('utf-8', errors='ignore')
                        if 'device_id:' in buffer_str:
                            # Found device ID message
                            device_id_start = buffer_str.find('device_id:')
                            if device_id_start != -1:
                                # Extract device ID (to newline or buffer end)
                                id_start = device_id_start + len('device_id:')
                                line_end = buffer_str.find('\n', id_start)
                                if line_end == -1:
                                    line_end = len(buffer_str)
                                
                                device_id = buffer_str[id_start:line_end].strip()
                                if device_id:                                    
                                    # Register TCP client and get recording duration
                                    recording_duration = register_client(device_id, "tcp", client_socket)
                                    # Send recording duration response to client (simple numeric format)
                                    try:
                                        response = f"{recording_duration}\n"
                                        client_socket.send(response.encode('utf-8'))
                                        logger.info(f"IP：{client_address} ID: {device_id} Duration: {recording_duration} S")
                                    except socket.error as e:
                                        logger.error(f"Send error: {e}")
                                        break  # Socket error, exit loop
                                    except Exception as e:
                                        logger.error(f"Send error: {e}")
                                    
                                    # Remove processed device ID message
                                    processed_bytes = device_id_start + len(f'device_id:{device_id}')
                                    if line_end < len(buffer_str):
                                        processed_bytes += 1  # Include newline character
                                    buffer = buffer[processed_bytes:]
                    except UnicodeDecodeError:
                        # If cannot decode as UTF-8, might be audio data
                        pass
                    
                    # If buffer is too large but device ID not found yet, clean up part of it
                    if len(buffer) > 2048:
                        buffer = buffer[-1024:]
                
                # If device ID exists, process audio data
                if device_id and len(buffer) >= 2048:
                    # Extract 2048 bytes of audio data
                    audio_data = buffer[:2048]
                    buffer = buffer[2048:]
                    
                    try:
                        # Use thread-safe method to forward audio data
                        threading.Thread(
                            target=lambda: asyncio.run(audio_to_listeners(device_id, audio_data)),
                            daemon=True
                        ).start()
                        # Use unified audio processing function
                        audio_to_save = process_audio(device_id, audio_data)
                        if audio_to_save:
                            # logger.info(f"💾 TCP device {device_id} 3-minute file save")
                            threading.Thread(
                                target=save_device_audio,
                                args=(device_id, audio_to_save),
                                daemon=True
                            ).start()
                        # TCP audio data also needs to be forwarded to real-time listeners (async execution outside lock)
                    except Exception as e:
                        logger.error(f"TCP Aduio: {str(e)}")
            except socket.timeout:
                continue
            except socket.error as e:
                logger.error(f"TCP Data: {str(e)}")
                break  # Socket error, disconnect
            except Exception as e:
                logger.error(f"TCP Data: {str(e)}")
                break
                
    except Exception as e:
        logger.error(f"TCP Client: {str(e)}")
    finally:
        # Clean up connection and socket - avoid duplicate closure
        if device_id:
            unregister_client(device_id)
        else:
            # If device ID not registered, manually close socket
            try:
                client_socket.close()
            except Exception as e:
                logger.debug(f"TCP connection closure error {client_address}: {e}")

def start_tcp_server():
    """Start TCP server"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 8883))
    server_socket.listen(5)
    
    logger.info("TCP Server : 8883")
    
    while not stop_flag:
        try:
            client_socket, client_address = server_socket.accept()
            client_socket.settimeout(3.0)  # Set timeout to avoid blocking
            client_thread = threading.Thread(
                target=handle_tcp_client, 
                args=(client_socket, client_address), 
                daemon=True
            )
            client_thread.start()
            
        except Exception as e:
            if not stop_flag:
                logger.error(f"❌ TCP server error: {str(e)}")
            break
    server_socket.close()
    logger.info("🔌 TCP server stopped")

# Real-time listening WebSocket endpoint
@app.websocket("/ws/live_audio/{device_id}")
async def websocket_live_audio_endpoint(websocket: WebSocket, device_id: str):
    """Real-time audio listening WebSocket endpoint"""
    await websocket.accept()
    
    try:
        # Register listener
        with live_listeners_lock:
            if device_id not in live_listeners:
                live_listeners[device_id] = []
            live_listeners[device_id].append(websocket)
        
        # logger.info(f"🎧 Started real-time listening for device: {device_id}")
        
        # Send connection confirmation message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "device_id": device_id,
            "message": f"Connected to audio stream of device {device_id}"
        }))
        
        # Keep connection and handle client messages
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                
                if message.get("type") == "websocket.receive":
                    if "text" in message:
                        # Handle client text messages (like heartbeat)
                        try:
                            client_msg = json.loads(message["text"])
                            if client_msg.get("type") == "ping":
                                await websocket.send_text(json.dumps({"type": "pong"}))
                        except json.JSONDecodeError:
                            logger.warning("Received invalid JSON message")
                elif message.get("type") == "websocket.disconnect":
                    # logger.info(f"🎧 Real-time listening disconnected: {device_id}")
                    break
                    
            except asyncio.TimeoutError:
                # 30-second timeout, send heartbeat to check connection
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except:
                    logger.info(f"Live discount: {device_id}")
                    break
            except Exception as e:
                logger.warning(f"🎧 Real-time listening message processing error: {e}")
                break
                
    except Exception as e:
        logger.error(f"❌ Real-time listening WebSocket error: {e}")
    finally:
        # Clean up listeners
        with live_listeners_lock:
            if device_id in live_listeners and websocket in live_listeners[device_id]:
                live_listeners[device_id].remove(websocket)
                if not live_listeners[device_id]:
                    del live_listeners[device_id]
        logger.info(f"Live clear: {device_id}")
 
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8882)