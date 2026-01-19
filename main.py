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
            old.connection_obj.close()
            logger.info(f"Cleaned old TCP connection for device {old.device_id}")
        except Exception as e:
            logger.warning(f"Failed to clean connection for device {old.device_id}: {e}")
    elif old.connection_type == "websocket" and old.connection_obj != new_obj:
        try:
            # WebSocket连接会在其自己的异常处理中关闭
            logger.info(f"WebSocket连接 {old.device_id} 将自动清理")
        except Exception as e:
            logger.warning(f"WebSocket {old.device_id} 清理警告: {e}")

def register_client(device_id: str, connection_type: str, connection_obj: object ) -> Dict:
    """注册客户端连接并返回录音状态"""
    current_time = time.time()
    duration = 0
    # 获取当前录音状态
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
        # 如果设备已存在，检查连接类型并处理旧连接
        if device_id in device_sessions:
            existing_session = device_sessions[device_id]
            if existing_session.connection_type == connection_type:
                # 相同连接类型，清理旧连接
                clean_connection(existing_session, connection_obj)
                logger.warning(f"设备 {device_id} 重复 {connection_type} 连接，已清理旧连接")
            else:
                # 不同连接类型，记录警告但允许共存
                logger.warning(f"设备 {device_id} 多连接类型: 已有 {existing_session.connection_type}, 新增 {connection_type}")
        
        # 创建或更新设备会话
        device_sessions[device_id] = DeviceSession(
            device_id=device_id,
            connection_type=connection_type,
            connection_obj=connection_obj,
            start_time=current_time,
        )
        logger.info(f"Device Connect: {device_id} via {connection_type}")
    return duration

def unregister_client(device_id: str) -> bool:
    """注销客户端连接"""
    with sessions_lock:
        if device_id in device_sessions:
            try:
                clean_connection(device_sessions[device_id], None)
                del device_sessions[device_id]
                logger.info(f"Device discount: {device_id}")
                return True
            except Exception as e:
                logger.error(f"注销设备 {device_id} 失败: {e}")
                return False
        return False

def clean_devices():
    """定期清理非活跃设备的后台任务"""
    while not stop_flag:
        try:
            current_time = time.time()
            inactive_devices = []
            with sessions_lock:
                for device_id, session in device_sessions.items():
                    if current_time - session.last_chunk_time > DEVICE_TIMEOUT:
                        inactive_devices.append(device_id)
            for device_id in inactive_devices:
                logger.warning(f"🕒 设备超时，自动清理: {device_id}")
                unregister_client(device_id)
        except Exception as e:
            logger.error(f"❌ 设备清理任务异常: {str(e)}")
        finally:
            time.sleep(DEVICE_TIMEOUT)

def check_device_status(session: str,device_id: str) -> dict:
    """统一的设备状态检查函数"""
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
    """获取设备列表（包括在线和离线设备）"""
    devices_info = []
    # 获取活跃设备信息
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
    """获取设备实时状态"""
    return check_device_status(device_sessions.get(device_id), device_id)


@app.get("/devices/{device_id}/recordings")
async def get_device_recordings(device_id: str, auth: bool = Depends(require_auth)):
    """获取指定设备的录音文件列表（按日期文件夹组织）"""
    device_dir = os.path.join(RECORDINGS, device_id)
    
    if not os.path.exists(device_dir):
        raise HTTPException(status_code=404, detail=f"设备 {device_id} 不存在")
    
    recordings_by_date = {}
    # 遍历设备目录下的日期文件夹
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
    """获取指定设备指定日期的音频播放列表"""
    date_dir = os.path.join(RECORDINGS, device_id, date)
    
    if not os.path.exists(date_dir):
        raise HTTPException(status_code=404, detail=f"设备 {device_id} 的日期 {date} 不存在")
    
    audio_files = []
    for file in os.listdir(date_dir):
        if file.endswith('.wav'):
            filepath = os.path.join(date_dir, file)
            stat = os.stat(filepath)
            
            # 读取音频信息
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
    
    # 按创建时间排序
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
    """获取音频文件"""
    filepath = os.path.join(RECORDINGS, device_id, date, filename)
    
    if not os.path.exists(filepath) or not filename.endswith('.wav'):
        raise HTTPException(status_code=404, detail="音频文件不存在")
    
    return FileResponse(
        filepath,
        media_type="audio/wav",
        filename=filename,
        headers={"Accept-Ranges": "bytes"}
    )

@app.post("/login")
async def login(request: LoginRequest, response: Response):
    """用户登录验证"""
    if request.password == SYSTEM_PASSWORD:
        # 创建会话
        session_token = generate_session_token()
        sessions[session_token] = {
            "created_at": time.time(),
            "last_activity": time.time()
        }
        
        # 设置会话cookie
        response.set_cookie(
            "session_token", 
            session_token, 
            max_age=86400,  # 24小时
            httponly=True,
            secure=False  # 在生产环境中应该设置为True
        )
        
        return {"success": True, "message": "登录成功"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="密码错误"
        )

@app.get("/check-auth")
async def check_auth(request: Request):
    """检查认证状态"""
    authenticated = verify_session(request)
    return {"authenticated": authenticated}

@app.post("/logout")
async def logout(request: Request, response: Response):
    """用户退出登录"""
    session_token = request.cookies.get("session_token")
    if session_token and session_token in sessions:
        del sessions[session_token]
    
    response.delete_cookie("session_token")
    return {"success": True, "message": "已退出登录"}

@app.get("/login", response_class=HTMLResponse)
async def get_login_page():
    """获取登录页面"""
    with open("login.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/", response_class=HTMLResponse)
async def get_playlist_page(request: Request):
    """获取音频播放页面（需要认证）"""
    if not verify_session(request):
        return RedirectResponse(url="/login")
    
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# 存储实时监听连接
live_listeners: Dict[str, List[WebSocket]] = {}
live_listeners_lock = threading.Lock()

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """WebSocket音频传输端点（优化的非阻塞版本）"""
    global stop_flag
    await websocket.accept()
    device_id = None
    
    try:       
        while not stop_flag:
            # 接收消息 - 可能是文本(设备ID)或二进制(音频数据)
            try:
                # 添加超时避免永久阻塞
                message = await asyncio.wait_for(websocket.receive(), timeout=5.0)
                # logger.info(f"🔍 WebSocket收到消息: {message}")
                
                if "text" in message:
                    # 处理文本消息 (设备ID注册)
                    try:
                        data = json.loads(message["text"])
                        # logger.info(f"🔍 解析JSON成功: {data}")
                        if data.get("type") == "device_id":
                            device_id = data.get("id")
                            recording_duration = register_client(device_id, "websocket", websocket )
                            try:
                                await websocket.send_text(recording_duration)
                                logger.info(f"✅ WebSocket设备注册成功: {device_id}")
                            except Exception as e:
                                logger.error(f"❌ WebSocket发送录音配置失败: {e}")
                        else:
                            logger.warning(f"❌ 未知消息类型: {data.get('type')}")
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ WebSocket收到无效JSON消息: {e}")
                        
                elif "bytes" in message:
                    if not device_id:
                        logger.warning("❌ 收到音频数据但设备未注册")
                        continue
                    raw_data = message["bytes"]
                    if raw_data:
                        try:
                            # 使用asyncio.create_task让音频处理不阻塞WebSocket接收
                            asyncio.create_task(
                                process_audio_async(device_id, raw_data)
                            )
                        except Exception as e:
                            logger.error(f"❌ 处理WebSocket音频数据失败: {str(e)}")
                            
            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except WebSocketDisconnect:
                logger.info(f"📱 WebSocket设备断开: {device_id}")
                break
                
    except Exception as e:
        logger.error(f"❌ WebSocket连接异常: {str(e)}")
    finally:
        if device_id:
            unregister_client(device_id)

def process_audio(device_id: str, raw_data: bytes):
    """统一的音频数据处理函数"""
    try:
        session=device_sessions.get(device_id)
        if not session:
            logger.warning(f"⚠️ 设备 {device_id} 会话不存在，跳过音频处理")
            return None
        current_time = time.time()
        with sessions_lock:
            session.last_chunk_time = current_time
            session.raw_audio.extend(np.frombuffer(raw_data, dtype=np.int16))
            
            # 检查是否需要保存（快速检查）
            if current_time - session.start_time >= 180.0 and len(session.raw_audio) > 0:
                audio_to_save = session.raw_audio.copy()
                session.raw_audio.clear()
                session.start_time = current_time
                return audio_to_save
            
    except Exception as e:
        logger.error(f"❌ 音频处理失败 {device_id}: {str(e)}")

    return None

async def process_audio_async(device_id: str, raw_data: bytes):
    # 转发音频数据给实时监听者（在锁外进行）
    await audio_to_listeners(device_id, raw_data)
    """异步音频数据处理包装器"""
    audio_to_save = process_audio(device_id, raw_data)
    if audio_to_save:
        # logger.info(f"💾 设备 {device_id} 3分钟文件保存")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_device_audio, device_id, audio_to_save)

# 转发音频数据给实时监听者
async def audio_to_listeners(device_id: str, raw_data: bytes):
    """将音频数据转发给实时监听的WebSocket连接（优化版本）"""
    if device_id not in live_listeners:
        return
    
    try:
        int16_data = np.frombuffer(raw_data, dtype=np.int16)
    
        # 优化：使用内存操作而非临时文件
        import io
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, int16_data, 16000, format='WAV')
        audio_wav_data = audio_buffer.getvalue()
        audio_buffer.close()
        
        # 转发给所有监听者
        disconnected_listeners = []
        with live_listeners_lock:
            listeners = live_listeners.get(device_id, []).copy()
        
        for listener_ws in listeners:
            try:
                await listener_ws.send_bytes(audio_wav_data)
            except Exception as e:
                logger.warning(f"转发音频数据失败: {e}")
                disconnected_listeners.append(listener_ws)
        
        # 清理断开的连接
        if disconnected_listeners:
            with live_listeners_lock:
                if device_id in live_listeners:
                    for ws in disconnected_listeners:
                        if ws in live_listeners[device_id]:
                            live_listeners[device_id].remove(ws)
                    if not live_listeners[device_id]:
                        del live_listeners[device_id]
                        
    except Exception as e:
        logger.error(f"转发音频数据处理失败: {e}")

@app.get("/devices/{device_id}/daily_transcript/{date}")
async def get_daily_transcript(device_id: str, date: str, auth: bool = Depends(require_auth)):
    """获取指定设备指定日期的完整转录文本"""
    date_dir = os.path.join(RECORDINGS, device_id, date)
    
    if not os.path.exists(date_dir):
        raise HTTPException(status_code=404, detail=f"设备 {device_id} 的日期 {date} 不存在")
    
    # 查找所有txt文件
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
            "message": "当日无转录文本"
        }
    
    # 按创建时间排序
    txt_files.sort(key=lambda x: x[1])
    
    # 拼接所有文本内容
    all_text = []
    total_lines = 0
    
    for filepath, _ in txt_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                all_text.extend(lines)
                total_lines += len(lines)
        except Exception as e:
            logger.warning(f"⚠️ 读取文本文件失败: {filepath}, {e}")
    
    # 拼接成完整文本
    full_transcript = "".join(all_text).strip()
    
    return {
        "device_id": device_id,
        "date": date,
        "transcript": full_transcript,
        "total_lines": total_lines,
        "files_count": len(txt_files)
    }

# 简化的录音计划管理API
@app.get("/schedules/{device_id}")
async def get_device_schedules(device_id: str, auth: bool = Depends(require_auth)):
    """获取指定设备的录音计划"""
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
    """添加录音计划"""
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
        return {"success": True, "message": "录音计划添加成功"}
        
    except Exception as e:
        logger.error(f"❌ 添加录音计划失败: {e}")
        raise HTTPException(status_code=400, detail=f"添加录音计划失败: {str(e)}")

def handle_tcp_client(client_socket, client_address):
    """处理单个TCP客户端连接"""
    device_id = None
    buffer = b""
    
    try:
        while not stop_flag:
            try:
                # 接收数据
                data = client_socket.recv(4096)
                if not data:
                    # logger.info(f"📱 TCP客户端主动断开: {client_address}")
                    break
                
                buffer += data
                # 如果还没有设备ID，先查找设备ID
                if device_id is None:
                    # 查找简化的设备ID消息格式：device_id:xxxxx
                    try:
                        buffer_str = buffer.decode('utf-8', errors='ignore')
                        if 'device_id:' in buffer_str:
                            # 找到设备ID消息
                            device_id_start = buffer_str.find('device_id:')
                            if device_id_start != -1:
                                # 提取设备ID（到换行符或缓冲区结束）
                                id_start = device_id_start + len('device_id:')
                                line_end = buffer_str.find('\n', id_start)
                                if line_end == -1:
                                    line_end = len(buffer_str)
                                
                                device_id = buffer_str[id_start:line_end].strip()
                                if device_id:                                    
                                    # 注册TCP客户端并获取录制时长
                                    recording_duration = register_client(device_id, "tcp", client_socket)
                                    # 发送录制时长响应给客户端（简单数字格式）
                                    try:
                                        response = f"{recording_duration}\n"
                                        client_socket.send(response.encode('utf-8'))
                                        logger.info(f"IP：{client_address} ID: {device_id} Duration: {recording_duration} S")
                                    except socket.error as e:
                                        logger.error(f"Send error: {e}")
                                        break  # Socket错误，跳出循环
                                    except Exception as e:
                                        logger.error(f"Send error: {e}")
                                    
                                    # 移除已处理的设备ID消息
                                    processed_bytes = device_id_start + len(f'device_id:{device_id}')
                                    if line_end < len(buffer_str):
                                        processed_bytes += 1  # 包含换行符
                                    buffer = buffer[processed_bytes:]
                    except UnicodeDecodeError:
                        # 如果不能解码为UTF-8，可能是音频数据
                        pass
                    
                    # 如果缓冲区过大但还没找到设备ID，清理一部分
                    if len(buffer) > 2048:
                        buffer = buffer[-1024:]
                
                # 如果有设备ID，处理音频数据
                if device_id and len(buffer) >= 2048:
                    # 提取1024字节的音频数据
                    audio_data = buffer[:2048]
                    buffer = buffer[2048:]
                    
                    try:
                        # 使用线程安全的方法转发音频数据
                        threading.Thread(
                            target=lambda: asyncio.run(audio_to_listeners(device_id, audio_data)),
                            daemon=True
                        ).start()
                        # 使用统一的音频处理函数
                        audio_to_save = process_audio(device_id, audio_data)
                        if audio_to_save:
                            # logger.info(f"💾 TCP设备 {device_id} 3分钟文件保存")
                            threading.Thread(
                                target=save_device_audio,
                                args=(device_id, audio_to_save),
                                daemon=True
                            ).start()
                        # TCP音频数据也需要转发给实时监听者（在锁外异步执行）
                    except Exception as e:
                        logger.error(f"TCP Aduio: {str(e)}")
            except socket.timeout:
                continue
            except socket.error as e:
                logger.error(f"TCP Data: {str(e)}")
                break  # Socket错误，断开连接
            except Exception as e:
                logger.error(f"TCP Data: {str(e)}")
                break
                
    except Exception as e:
        logger.error(f"TCP Client: {str(e)}")
    finally:
        # 清理连接
        if device_id:
            unregister_client(device_id)
        try:
            client_socket.close()
        except Exception as e:
            logger.debug(f"关闭TCP连接异常 {client_address}: {e}")

def start_tcp_server():
    """启动TCP服务器"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 8883))
    server_socket.listen(5)
    
    logger.info("TCP Server : 8883")
    
    while not stop_flag:
        try:
            client_socket, client_address = server_socket.accept()
            client_socket.settimeout(3.0)  # 设置超时避免阻塞
            client_thread = threading.Thread(
                target=handle_tcp_client, 
                args=(client_socket, client_address), 
                daemon=True
            )
            client_thread.start()
            
        except Exception as e:
            if not stop_flag:
                logger.error(f"❌ TCP服务器异常: {str(e)}")
            break
    server_socket.close()
    logger.info("🔌 TCP服务器已停止")

# 实时监听WebSocket端点
@app.websocket("/ws/live_audio/{device_id}")
async def websocket_live_audio_endpoint(websocket: WebSocket, device_id: str):
    """实时音频监听WebSocket端点"""
    await websocket.accept()
    
    try:
        # 注册监听者
        with live_listeners_lock:
            if device_id not in live_listeners:
                live_listeners[device_id] = []
            live_listeners[device_id].append(websocket)
        
        # logger.info(f"🎧 开始实时监听设备: {device_id}")
        
        # 发送连接确认消息
        await websocket.send_text(json.dumps({
            "type": "connected",
            "device_id": device_id,
            "message": f"已连接到设备 {device_id} 的音频流"
        }))
        
        # 保持连接并处理客户端消息
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                
                if message.get("type") == "websocket.receive":
                    if "text" in message:
                        # 处理客户端文本消息（如心跳）
                        try:
                            client_msg = json.loads(message["text"])
                            if client_msg.get("type") == "ping":
                                await websocket.send_text(json.dumps({"type": "pong"}))
                        except json.JSONDecodeError:
                            logger.warning("收到无效的JSON消息")
                elif message.get("type") == "websocket.disconnect":
                    # logger.info(f"🎧 实时监听断开: {device_id}")
                    break
                    
            except asyncio.TimeoutError:
                # 30秒超时，发送心跳检查连接
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except:
                    logger.info(f"Live discount: {device_id}")
                    break
            except Exception as e:
                logger.warning(f"🎧 实时监听消息处理错误: {e}")
                break
                
    except Exception as e:
        logger.error(f"❌ 实时监听WebSocket错误: {e}")
    finally:
        # 清理监听者
        with live_listeners_lock:
            if device_id in live_listeners and websocket in live_listeners[device_id]:
                live_listeners[device_id].remove(websocket)
                if not live_listeners[device_id]:
                    del live_listeners[device_id]
        logger.info(f"Live clear: {device_id}")
 
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8882)