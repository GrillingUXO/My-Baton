import cv2
import numpy as np
import time
from tkinter import Tk, filedialog
from mido import MidiFile
from collections import deque
import fluidsynth
import threading
from threading import Lock
import mediapipe as mp
from enum import Enum
import math

class ProgramState(Enum):
    INITIALIZING_BPM = 1
    PLAYING_MIDI = 2
    EXITING = 3


# 新增全局变量区
current_beat = 0                 # 当前播放的拍子序号
is_auto_playing = False          # 是否处于自动播放模式
last_rhythm_hand_time = None     # 最后一次检测到节奏手的时间
AUTO_PLAY_TIMEOUT = 1.0          # 自动播放超时时间（秒）
beat_lock = threading.Lock()     # 节拍计数器锁


global_active_notes = {}  # 格式: {(channel, pitch): {"end_sec": float, "velocity": int}}
global_notes_lock = threading.Lock()



# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)


rhythm_hand_label = None  # 节奏手的左右信息（"Left" 或 "Right"）
control_hand_label = None  # 变化手的左右信息（"Left" 或 "Right"）



last_distance_to_torso = None  # 记录变化手上一帧与躯干的距离

beat_lock = Lock()
playback_thread = None
stop_playback = False


# 在全局变量定义区添加以下变量
pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
last_volume_update_time = None  # 上一次音量调整的时间戳
velocity = 64  # 初始音量（0-127）


# 新增全局变量
prev_palm_position = None  # 用于记录上一帧手掌位置
volume = 100  # 初始音量


fluid_lock = Lock()

STOP_THRESHOLD = 20
STOP_DURATION = 0.02
NOTE_INTERVAL = 0.1  # Interval between playing consecutive notes within a beat (seconds)

velocity = 64
bpm = 120  # Initial BPM based on test bar

prev_position = None
prev_time = None
last_stop_time = None
prev_direction = None
current_beat = 0
play_parameters = {"velocity": velocity, "volume": volume}
# 初始化全局变量


# 全局变量
hand_hist = None  # 手部直方图
motion_amplitude = []  # 运动幅度缓冲区
animation_point = None  # 停顿动画位置
animation_start_time = None  # 停顿动画开始时间
animation_duration = 0.5  # 停顿动画持续时间
STOP_THRESHOLD = 50  # 停顿检测的运动幅度阈值


last_stop_position = None
motion_distance_since_last_stop = 0
trajectory = deque(maxlen=30)
animation_point = None
animation_start_time = None
animation_duration = 0.5

root = Tk()
root.withdraw()

fs = None  # FluidSynth instance


def select_midi_and_soundfont_files():
    """Select MIDI and SoundFont files using a GUI dialog."""
    global soundfont_path

    # Select MIDI file
    print("请选择 MIDI 文件。")
    midi_file_path = filedialog.askopenfilename(
        title="选择 MIDI 文件",
        filetypes=[("MIDI 文件", "*.mid"), ("所有文件", "*.*")]
    )
    if not midi_file_path:
        print("未选择 MIDI 文件，程序退出。")
        cleanup_fluidsynth()
        exit()

    # Select SoundFont file
    print("请选择 SoundFont 文件。")
    soundfont_path = filedialog.askopenfilename(
        title="选择 SoundFont 文件",
        filetypes=[("SoundFont 文件", "*.sf2"), ("所有文件", "*.*")]
    )
    if not soundfont_path:
        print("未选择 SoundFont 文件，程序退出。")
        cleanup_fluidsynth()
        exit()

    return midi_file_path



def process_frame_with_hand_detection(frame, hand_hist, prev_position, stop_detected, current_beat, beats_notes, total_beats, last_stop_time):
    """
    使用 MediaPipe 检测两只手的动作并处理逻辑，返回控制信号和状态
    修改点：
    - 新增返回值 play_beat_command (是否触发播放)
    - 新增返回值 current_bpm (当前帧计算的 BPM)
    - 解耦播放触发和 BPM 计算逻辑
    """
    global hands, mp_drawing, mp_hands, pose, bpm, motion_amplitude, last_pause_info, playback_thread, stop_playback
    global rhythm_hand_label, control_hand_label, velocity, last_volume_update_time, prev_time

    # 初始化控制信号变量
    play_beat_command = False  # 节拍播放触发标志
    current_bpm = bpm          # 当前计算的 BPM（默认使用全局值）
    new_last_stop_time = last_stop_time  # 用于临时存储新时间戳

    avg_motion = 0
    recognized_hand_mouvement_scrolling = None
    speed_threshold = 300
    distance_threshold = 200

    # 躯干检测逻辑（完整保留）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb_frame)
    torso_center = None
    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        left_shoulder = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        torso_center = (
            int((left_shoulder.x + right_shoulder.x)/2 * frame.shape[1]),
            int((left_shoulder.y + right_shoulder.y)/2 * frame.shape[0])
        )
        cv2.circle(frame, torso_center, 5, (0,255,0), -1)

    # 手势检测逻辑（完整保留）
    hand_result = hands.process(rgb_frame)
    if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
        hand_landmarks_list = hand_result.multi_hand_landmarks
        handedness_list = hand_result.multi_handedness

        # 绑定节奏手和变化手（完整逻辑）
        if rhythm_hand_label is None or control_hand_label is None:
            for idx, handedness in enumerate(handedness_list):
                label = handedness.classification[0].label
                if rhythm_hand_label is None:
                    rhythm_hand_label = label
                elif control_hand_label is None and label != rhythm_hand_label:
                    control_hand_label = label

        # 获取节奏手和变化手实例
        rhythm_hand = None
        control_hand = None
        for idx, handedness in enumerate(handedness_list):
            label = handedness.classification[0].label
            if label == rhythm_hand_label:
                rhythm_hand = hand_landmarks_list[idx]
            elif label == control_hand_label:
                control_hand = hand_landmarks_list[idx]

        # 节奏手逻辑（修改触发逻辑）
        if rhythm_hand:
            mp_drawing.draw_landmarks(frame, rhythm_hand, mp_hands.HAND_CONNECTIONS)
            wrist = rhythm_hand.landmark[mp_hands.HandLandmark.WRIST]
            wrist_pos = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))

            # 方向检测（完整保留）
            if prev_position is not None:
                dx = wrist_pos[0] - prev_position[0]
                dy = wrist_pos[1] - prev_position[1]
                mouvement_distance = (dx**2 + dy**2)**0.5
                mouvement_distance_threshold = 0.02 * frame.shape[0]
                
                if mouvement_distance > mouvement_distance_threshold:
                    angle = math.degrees(math.atan2(dy, dx))
                    if -45 <= angle < 45:
                        recognized_hand_mouvement_scrolling = "Scrolling right"
                    elif 45 <= angle < 135:
                        recognized_hand_mouvement_scrolling = "Scrolling up"
                    elif angle >= 135 or angle < -135:
                        recognized_hand_mouvement_scrolling = "Scrolling left"
                    elif -135 <= angle < -45:
                        recognized_hand_mouvement_scrolling = "Scrolling down"
                    cv2.putText(frame, recognized_hand_mouvement_scrolling, 
                               (wrist_pos[0]+20, wrist_pos[1]-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            # 触发条件检测
            if prev_position is not None and prev_time is not None:
                current_time = time.time()
                delta_time = current_time - prev_time
                dx = wrist_pos[0] - prev_position[0]
                dy = wrist_pos[1] - prev_position[1]
                distance = (dx**2 + dy**2)**0.5
                speed = distance / delta_time

                if speed > speed_threshold and distance > distance_threshold:
                    # 计算新 BPM
                    if last_stop_time is not None:
                        interval = current_time - last_stop_time
                        current_bpm = max(60, min(200, 60 / interval))  # 写入临时变量
                    
                    # 设置播放命令
                    play_beat_command = True
                    new_last_stop_time = current_time  # 更新临时时间戳

                prev_time = current_time
            else:
                prev_time = time.time()

            prev_position = wrist_pos
            cv2.putText(frame, "Rhythm Hand", (wrist_pos[0]-50, wrist_pos[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            

        # 变化手逻辑
        if control_hand:
            mp_drawing.draw_landmarks(frame, control_hand, mp_hands.HAND_CONNECTIONS)
            control_wrist = control_hand.landmark[mp_hands.HandLandmark.WRIST]
            control_wrist_pos = (int(control_wrist.x * frame.shape[1]), 
                                int(control_wrist.y * frame.shape[0]))
            
            if torso_center:
                distance_to_torso = ((control_wrist_pos[0]-torso_center[0])**2 + 
                                    (control_wrist_pos[1]-torso_center[1])**2)**0.5
                current_time = time.time()
                if last_volume_update_time is None or current_time - last_volume_update_time > 0.2:
                    if distance_to_torso < 150:
                        velocity = max(0, velocity-5)
                    elif distance_to_torso > 250:
                        velocity = min(127, velocity+5)
                    with fluid_lock:
                        fs.cc(0, 7, velocity)
                    last_volume_update_time = current_time
                cv2.putText(frame, f"Distance: {distance_to_torso:.2f}", (10,100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.putText(frame, "Control Hand", (control_wrist_pos[0]-50, control_wrist_pos[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(frame, f"BPM: {current_bpm:.2f}", (10,30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Play Command: {play_beat_command}", (10,200),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    cv2.putText(frame, f"Velocity: {velocity}", (10,70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    return (
        prev_position,
        stop_detected,
        current_beat,
        new_last_stop_time if 'new_last_stop_time' in locals() else last_stop_time,
        play_beat_command, 
        current_bpm       
    )


    
def handle_auto_play_control(hand_result, all_voices_notes, current_bpm, volume):
    """
    改进的自动播放控制函数：
    - 仅根据节奏手（rhythm_hand_label）切换模式，变化手不影响切换逻辑。
    - 节奏手消失 1 秒后才切换到自动播放。
    - 节奏手出现 1 秒后才切换到手动模式。
    - 避免检测到变化手导致错误触发切换（播放 1 秒，停 1 秒问题）。
    """
    global is_auto_playing, last_rhythm_hand_time, stop_playback

    # 确保 BPM 合法
    if current_bpm is None or current_bpm <= 0:
        current_bpm = 120  # 设置安全默认值

    current_time = time.time()
    rhythm_detected = False

    # 检测是否找到 **节奏手**
    if hand_result.multi_hand_landmarks:
        for i, handedness in enumerate(hand_result.multi_handedness):
            label = handedness.classification[0].label
            if label == rhythm_hand_label:  # 确保只用节奏手更新状态
                rhythm_detected = True
                if last_rhythm_hand_time is None:
                    last_rhythm_hand_time = current_time  # 记录节奏手首次出现的时间
                break  # **跳出循环，避免误判**

    # **避免错误地更新 last_rhythm_hand_time**
    if not rhythm_detected:
        if last_rhythm_hand_time is None:  # 只有第一次未检测到时才初始化
            last_rhythm_hand_time = current_time

    # **模式切换逻辑**
    if rhythm_detected:
        # **节奏手稳定出现超过 1 秒，切换到手动模式**
        if is_auto_playing and (current_time - last_rhythm_hand_time >= 1.0):
            print("切回手动模式，保留播放位置")
            is_auto_playing = False
            stop_playback = True  # 立即停止自动播放
    else:
        # **节奏手消失超过 1 秒，切换到自动播放**
        if not is_auto_playing and (current_time - last_rhythm_hand_time >= 1.0):
            print("节奏手消失超过 1 秒，进入自动播放模式")
            is_auto_playing = True
            start_full_midi_playback(all_voices_notes, current_bpm, volume)

    # **只有在检测到节奏手时才更新 last_rhythm_hand_time**
    if rhythm_detected:
        last_rhythm_hand_time = current_time


                

def start_full_midi_playback(all_voices_notes, current_bpm, volume):
    """
    改进的完整MIDI播放函数（解决复音限制问题）
    """
    global fs, playback_thread, stop_playback, current_playback_position
    
    def playback_worker():
        global current_playback_position
        nonlocal current_bpm
        
        try:
            # 强制终止所有残留音符
            with fluid_lock:
                for channel in range(16):
                    fs.cc(channel, 123, 0)  # All Notes Off
                    fs.cc(channel, 121, 0)  # Reset All Controllers

            # BPM有效性检查
            current_bpm = max(20, min(300, current_bpm)) if isinstance(current_bpm, (int, float)) else 120
            
            # 构建精确事件队列
            events = []
            for voice in all_voices_notes:
                program = voice["program"]
                for note in voice["notes"]:
                    events.append({
                        "type": "on",
                        "time": note["start_sec"],
                        "channel": program,
                        "pitch": note["pitch"],
                        "velocity": int(note["velocity"] * (volume / 127.0))
                    })
                    events.append({
                        "type": "off",
                        "time": note["end_sec"],
                        "channel": program,
                        "pitch": note["pitch"]
                    })
            
            # 按时间排序事件
            events.sort(key=lambda x: x["time"])
            
            # 播放执行
            start_time = time.time()
            event_idx = 0
            last_clean = time.time()
            
            while not stop_playback and is_auto_playing:
                current_time = time.time() - start_time
                
                # 处理到期事件
                while event_idx < len(events) and events[event_idx]["time"] <= current_time:
                    event = events[event_idx]
                    with fluid_lock:
                        if event["type"] == "on":
                            fs.noteon(event["channel"], event["pitch"], event["velocity"])
                        else:
                            fs.noteoff(event["channel"], event["pitch"])
                    event_idx += 1
                
                # 定期清理（防止事件堆积）
                if time.time() - last_clean > 0.5:
                    with fluid_lock:
                        for ch in range(16):
                            fs.cc(ch, 123, 0)  # 每0.5秒强制清理
                    last_clean = time.time()
                
                time.sleep(0.001)  # 降低CPU占用

        finally:
            # 终止时强制静音
            with fluid_lock:
                for channel in range(16):
                    fs.cc(channel, 123, 0)
                    fs.cc(channel, 121, 0)

    # 线程管理
    if playback_thread and playback_thread.is_alive():
        stop_playback = True
        try:
            playback_thread.join(timeout=0.5)
        except RuntimeError:
            pass
    
    stop_playback = False
    playback_thread = threading.Thread(target=playback_worker)
    playback_thread.start()

    


def calculate_angle(vec1, vec2):
    """计算两向量之间的夹角（单位：度）"""
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    magnitude1 = (vec1[0]**2 + vec1[1]**2)**0.5
    magnitude2 = (vec2[0]**2 + vec2[1]**2)**0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # 防止除以零
    cos_theta = dot_product / (magnitude1 * magnitude2)
    cos_theta = max(-1, min(1, cos_theta))  # 防止浮点误差导致超出范围
    return math.degrees(math.acos(cos_theta))




def hist_masking(frame, hist):
    """基于 MediaPipe Hands 检测手部区域，并生成掩模。"""
    global hands, mp_drawing, mp_hands

    # 将图像转换为 RGB 格式，因为 MediaPipe 使用的是 RGB 输入
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # 如果检测到手部
    if result.multi_hand_landmarks:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for hand_landmarks in result.multi_hand_landmarks:
            # 将手部关键点投影到图像上
            hand_points = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) for p in hand_landmarks.landmark]
            hull = cv2.convexHull(np.array(hand_points, dtype=np.int32))  # 生成手部凸包
            cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)  # 填充手部区域

        # 使用生成的掩模对帧进行隔离
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame
    else:
        # 未检测到手时返回空帧
        return np.zeros_like(frame)

def contours(hist_mask_image):
    """返回基于 MediaPipe 检测的手部关键点轮廓。"""
    global hands, mp_hands

    rgb_frame = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        contours_list = []
        for hand_landmarks in result.multi_hand_landmarks:
            # 获取手部关键点投影
            hand_points = [(int(p.x * hist_mask_image.shape[1]), int(p.y * hist_mask_image.shape[0])) for p in hand_landmarks.landmark]
            hull = cv2.convexHull(np.array(hand_points, dtype=np.int32))
            contours_list.append(hull)
        return contours_list
    return []


def centroid(max_contour):
    """计算基于 MediaPipe 关键点的手部中心点。"""
    if max_contour is None or len(max_contour) == 0:
        return None
    moment = cv2.moments(np.array(max_contour))
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    return None



def start_fluidsynth():
    global fs, soundfont_path, sfid

    if fs is not None:
        return

    try:
        fs = fluidsynth.Synth()
        fs.start(driver="coreaudio")  # Windows 用 "dsound", Linux 用 "alsa"

        
        # 加载 SoundFont
        sfid = fs.sfload(soundfont_path)
        
        # 初始化所有通道
        with fluid_lock:
            for channel in range(16):
                fs.program_select(channel, sfid, 0, 0)
        print(f"FluidSynth 初始化完成 | 复音数: 1024")
    except Exception as e:
        print(f"初始化 FluidSynth 时出错: {e}")
        cleanup_fluidsynth()
        exit()


def cleanup_fluidsynth():
    """Clean up FluidSynth resources. Ensure it is called only once."""
    global fs, playback_thread

    if playback_thread and playback_thread.is_alive():
        print("等待播放线程结束...")
        stop_playback = True
        playback_thread.join()

    if fs is not None:
        print("Cleaning up FluidSynth...")
        fs.delete()
        fs = None



def calculate_note_durations(bpm):
    """
    计算音符时值（基于 BPM）
    改进点：
    - 保留原始逻辑，无需修改
    """
    beat_duration = 60 / bpm  # Duration of a quarter note (4th note)
    return {
        "16th": beat_duration / 4,
        "8th": beat_duration / 2,
        "4th": beat_duration,
    }



# GUI setup and MIDI preprocessing logic
root = Tk()
root.withdraw()

midi_file_path = filedialog.askopenfilename(title="选择 MIDI 文件", filetypes=[("MIDI 文件", "*.mid"), ("所有文件", "*.*")])
if not midi_file_path:
    print("未选择 MIDI 文件，程序退出。")
    cleanup_fluidsynth()
    exit()





import pretty_midi

from mido import MidiFile, MidiTrack, MetaMessage, Message
import pretty_midi


current_playback_position = 0.0  # 当前播放位置（秒）
is_auto_playing = False
playback_events = []  # 预处理的全局播放事件列表

def preprocess_midi(midi_file_path):
    """
    解析 MIDI 文件并提取多声部音符数据
    返回值：beats_notes, ticks_per_beat, bpm
    """
    global sfid  # 访问全局SoundFont ID

    try:
        # 使用 pretty_midi 解析 MIDI 文件
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        bpm = midi_data.estimate_tempo()  # 获取估计的 BPM
        ticks_per_beat = midi_data.resolution  # 获取每拍的 ticks 数

        # 初始化多声部数据结构
        all_voices_notes = []

        # 遍历每个乐器（声部）
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue  # 跳过鼓轨道

            # 动态设置通道音色
            channel = instrument.program % 16  # 确保通道号在0-15范围内
            target_program = instrument.program % 128  # 规范Program范围(0-127)
            
            with fluid_lock:
                try:
                    # 关键修复：使用正确的sfid参数，添加bank自动探测
                    # 尝试默认bank 0
                    fs.program_select(channel, sfid, 0, target_program)
                    print(f"通道{channel} 设置成功: Bank 0, Program {target_program}")
                except fluidsynth.FluidError:
                    try:
                        # 尝试通用bank 128
                        fs.program_select(channel, sfid, 128, target_program)
                        print(f"通道{channel} 使用Bank 128, Program {target_program}")
                    except fluidsynth.FluidError:
                        # 回退到默认钢琴音色
                        fs.program_select(channel, sfid, 0, 0)
                        print(f"通道{channel} 音色不可用，已回退到钢琴")

            voice_notes = {
                "name": instrument.name if instrument.name else "Unnamed",
                "program": channel,
                "notes": [],
                "original_bpm": bpm
            }

            # 收集原始音符数据（秒为单位）
            for note in instrument.notes:
                voice_notes["notes"].append({
                    "pitch": note.pitch,
                    "start_sec": note.start,
                    "end_sec": note.end,
                    "velocity": note.velocity,
                    "duration_sec": note.end - note.start
                })

            all_voices_notes.append(voice_notes)

        return all_voices_notes, ticks_per_beat, bpm

    except Exception as e:
        print(f"解析 MIDI 文件时出错: {e}")
        cleanup_fluidsynth()
        exit()
        



# Select and load MIDI and SoundFont files
midi_file_path = select_midi_and_soundfont_files()
start_fluidsynth()


beats_notes, ticks_per_beat, bpm = preprocess_midi(midi_file_path)
note_durations = calculate_note_durations(bpm)



total_beats = len(beats_notes)
last_pause_info = {"bpm": None, "tap_times": []}


from queue import Queue
import threading

def detect_pause_and_calculate_bpm(centroid, current_time, frame):
    """
    检测停顿后动态更新 BPM，并播放对应节拍音符，同时更新动画。
    """
    global last_stop_time, bpm, note_durations, last_pause_info

    if last_stop_time is not None:
        interval = current_time - last_stop_time
        if interval > 0:
            bpm = max(60, min(200, 60 / interval))  # 限制 BPM 范围
            note_durations = calculate_note_durations(bpm)
            print(f"动态更新 BPM：{bpm:.2f}")

    last_stop_time = current_time

    # 更新停顿记录
    if "tap_times" in last_pause_info:
        last_pause_info["tap_times"].append(current_time)
    else:
        last_pause_info["tap_times"] = [current_time]
    last_pause_info["bpm"] = bpm

    # 显示动画
    for idx, tap_time in enumerate(last_pause_info["tap_times"][-4:], 1):
        cv2.putText(frame, f"Stop {idx}: {tap_time:.2f}s", (10, 60 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)





def detect_pause(motion_amplitude, prev_position, current_position, stop_detected):
    """
    通用的停顿检测逻辑，用于检测手势停顿。
    返回停顿状态和更新后的运动幅度缓冲区。
    """
    STOP_THRESHOLD = 50  # 停顿运动幅度阈值
    dx = current_position[0] - prev_position[0]
    dy = current_position[1] - prev_position[1]
    distance = (dx**2 + dy**2)**0.5
    motion_amplitude.append(distance)

    # 保持缓冲区长度为 10
    if len(motion_amplitude) > 10:
        motion_amplitude.pop(0)

    avg_motion = np.mean(motion_amplitude)

    if avg_motion < STOP_THRESHOLD:
        if not stop_detected:
            stop_detected = True
            return True, motion_amplitude, stop_detected
    else:
        stop_detected = False  # 停顿结束时清除标记

    return False, motion_amplitude, stop_detected




import threading


def play_midi_beat_persistent(all_voices_notes, play_beat_command, current_bpm, volume, frame):
    """
    改进的 MIDI 播放逻辑：
    1. 允许长音跨拍播放，而不会被截断。
    2. 确保 `global_active_notes` 中的音符在下个拍子继续触发 `note_on`。
    3. 仅在真正需要时发送 `note_off`，避免音符被提前切断。
    """
    global fs, playback_thread, stop_playback, current_beat, interrupt_flag, fluid_lock
    global global_active_notes, global_notes_lock

    if not play_beat_command or fs is None:
        return

    def panic_all_notes():
        """清除所有音符，避免残留声音。"""
        with fluid_lock:
            for channel in range(16):
                fs.cc(channel, 123, 0)  # 123 = All Notes Off
                fs.cc(channel, 121, 0)  # 121 = Reset All Controllers
        with global_notes_lock:
            global_active_notes.clear()

    if playback_thread and playback_thread.is_alive():
        stop_playback = True
        try:
            playback_thread.join(timeout=0.05)
        except RuntimeError:
            pass
        finally:
            panic_all_notes()

    # 计算当前节拍时间范围
    with beat_lock:
        current_beat += 1
        interrupt_flag = True
        beat_duration = 60.0 / max(20, min(current_bpm, 300))  # 限制 BPM 范围
        start_time = (current_beat - 1) * beat_duration
        end_time = current_beat * beat_duration

    def play_notes():
        global stop_playback, interrupt_flag, global_active_notes
        stop_playback = False
        interrupt_flag = False
        events = []

        try:
            carry_over_notes = {}
            with global_notes_lock:
                # **确保跨拍音符继续播放**
                for (ch, pitch), note_info in list(global_active_notes.items()):
                    note_info["remaining_beats"] -= 1
                    if note_info["remaining_beats"] > 0:
                        carry_over_notes[(ch, pitch)] = note_info
                        # **确保音符继续播放**
                        events.append({
                            "type": "note_on",
                            "time": start_time,
                            "pitch": pitch,
                            "velocity": note_info["velocity"],
                            "channel": ch
                        })
                    else:
                        # 只有真正结束的音符才触发 note_off
                        events.append({
                            "type": "note_off",
                            "time": end_time,
                            "pitch": pitch,
                            "channel": ch
                        })

                global_active_notes = carry_over_notes.copy()

            # 处理新进入的音符
            new_notes = {}
            for voice in all_voices_notes:
                program = voice["program"]
                for note in voice["notes"]:
                    note_start = note["start_sec"]
                    note_end = note["end_sec"]

                    if note_end <= note_start:
                        continue

                    note_duration_beats = (note_end - note_start) / beat_duration

                    if start_time <= note_start < end_time:
                        events.append({
                            "type": "note_on",
                            "time": note_start,
                            "pitch": note["pitch"],
                            "velocity": int(note["velocity"] * (volume / 127.0)),
                            "channel": program
                        })

                        if note_duration_beats > 1:
                            new_notes[(program, note["pitch"])] = {
                                "remaining_beats": note_duration_beats - 1,
                                "velocity": int(note["velocity"] * (volume / 127.0))
                            }
                        else:
                            events.append({
                                "type": "note_off",
                                "time": note_end,
                                "pitch": note["pitch"],
                                "channel": program
                            })

            # **更新全局状态，确保音符不会被丢失**
            with global_notes_lock:
                global_active_notes.update(new_notes)

            # 事件排序
            events.sort(key=lambda x: x["time"])
            playback_start = time.perf_counter()

            # 播放 MIDI 事件
            for event in events:
                if stop_playback or interrupt_flag:
                    break

                event_offset = event["time"] - start_time
                while (time.perf_counter() - playback_start) < event_offset:
                    if stop_playback or interrupt_flag:
                        break
                    time.sleep(0.0005)

                with fluid_lock:
                    try:
                        if event["type"] == "note_on":
                            fs.noteon(event["channel"], event["pitch"], event["velocity"])
                        else:
                            fs.noteoff(event["channel"], event["pitch"])
                    except fluidsynth.FluidError as e:
                        if "noteon" in str(e):
                            current_poly = fs.get_setting("synth.polyphony")
                            new_poly = min(current_poly * 2, 4096)
                            fs.set_setting("synth.polyphony", str(new_poly))
                            print(f"动态调整复音数: {current_poly} -> {new_poly}")
                            fs.noteon(event["channel"], event["pitch"], event["velocity"])

            # 清理未正确释放的音符
            residual_clean = []
            with fluid_lock:
                for event in events:
                    if event["type"] == "note_on" and not any(
                        e["type"] == "note_off" and 
                        e["channel"] == event["channel"] and 
                        e["pitch"] == event["pitch"]
                        for e in events
                    ):
                        residual_clean.append((event["channel"], event["pitch"]))

            if residual_clean:
                with fluid_lock:
                    for ch, pitch in residual_clean:
                        fs.noteoff(ch, pitch)

        finally:
            panic_all_notes()

    # 启动线程
    playback_thread = threading.Thread(target=play_notes, daemon=True)
    playback_thread.start()


    

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        cleanup_fluidsynth()
        exit()

    print("按 'q' 键退出程序。")

    # MIDI预处理（确保返回有效bpm）
    try:
        beats_notes, ticks_per_beat, bpm = preprocess_midi(midi_file_path)
    except Exception as e:
        print(f"MIDI预处理失败: {e}")
        cleanup_fluidsynth()
        exit()

    # 强制初始化BPM（双重保障）
    safe_bpm = 120  # 默认安全值
    if not isinstance(bpm, (int, float)) or bpm <= 0:
        print(f"无效BPM值 {bpm}，使用默认值120")
        bpm = safe_bpm
    
    total_beats = len(beats_notes) if beats_notes else 0

    # 计算音符时值（添加异常保护）
    try:
        note_durations = calculate_note_durations(bpm)
    except Exception as e:
        print(f"计算音符时值失败: {e}")
        note_durations = calculate_note_durations(safe_bpm)  # 使用安全值重试

    # 初始化手势相关变量
    stop_detected = False
    prev_position = None
    current_beat = 0
    last_stop_time = None  # 初始化 last_stop_time

    try:
        while True:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧，退出程序。")
                break

            frame = cv2.flip(frame, 1)

            # 处理手势检测（添加返回值验证）
            try:
                result = process_frame_with_hand_detection(
                    frame, None, prev_position, stop_detected, 
                    current_beat, beats_notes, total_beats, last_stop_time
                )
                (prev_position, stop_detected, current_beat, 
                 last_stop_time, play_beat_command, current_bpm) = result
            except Exception as e:
                print(f"手势检测异常: {e}")
                current_bpm = bpm  # 使用全局BPM作为后备

            # 动态BPM有效性检查
            if not isinstance(current_bpm, (int, float)) or current_bpm <= 0:
                current_bpm = bpm
                print(f"无效动态BPM，恢复为全局BPM {bpm}")

            # 自动播放控制（添加参数验证）
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_result = hands.process(rgb_frame)
                handle_auto_play_control(
                    hand_result, 
                    beats_notes, 
                    current_bpm if current_bpm else bpm,  # 双重保障
                    volume
                )
            except Exception as e:
                print(f"自动播放控制异常: {e}")

            # 手动播放（添加参数验证）
            try:
                play_midi_beat_persistent(
                    beats_notes, 
                    play_beat_command if play_beat_command else False,  # 防止None
                    current_bpm if current_bpm else bpm,
                    volume, 
                    frame
                )
            except Exception as e:
                print(f"手动播放异常: {e}")

            cv2.imshow("Hand Gesture MIDI Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"程序运行时发生错误: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cleanup_fluidsynth()

if __name__ == '__main__':
    main()
    
