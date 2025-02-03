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
from queue import Queue
midi_queue = Queue()


class ProgramState(Enum):
    INITIALIZING_BPM = 1
    PLAYING_MIDI = 2
    EXITING = 3



current_beat = 0                 # 当前播放的拍子序号
beat_lock = threading.Lock()     # 节拍计数器锁
global_time_signature = (4, 4)
prev_control_position = None


global_active_notes = {}  # 格式: {(channel, pitch): {"end_sec": float, "velocity": int}}
global_notes_lock = threading.Lock()



# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2, min_tracking_confidence=0.2)


rhythm_hand_label = None  # 节奏手的左右信息（"Left" 或 "Right"）
control_hand_label = None  # 变化手的左右信息（"Left" 或 "Right"）



last_distance_to_torso = None  # 记录变化手上一帧与躯干的距离

beat_lock = Lock()
playback_thread = None
stop_playback = False


# 在全局变量定义区添加以下变量
pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.2, min_tracking_confidence=0.2)
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

    
    global hands, mp_drawing, mp_hands, pose, bpm, motion_amplitude, last_pause_info, playback_thread, stop_playback
    global rhythm_hand_label, control_hand_label, velocity, last_volume_update_time, global_active_notes, global_notes_lock


    # 初始化控制信号变量
    play_beat_command = False      # 节拍播放触发标志
    current_bpm = bpm              # 当前计算的 BPM
    new_last_stop_time = last_stop_time  # 用于存储新的挥手时间

    speed_threshold = 100
    distance_threshold = 60
    MIN_INTERVAL = 0.1            # 最小挥手间隔（秒）

    # 静态变量：记录上一次挥手时间以及是否在等待手势放缓
    if not hasattr(process_frame_with_hand_detection, "last_swing_time"):
        process_frame_with_hand_detection.last_swing_time = None
    if not hasattr(process_frame_with_hand_detection, "waiting_for_rest"):
        process_frame_with_hand_detection.waiting_for_rest = False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb_frame)
    torso_center = None
    if pose_result.pose_landmarks:
        # mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        left_shoulder = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        torso_center = (
            int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1]),
            int((left_shoulder.y + right_shoulder.y) / 2 * frame.shape[0])
        )
        cv2.circle(frame, torso_center, 5, (0, 255, 0), -1)

    hand_result = hands.process(rgb_frame)
    if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
        hand_landmarks_list = hand_result.multi_hand_landmarks
        handedness_list = hand_result.multi_handedness

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

        # 节奏手逻辑
        if rhythm_hand:
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
                    # 此处仍保留滚动方向的绘制（若需要，可注释掉）
                    cv2.putText(frame, f"{angle:.1f}", (wrist_pos[0] + 20, wrist_pos[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 挥手触发检测
            if prev_position is not None:
                dx = wrist_pos[0] - prev_position[0]
                dy = wrist_pos[1] - prev_position[1]
                distance = (dx**2 + dy**2)**0.5
                if distance > distance_threshold and (distance > 0):
                    if not process_frame_with_hand_detection.waiting_for_rest:
                        with global_notes_lock:
                            active_non_cross = any(
                                not note_info.get("cross_beat", False)
                                for note_info in global_active_notes.values()
                            )
                        if active_non_cross:
                            play_beat_command = False
                        else:
                            if process_frame_with_hand_detection.last_swing_time is not None:
                                interval = time.time() - process_frame_with_hand_detection.last_swing_time
                                if interval >= MIN_INTERVAL:
                                    current_bpm = max(60, min(200, 60 / interval))
                                    print(f"检测到动态 BPM: {current_bpm:.2f}")
                                    play_beat_command = True
                                    new_last_stop_time = time.time()
                            else:
                                new_last_stop_time = time.time()
                            process_frame_with_hand_detection.last_swing_time = time.time()
                            process_frame_with_hand_detection.waiting_for_rest = True
                else:
                    process_frame_with_hand_detection.waiting_for_rest = False

            prev_position = wrist_pos
            cv2.circle(frame, wrist_pos, 8, (0, 255, 0), -1)

        if control_hand:
            control_wrist = control_hand.landmark[mp_hands.HandLandmark.WRIST]
            control_wrist_pos = (int(control_wrist.x * frame.shape[1]), int(control_wrist.y * frame.shape[0]))
            if torso_center:
                distance_to_torso = ((control_wrist_pos[0] - torso_center[0])**2 +
                                     (control_wrist_pos[1] - torso_center[1])**2)**0.5
                current_time = time.time()
                if last_volume_update_time is None or current_time - last_volume_update_time > 0.2:
                    if distance_to_torso < 80:
                        velocity = max(10, velocity - 5)
                    elif distance_to_torso > 1:
                        velocity = min(127, velocity + 5)
                    with fluid_lock:
                        fs.cc(0, 7, velocity)
                    last_volume_update_time = current_time
                cv2.putText(frame, f"Distance: {distance_to_torso:.2f}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.circle(frame, control_wrist_pos, 8, (0, 0, 255), -1)

    cv2.putText(frame, f"Play Command: {play_beat_command}", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"Velocity: {velocity}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return (
        prev_position,
        stop_detected,
        current_beat,
        new_last_stop_time if 'new_last_stop_time' in locals() else last_stop_time,
        play_beat_command,
        current_bpm
    )



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


def start_fluidsynth():
    global fs, soundfont_path, sfid

    if fs is not None:
        return

    try:
        fs = fluidsynth.Synth()
        fs.start(driver="coreaudio")
        
        # 加载 SoundFont
        sfid = fs.sfload(soundfont_path)
        
        # 初始化所有通道
        with fluid_lock:
            for channel in range(16):
                fs.program_select(channel, sfid, 0, 0)
        print(f"FluidSynth 初始化完成 | 复音数: 256")
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



def calculate_note_durations(bpm, time_signature=global_time_signature):

    # 取拍号分母
    numerator, denominator = time_signature
    beat_duration = (60 / bpm) * (4 / denominator)
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
    返回值：all_voices_notes, ticks_per_beat, bpm  
    同时提取拍号信息，保存到全局变量 global_time_signature  
    为每个声部新增 'next_note_index' 用于记录下一次要播放的音符索引。
    """
    global sfid, global_time_signature

    try:
        # 使用 pretty_midi 解析 MIDI 文件
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        bpm = midi_data.estimate_tempo()  # 获取估计的 BPM
        ticks_per_beat = midi_data.resolution  # 获取每拍的 ticks 数

        # 提取拍号信息（取第一个拍号事件，如果存在）
        if midi_data.time_signature_changes:
            ts = midi_data.time_signature_changes[0]
            global_time_signature = (ts.numerator, ts.denominator)
            print(f"检测到拍号: {ts.numerator}/{ts.denominator}")
        else:
            print("未检测到拍号信息，默认 4/4")

        all_voices_notes = []

        # 遍历每个乐器（声部）
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue  # 跳过鼓轨道

            channel = instrument.program % 16
            target_program = instrument.program % 128

            with fluid_lock:
                try:
                    fs.program_select(channel, sfid, 0, target_program)
                    print(f"通道{channel} 设置成功: Bank 0, Program {target_program}")
                except fluidsynth.FluidError:
                    try:
                        fs.program_select(channel, sfid, 128, target_program)
                        print(f"通道{channel} 使用Bank 128, Program {target_program}")
                    except fluidsynth.FluidError:
                        fs.program_select(channel, sfid, 0, 0)
                        print(f"通道{channel} 音色不可用，已回退到钢琴")

            voice_notes = {
                "name": instrument.name if instrument.name else "Unnamed",
                "program": channel,
                "notes": [],
                "original_bpm": bpm,
                "next_note_index": 0  # 新增字段，用于跟踪本声部下一次要播放的音符
            }

            # 收集原始音符数据（以秒为单位）
            for note in instrument.notes:
                voice_notes["notes"].append({
                    "pitch": note.pitch,
                    "start_sec": note.start,
                    "end_sec": note.end,
                    "velocity": note.velocity,
                    "duration_sec": note.end - note.start
                })
            # 对音符列表按起始时间排序，确保后续按顺序播放
            voice_notes["notes"].sort(key=lambda n: n["start_sec"])
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
    STOP_THRESHOLD = 30  # 停顿运动幅度阈值
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



def midi_event_processor():
    """
    修改后的 MIDI 事件处理线程：
      - 处理 note_on 和 note_off 事件；
      - 所有 note_off 事件都将被正常执行，确保跨拍音符在到达预定时长后能被关闭，
        避免出现无限延长的问题。
    """
    global fs, stop_playback, fluid_lock, global_active_notes, global_notes_lock, midi_queue, current_beat
    while True:
        event = midi_queue.get()  # 阻塞等待事件
        with fluid_lock:
            try:
                if event["type"] == "note_on":
                    fs.noteon(event["channel"], event["pitch"], event["velocity"])
                    with global_notes_lock:
                        global_active_notes[(event["channel"], event["pitch"])] = {
                            "cross_beat": event.get("cross_beat", False),
                            "beat": event.get("beat")
                        }
                elif event["type"] == "note_off":
                    fs.noteoff(event["channel"], event["pitch"])
                    with global_notes_lock:
                        key = (event["channel"], event["pitch"])
                        if key in global_active_notes:
                            del global_active_notes[key]
            except fluidsynth.FluidError as e:
                print(f"FluidSynth Error in midi_event_processor: {e}")
        midi_queue.task_done()




def panic_non_cross_notes():
    """
    关闭所有正在播放的非跨拍音符，
    使得当新的 play beat 指令触发时，仅中断上一拍中已启动且不跨拍的音符，
    而跨拍音符依然可以继续播放。
    """
    global fs, global_active_notes, global_notes_lock, fluid_lock
    with fluid_lock:
        with global_notes_lock:
            keys_to_remove = []
            for key, note_info in global_active_notes.items():
                if not note_info.get("cross_beat", False):
                    channel, pitch = key
                    try:
                        fs.noteoff(channel, pitch)
                    except fluidsynth.FluidError as e:
                        print(f"Error turning off note {key}: {e}")
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del global_active_notes[key]


import threading

def play_midi_beat_persistent(all_voices_notes, play_beat_command, current_bpm, volume, frame):
    global fs, playback_thread, stop_playback, current_beat, interrupt_flag, fluid_lock
    global global_active_notes, global_notes_lock, midi_queue, beat_lock, global_time_signature
    global global_playback_start_time  # 用于记录当前播放段起始时间

    if not play_beat_command or fs is None:
        return

    # 如果已有播放线程仍在运行，则判断是否允许打断当前播放段
    if playback_thread and playback_thread.is_alive():
        # 根据当前 BPM 计算当前拍（beat）的时长
        _, denominator = global_time_signature
        beat_duration = (60.0 / current_bpm) * (4 / denominator)
        # 后25%的时长允许打断
        allowed_interrupt_offset = beat_duration * 0.75
        # 计算当前播放段已进行的时长
        if 'global_playback_start_time' not in globals() or global_playback_start_time is None:
            current_progress = 0
        else:
            current_progress = time.perf_counter() - global_playback_start_time
        # 如果当前播放进度尚未进入最后25%，则不允许打断
        if current_progress < allowed_interrupt_offset:
            return

        # 达到或超过后25%时，允许打断：中断当前播放线程并关闭所有非跨拍音符
        stop_playback = True
        try:
            playback_thread.join(timeout=0.05)
        except RuntimeError:
            pass
        finally:
            panic_non_cross_notes()

    # 更新拍子相关信息（使用 beat_lock 保护对 current_beat 的更新）
    with beat_lock:
        current_beat += 1
        interrupt_flag = True
        _, denominator = global_time_signature
        beat_duration = (60.0 / current_bpm) * (4 / denominator)
        start_time = (current_beat - 1) * beat_duration
        end_time = current_beat * beat_duration

    def play_notes():
        global stop_playback, interrupt_flag, midi_queue, current_beat, global_playback_start_time
        stop_playback = False
        interrupt_flag = False
        events = []
        try:
            # 遍历所有声部，收集当前拍内的音符事件
            for voice in all_voices_notes:
                program = voice["program"]
                notes = voice["notes"]
                idx = voice["next_note_index"]
                # 处理从 idx 开始的音符（音符列表已排序）
                while idx < len(notes):
                    note = notes[idx]
                    note_start = note["start_sec"]
                    note_end = note["end_sec"]
                    if note_end <= note_start:
                        idx += 1
                        continue
                    # 跳过当前拍之前的音符
                    if note_start < start_time:
                        idx += 1
                        continue
                    # 仅处理落在当前拍内的音符
                    if start_time <= note_start < end_time:
                        if note_end > end_time:
                            # 跨拍音符：按当前 BPM 重新计算时值
                            new_duration = note["duration_sec"] * (current_bpm / 60)
                            note_off_time = note_start + new_duration
                            cross_flag = True
                        else:
                            note_off_time = note_end
                            cross_flag = False
                        events.append({
                            "type": "note_on",
                            "time": note_start,
                            "pitch": note["pitch"],
                            "velocity": int(note["velocity"] * (volume / 127.0)),
                            "channel": program,
                            "cross_beat": cross_flag,
                            "beat": current_beat
                        })
                        events.append({
                            "type": "note_off",
                            "time": note_off_time,
                            "pitch": note["pitch"],
                            "channel": program,
                            "cross_beat": cross_flag,
                            "beat": current_beat
                        })
                        idx += 1
                    else:
                        # 后续音符不在当前拍内
                        break
                # 更新该声部的 next_note_index
                voice["next_note_index"] = idx

            # 按时间顺序排序所有事件
            events.sort(key=lambda x: x["time"])
            local_playback_start = time.perf_counter()
            # 记录播放段起始时间，用于打断判断
            global_playback_start_time = local_playback_start

            # 逐个调度事件
            for event in events:
                event_offset = event["time"] - start_time
                while (time.perf_counter() - local_playback_start) < event_offset:
                    if (stop_playback or interrupt_flag) and event["type"] == "note_on":
                        break
                    time.sleep(0.01)
                if (stop_playback or interrupt_flag) and event["type"] == "note_on":
                    continue
                midi_queue.put(event)
        finally:
            pass

    playback_thread = threading.Thread(target=play_notes, daemon=True)
    playback_thread.start()




def trigger_tuning():
    """
    播放调音音：所有通道同时播放 A4（69），持续 1 秒，
    然后发送 note_off 消除声音。
    """
    global fs, fluid_lock
    # 播放 A4 音
    with fluid_lock:
        for channel in range(16):
            try:
                fs.noteon(channel, 69, 127)
            except Exception as e:
                print(f"Error in tuning noteon on channel {channel}: {e}")
    # 持续 1 秒
    time.sleep(2)
    # 关闭 A4 音
    with fluid_lock:
        for channel in range(16):
            try:
                fs.noteoff(channel, 69)
            except Exception as e:
                print(f"Error in tuning noteoff on channel {channel}: {e}")
    print("Tuning triggered: A4 played for 1 second.")


def check_and_trigger_tuning(frame):
    """
    检测画面中是否同时出现节奏手和变化手，并满足如下条件：
      - 两只手腕都在躯干中点以上（躯干中点取左右肩膀中点）
      - 节奏手的腕部比变化手更高（图像中 y 坐标更小）
    如果满足条件，则触发一次调音（播放 A4 1秒）。
    此操作仅能触发一次。
    """
    global tuning_triggered, rhythm_hand_label, control_hand_label, hands, mp_hands, pose

    if tuning_triggered:
        return

    # 将图像转为 RGB 格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理姿态，计算躯干中点（左右肩膀的中点）
    pose_result = pose.process(rgb_frame)
    torso_center = None
    if pose_result.pose_landmarks:
        left_shoulder = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        torso_center = (
            int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1]),
            int((left_shoulder.y + right_shoulder.y) / 2 * frame.shape[0])
        )
    else:
        return  # 无法检测躯干时不触发

    # 检测手部
    hand_result = hands.process(rgb_frame)
    if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
        hand_landmarks_list = hand_result.multi_hand_landmarks
        handedness_list = hand_result.multi_handedness

        # 若全局变量未设置，则按出现顺序绑定节奏手和变化手
        if rhythm_hand_label is None or control_hand_label is None:
            for idx, handedness in enumerate(handedness_list):
                label = handedness.classification[0].label
                if rhythm_hand_label is None:
                    rhythm_hand_label = label
                elif control_hand_label is None and label != rhythm_hand_label:
                    control_hand_label = label

        rhythm_hand = None
        control_hand = None
        for idx, handedness in enumerate(handedness_list):
            label = handedness.classification[0].label
            if label == rhythm_hand_label:
                rhythm_hand = hand_landmarks_list[idx]
            elif label == control_hand_label:
                control_hand = hand_landmarks_list[idx]

        if rhythm_hand and control_hand:
            # 获取两只手腕的图像坐标
            rhythm_wrist = rhythm_hand.landmark[mp_hands.HandLandmark.WRIST]
            control_wrist = control_hand.landmark[mp_hands.HandLandmark.WRIST]
            rhythm_wrist_pos = (int(rhythm_wrist.x * frame.shape[1]), int(rhythm_wrist.y * frame.shape[0]))
            control_wrist_pos = (int(control_wrist.x * frame.shape[1]), int(control_wrist.y * frame.shape[0]))

            # 检查两只手腕是否都在躯干中点以上（y 坐标小于 torso_center[1]）
            # 且节奏手更高（节奏手的 y 坐标小于变化手的 y 坐标）
            if (rhythm_wrist_pos[1] < torso_center[1] and 
                control_wrist_pos[1] < torso_center[1] and 
                rhythm_wrist_pos[1] < control_wrist_pos[1]):
                tuning_triggered = True
                # 在独立线程中触发调音，避免阻塞主循环
                threading.Thread(target=trigger_tuning, daemon=True).start()


def main():
    global tuning_triggered
    tuning_triggered = False  # 用于确保调音只触发一次

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        cleanup_fluidsynth()
        exit()

    print("按 'q' 键退出程序。")

    beats_notes, ticks_per_beat, bpm = preprocess_midi(midi_file_path)
    total_beats = len(beats_notes) if beats_notes else 0  # 确保存在有效的节拍数据

    # 使用 MIDI 文件原曲 BPM 初始化
    bpm = None  # 初始化 BPM 为 None，确保我们明确地从 MIDI 获取
    if beats_notes:  # 如果解析成功，使用原曲 BPM
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)
            tempo_changes = midi_data.get_tempo_changes()
            if tempo_changes[1].size > 0:  # 检测到 BPM 数据
                bpm = tempo_changes[1][0]
                print(f"原曲 BPM 初始化为：{bpm:.2f}")
            else:
                print("未检测到原曲 BPM，程序无法继续。")
                cleanup_fluidsynth()
                exit()
        except Exception as e:
            print(f"无法读取原曲 BPM，程序无法继续，错误信息：{e}")
            cleanup_fluidsynth()
            exit()

    # 计算音符时值
    print(f"使用原曲 BPM：{bpm:.2f}")
    note_durations = calculate_note_durations(bpm)

    # 初始化手势相关变量
    stop_detected = False
    prev_position = None
    current_beat = 0
    last_stop_time = None  # 初始化 last_stop_time

    # 启动 MIDI 事件处理线程
    midi_thread = threading.Thread(target=midi_event_processor, daemon=True)
    midi_thread.start()

    try:
        while True:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧，退出程序。")
                break

            frame = cv2.flip(frame, 1)  # 镜像翻转以获得正确视角
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.7), int(frame.shape[0] * 0.7)))

            check_and_trigger_tuning(frame)

            # MIDI 播放流程
            prev_position, stop_detected, current_beat, last_stop_time, play_beat_command, current_bpm = process_frame_with_hand_detection(
                frame, None, prev_position, stop_detected, current_beat, beats_notes, total_beats, last_stop_time
            )

            # 动态播放 MIDI 音符
            play_midi_beat_persistent(beats_notes, play_beat_command, current_bpm, volume, frame)

            # 显示摄像头帧
            cv2.imshow("Hand Gesture MIDI Control", frame)

            # 退出程序
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
    
