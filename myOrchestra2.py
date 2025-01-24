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


# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)


rhythm_hand_label = None  # 节奏手的左右信息（"Left" 或 "Right"）
control_hand_label = None  # 变化手的左右信息（"Left" 或 "Right"）



last_distance_to_torso = None  # 记录变化手上一帧与躯干的距离



# 在全局变量定义区添加以下变量
pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
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

playback_thread = None
stop_playback = False


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



def process_frame_with_hand_detection(frame, hand_hist, prev_position, stop_detected, current_beat, beats_notes, total_beats):
    """
    使用 MediaPipe 检测两只手的动作并处理逻辑：
    1. 记录第一只出现的手是左手还是右手，并绑定为节奏手。
    2. 第二只出现的手绑定为变化手。
    3. 节奏手：检测滚动方向，并根据移动速度和移动距离触发 MIDI 音符播放。
    4. 变化手：根据手势和与躯干的相对距离调节音量。
    """
    global hands, mp_drawing, mp_hands, pose, bpm, motion_amplitude, last_pause_info, playback_thread, stop_playback
    global rhythm_hand_label, control_hand_label, velocity, last_volume_update_time, prev_time

    avg_motion = 0  # 初始化 avg_motion，防止未赋值错误
    recognized_hand_mouvement_scrolling = None  # 用于存储滚动方向的变量
    speed_threshold = 200  # 节奏手移动速度阈值（像素/秒）
    distance_threshold = 150  # 移动距离阈值（像素）

    # 使用 MediaPipe Pose 检测躯干
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb_frame)

    # 绘制躯干骨架
    torso_center = None  # 躯干中心点（肩膀中点）
    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        # 计算躯干中心（肩膀中点）
        left_shoulder = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        torso_center = (
            int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1]),
            int((left_shoulder.y + right_shoulder.y) / 2 * frame.shape[0]),
        )
        cv2.circle(frame, torso_center, 5, (0, 255, 0), -1)  # 绘制躯干中心点

    # 使用 MediaPipe Hands 检测手势
    hand_result = hands.process(rgb_frame)

    if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
        hand_landmarks_list = hand_result.multi_hand_landmarks
        handedness_list = hand_result.multi_handedness

        # 绑定节奏手和变化手
        if rhythm_hand_label is None or control_hand_label is None:
            for idx, handedness in enumerate(handedness_list):
                label = handedness.classification[0].label  # "Left" 或 "Right"

                if rhythm_hand_label is None:  # 第一只出现的手
                    rhythm_hand_label = label
                    print(f"节奏手绑定为：{label}")
                elif control_hand_label is None and label != rhythm_hand_label:  # 第二只手
                    control_hand_label = label
                    print(f"变化手绑定为：{label}")

        # 根据绑定的左右信息获取节奏手和变化手
        rhythm_hand = None
        control_hand = None
        for idx, handedness in enumerate(handedness_list):
            label = handedness.classification[0].label  # "Left" 或 "Right"
            if label == rhythm_hand_label:
                rhythm_hand = hand_landmarks_list[idx]
            elif label == control_hand_label:
                control_hand = hand_landmarks_list[idx]

        # 处理节奏手逻辑
        if rhythm_hand:
            # 绘制骨骼
            mp_drawing.draw_landmarks(frame, rhythm_hand, mp_hands.HAND_CONNECTIONS)

            # 获取节奏手手腕点位置
            wrist = rhythm_hand.landmark[mp_hands.HandLandmark.WRIST]
            wrist_pos = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))

            # 检测滚动方向
            if prev_position is not None:
                dx = wrist_pos[0] - prev_position[0]
                dy = wrist_pos[1] - prev_position[1]
                mouvement_distance = (dx**2 + dy**2)**0.5
                mouvement_distance_threshold = 0.02 * frame.shape[0]  # 基于画面高度调整阈值

                if mouvement_distance > mouvement_distance_threshold:
                    # 计算滚动角度
                    angle = math.degrees(math.atan2(dy, dx))
                    if -45 <= angle < 45:
                        recognized_hand_mouvement_scrolling = "Scrolling right"
                    elif 45 <= angle < 135:
                        recognized_hand_mouvement_scrolling = "Scrolling up"
                    elif angle >= 135 or angle < -135:
                        recognized_hand_mouvement_scrolling = "Scrolling left"
                    elif -135 <= angle < -45:
                        recognized_hand_mouvement_scrolling = "Scrolling down"
                    print(f"节奏手滚动方向: {recognized_hand_mouvement_scrolling}")

                    # 在画面上显示滚动方向
                    cv2.putText(frame, recognized_hand_mouvement_scrolling, (wrist_pos[0] + 20, wrist_pos[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 检测移动速度和距离并播放 MIDI 音符
            if prev_position is not None and prev_time is not None:
                current_time = time.time()
                delta_time = current_time - prev_time
                dx = wrist_pos[0] - prev_position[0]
                dy = wrist_pos[1] - prev_position[1]
                distance = (dx**2 + dy**2)**0.5
                speed = distance / delta_time  # 计算速度（像素/秒）

                # 当速度和距离均超过阈值时触发音符播放
                if speed > speed_threshold and distance > distance_threshold:
                    print(f"检测到速度: {speed:.2f}，移动距离: {distance:.2f}，触发 MIDI 音符播放。")
                    if 0 <= current_beat < total_beats:  # 确保 current_beat 在有效范围
                        play_midi_beat_persistent(beats_notes, current_beat, bpm, volume, frame)
                        current_beat += 1  # 更新节拍
                prev_time = current_time  # 更新时间戳
            else:
                prev_time = time.time()  # 初始化时间戳

            # 更新上一帧位置
            prev_position = wrist_pos

            # 标记节奏手
            cv2.putText(frame, "Rhythm Hand", (wrist_pos[0] - 50, wrist_pos[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 处理变化手逻辑（保持不变）
        if control_hand:
            # 绘制骨骼
            mp_drawing.draw_landmarks(frame, control_hand, mp_hands.HAND_CONNECTIONS)

            # 获取变化手手腕点位置
            control_wrist = control_hand.landmark[mp_hands.HandLandmark.WRIST]
            control_wrist_pos = (int(control_wrist.x * frame.shape[1]), int(control_wrist.y * frame.shape[0]))

            # 根据变化手和躯干的距离调整音量
            if torso_center:
                distance_to_torso = ((control_wrist_pos[0] - torso_center[0])**2 + 
                                     (control_wrist_pos[1] - torso_center[1])**2) ** 0.5

                current_time = time.time()
                if last_volume_update_time is None or current_time - last_volume_update_time > 0.2:
                    # 判断距离范围，调整音量
                    if distance_to_torso < 150:  # 靠近躯干
                        velocity = max(0, velocity - 5)
                        print(f"变化手靠近躯干，降低音量：{velocity}")
                    elif distance_to_torso > 250:  # 远离躯干
                        velocity = min(127, velocity + 5)
                        print(f"变化手远离躯干，提升音量：{velocity}")
                    # 更新时间戳
                    last_volume_update_time = current_time
                else:
                    print("音量维持不变。")

                # 显示手腕与躯干的实时距离
                cv2.putText(frame, f"Distance: {distance_to_torso:.2f}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 标记变化手
            cv2.putText(frame, "Control Hand", (control_wrist_pos[0] - 50, control_wrist_pos[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # 显示运动幅度和停顿状态
    cv2.putText(frame, f"Motion: {avg_motion:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 显示当前 BPM 和音量
    if last_pause_info["bpm"] is not None:
        cv2.putText(frame, f"BPM: {last_pause_info['bpm']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Velocity: {velocity}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return prev_position, stop_detected, current_beat  # 返回更新后的状态




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
    """Initialize the FluidSynth instance and load the SoundFont."""
    global fs, soundfont_path

    if fs is not None:
        print("FluidSynth 已经初始化。")
        return

    if not soundfont_path:
        print("SoundFont 文件路径未定义。")
        cleanup_fluidsynth()
        exit()

    try:
        fs = fluidsynth.Synth()
        fs.start(driver="coreaudio")  # Use "alsa", "dsound", etc., on other platforms
        sfid = fs.sfload(soundfont_path)
        fs.program_select(0, sfid, 0, 0)  # Select the first bank and program
        print("FluidSynth 初始化完成，并加载了 SoundFont 文件。")
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
    """Calculate durations for 16th, 8th, and 4th notes based on BPM."""
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




from mido import MidiFile, MetaMessage

import pretty_midi

def preprocess_midi(midi_file_path):
    """
    Parse a multi-voice MIDI file to extract notes, durations, and velocities for each beat and voice.
    Returns:
        - all_beats_notes: A list of lists, where each sublist corresponds to a voice's beats_notes.
        - ticks_per_beat: Number of ticks per beat (PPQN) in the MIDI file.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        print(f"成功加载 MIDI 文件: {midi_file_path}")
    except Exception as e:
        print(f"无法加载 MIDI 文件: {e}")
        cleanup_fluidsynth()
        exit()

    # Get tempo and ticks_per_beat
    if midi_data.get_tempo_changes()[0].size > 0:
        tempo = midi_data.get_tempo_changes()[1][0]  # Use the first tempo
    else:
        tempo = 120  # Default tempo if not provided
    bpm = tempo
    ticks_per_beat = midi_data.resolution  # Ticks per quarter note (PPQN)

    # Initialize all_beats_notes: One list for each instrument/voice
    all_beats_notes = []

    # Process each instrument separately
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue  # Skip drum tracks

        print(f"Processing instrument: {instrument.name if instrument.name else 'Unknown'}")

        # Initialize beats_notes for this instrument
        beats_notes = []
        for note in instrument.notes:
            # Calculate note duration in beats
            note_duration = (note.end - note.start) * bpm / 60  # Convert seconds to beats

            # Determine the note value (quantized to standard divisions)
            note_value = 4 / note_duration  # Inverse of duration in beats

            # Find the beat index where this note belongs
            start_beat = int(note.start * bpm / 60)  # Convert note start time to beat index
            while len(beats_notes) <= start_beat:
                beats_notes.append([])  # Ensure beats_notes has enough space

            # Append note to the corresponding beat
            beats_notes[start_beat].append((note.pitch, note_value, note.velocity))

        # Remove empty beats and add to all_beats_notes
        beats_notes = [beat for beat in beats_notes if beat]
        all_beats_notes.append(beats_notes)

    print(f"MIDI 文件原曲 BPM: {bpm:.2f}")
    print(f"预处理完成，共解析出 {len(all_beats_notes)} 个声部。")
    return all_beats_notes, ticks_per_beat



# Select and load MIDI and SoundFont files
midi_file_path = select_midi_and_soundfont_files()
start_fluidsynth()

# Initialize MIDI playback parameters
beats_notes, ticks_per_beat = preprocess_midi(midi_file_path)
note_durations = calculate_note_durations(bpm)


from queue import Queue
import threading


def detect_bpm(cap, hand_hist):
    """
    使用 MediaPipe 检测手部运动和停顿以初始化 BPM。
    Args:
        cap: OpenCV VideoCapture 对象。
        hand_hist: 保持参数一致，但此实现中不使用它。
    Returns:
        检测到的 BPM 值。
    """
    global hands, mp_drawing, mp_hands
    print("通过挥动和停顿初始化 BPM。开始挥动时，停顿以记录节拍。按 'q' 键退出。")
    
    motion_amplitude = []
    tap_times = []
    prev_position = None
    stop_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧。")
            return None

        frame = cv2.flip(frame, 1)  # 镜像翻转

        # 使用 MediaPipe Hands 检测手势
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # 绘制手部骨骼
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 获取手腕点位置
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_pos = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))

                # 检测手部运动
                if prev_position is not None:
                    dx = wrist_pos[0] - prev_position[0]
                    dy = wrist_pos[1] - prev_position[1]
                    distance = (dx**2 + dy**2)**0.5
                    motion_amplitude.append(distance)

                    if len(motion_amplitude) > 10:
                        motion_amplitude.pop(0)

                    avg_motion = np.mean(motion_amplitude)

                    if avg_motion < STOP_THRESHOLD:
                        if not stop_detected:
                            stop_detected = True
                            tap_time = time.time()
                            tap_times.append(tap_time)
                            print(f"记录停顿：第 {len(tap_times)} 次，时间戳：{tap_time:.2f}")

                            if len(tap_times) == 4:  # 如果记录到4次停顿，计算 BPM
                                print("已完成测试小节，开始计算 BPM。")
                                intervals = [tap_times[i + 1] - tap_times[i] for i in range(len(tap_times) - 1)]
                                average_interval = sum(intervals) / len(intervals)
                                bpm = max(60, 60 / average_interval)  # 限制最低 BPM 为 60
                                print(f"最终计算的 BPM：{bpm:.2f}")
                                return bpm
                    else:
                        stop_detected = False

                prev_position = wrist_pos

        # 显示当前停顿状态
        cv2.putText(frame, f"Recorded stops: {len(tap_times)}/4", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("BPM Initialization", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return None


note_durations = calculate_note_durations(bpm)
beats_notes, ticks_per_beat = preprocess_midi(midi_file_path)
total_beats = len(beats_notes)



last_pause_info = {"bpm": None, "tap_times": []}




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

def play_midi_beat_persistent(beats_notes, beat_index, bpm, volume, frame):
    """
    播放指定节拍的音符，并动态更新节拍索引。
    """
    global fs, playback_thread, stop_playback, fluid_lock

    if fs is None:
        print("FluidSynth 未启动。")
        return

    if beat_index >= len(beats_notes[0]):  # 确保当前节拍索引不越界
        print("所有节拍已播放完毕。")
        return

    # 停止当前播放线程（如果存在）
    if playback_thread and playback_thread.is_alive():
        stop_playback = True
        playback_thread.join()

    # 播放音符的函数
    def play_notes():
        global stop_playback
        stop_playback = False

        for voice_notes in beats_notes:
            if beat_index < len(voice_notes):  # 确保有对应节拍内容
                for note, note_value, velocity in voice_notes[beat_index]:
                    if stop_playback:  # 检查是否需要中断播放
                        print("Playback interrupted.")
                        return

                    # 调整音符音量和时值
                    adjusted_velocity = int(velocity * (volume / 127.0))
                    adjusted_velocity = max(0, min(127, adjusted_velocity))
                    playback_duration = (60 / bpm) * (4 / note_value)

                    with fluid_lock:
                        fs.noteon(0, note, adjusted_velocity)
                        time.sleep(playback_duration)
                        fs.noteoff(0, note)

                    print(f"Played note {note} with value {note_value:.2f}, velocity {adjusted_velocity}, duration {playback_duration:.2f}s.")

    # 启动新线程播放音符
    playback_thread = threading.Thread(target=play_notes)
    playback_thread.start()




def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        cleanup_fluidsynth()
        exit()

    print("按 'q' 键退出程序。")

    # 初始化阶段
    current_state = ProgramState.INITIALIZING_BPM

    # 初始化 MIDI 文件和播放相关变量
    note_durations = None
    beats_notes, ticks_per_beat = preprocess_midi(midi_file_path)
    total_beats = len(beats_notes[0]) if beats_notes else 0  # 确保存在有效的节拍数据

    # 初始化手势相关变量
    stop_detected = False
    prev_position = None
    current_beat = 0
    bpm = None

    try:
        while True:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧，退出程序。")
                break

            frame = cv2.flip(frame, 1)  # 镜像翻转以获得正确视角

            # 当前状态逻辑
            if current_state == ProgramState.INITIALIZING_BPM:
                bpm = detect_bpm(cap, None)  # 调用 BPM 初始化逻辑
                if bpm is None:
                    print("用户取消了 BPM 初始化，程序退出。")
                    break

                note_durations = calculate_note_durations(bpm)
                current_state = ProgramState.PLAYING_MIDI
                print(f"BPM 初始化完成，值为 {bpm:.2f}，进入 MIDI 播放流程。")

            elif current_state == ProgramState.PLAYING_MIDI:
                # 调用处理函数，并更新当前节拍状态
                prev_position, stop_detected, current_beat = process_frame_with_hand_detection(
                    frame, None, prev_position, stop_detected, current_beat, beats_notes, total_beats
                )

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
