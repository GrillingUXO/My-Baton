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

class ProgramState(Enum):
    INITIALIZING_BPM = 1
    PLAYING_MIDI = 2
    EXITING = 3


# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 新增全局变量
prev_palm_position = None  # 用于记录上一帧手掌位置
volume = 100  # 初始音量


fluid_lock = Lock()

STOP_THRESHOLD = 50
STOP_DURATION = 0.05
NOTE_INTERVAL = 0.1  # Interval between playing consecutive notes within a beat (seconds)

velocity = 64
volume = 64
bpm = 120  # Initial BPM based on test bar

prev_position = None
prev_time = None
last_stop_time = None
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

# 手部检测矩形配置
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None
hand_rect_two_x = None
hand_rect_two_y = None

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
    使用 MediaPipe 检测手腕点的停顿并处理节拍播放。
    """
    global hands, mp_drawing, mp_hands, bpm, motion_amplitude, last_pause_info

    avg_motion = 0  # 初始化 avg_motion，防止未赋值错误

    # 使用 MediaPipe 检测手势
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 绘制骨骼
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 获取手腕点（WRIST）位置
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_pos = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))

            # 检测手腕点运动和停顿
            if prev_position is not None:
                dx = wrist_pos[0] - prev_position[0]
                dy = wrist_pos[1] - prev_position[1]
                distance = (dx**2 + dy**2)**0.5
                motion_amplitude.append(distance)

                # 计算平均运动幅度
                if len(motion_amplitude) > 10:
                    motion_amplitude.pop(0)

                avg_motion = np.mean(motion_amplitude)

                if avg_motion < STOP_THRESHOLD:  # 检测到停顿
                    if not stop_detected:
                        stop_detected = True
                        current_time = time.time()
                        print(f"检测到停顿：时间戳 {current_time:.2f}")

                        # 播放当前节拍音符
                        if 0 <= current_beat < total_beats:  # 确保 current_beat 在有效范围
                            play_midi_beat_persistent(beats_notes, current_beat, bpm, volume, frame)
                            current_beat += 1  # 确保节拍正确递增

                        # 更新 BPM 和动画
                        detect_pause_and_calculate_bpm(wrist_pos, current_time, frame)
                else:
                    stop_detected = False

            # 更新上一帧位置
            prev_position = wrist_pos

    # 显示运动幅度和停顿状态
    cv2.putText(frame, f"Motion: {avg_motion:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 显示当前 BPM
    if last_pause_info["bpm"] is not None:
        cv2.putText(frame, f"BPM: {last_pause_info['bpm']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return prev_position, stop_detected, current_beat  # 返回更新后的 current_beat




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
