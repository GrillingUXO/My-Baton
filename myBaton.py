import cv2
import numpy as np
import time
from tkinter import Tk, filedialog
from mido import MidiFile
from collections import deque
import fluidsynth
import threading
from threading import Lock


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
trajectory = deque(maxlen=30)  # 手势轨迹
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


def draw_rect(frame):
    """在帧中绘制用于初始化直方图的矩形区域。"""
    rows, cols, _ = frame.shape
    total_rectangle = 9  # 矩形数量

    # 定义每个矩形的起点和终点
    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 
         9 * rows / 20, 9 * rows / 20, 9 * rows / 20,
         12 * rows / 20, 12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 
         9 * cols / 20, 10 * cols / 20, 11 * cols / 20,
         9 * cols / 20, 10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    # 在每个矩形区域绘制绿色框
    for i in range(total_rectangle):
        cv2.rectangle(frame, 
                      (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def hand_histogram(frame):
    """通过指定的矩形框提取手部颜色的直方图。"""
    rows, cols, _ = frame.shape
    total_rectangle = 9

    # 定义矩形区域
    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20,
         9 * rows / 20, 9 * rows / 20, 9 * rows / 20,
         12 * rows / 20, 12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 
         9 * cols / 20, 10 * cols / 20, 11 * cols / 20,
         9 * cols / 20, 10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    # 从矩形区域提取 HSV 值
    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[
            hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
            hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    # 计算 HSV 直方图并归一化
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)



def hist_masking(frame, hist):
    """使用直方图对帧进行掩模操作，隔离手部区域。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 反向投影，生成概率图
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    # 平滑处理
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    # 二值化生成二进制掩模
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    thresh = cv2.merge((thresh, thresh, thresh))  # 转换为彩色图像掩模

    return cv2.bitwise_and(frame, thresh)


def contours(hist_mask_image):
    """找到手部区域的轮廓。"""
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def centroid(max_contour):
    """计算轮廓的质心（中心点）。"""
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
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

    
def initialize_hand_histogram():
    """Initialize the hand histogram for tracking using predefined rectangles."""
    print("请按 'z' 键来初始化手部直方图。按 'q' 键退出。")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        cleanup_fluidsynth()
        exit()

    global hand_hist
    is_hand_hist_created = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧。")
            break

        frame = cv2.flip(frame, 1)
        if not is_hand_hist_created:
            frame = draw_rect(frame)
            cv2.putText(frame, "Press 'z' to capture hand histogram", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Hand histogram initialized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Initialize Hand Histogram", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('z') and not is_hand_hist_created:
            hand_hist = hand_histogram(frame)
            is_hand_hist_created = True
            print("手部直方图已初始化。")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not is_hand_hist_created:
        print("未能成功初始化手部直方图。程序退出。")
        cleanup_fluidsynth()
        exit()

    return hand_hist




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
    Detect BPM using motion and pause gestures from the camera.
    Args:
        cap: OpenCV VideoCapture object.
        hand_hist: Histogram for hand detection.
    Returns:
        Detected BPM value.
    """
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

        # 检测手势并记录停顿
        if hand_hist is not None:
            masked_frame = hist_masking(frame, hand_hist)
            contours_list = contours(masked_frame)

            if contours_list:
                max_contour = max(contours_list, key=cv2.contourArea)
                if cv2.contourArea(max_contour) > 500:
                    cnt_centroid = centroid(max_contour)
                    if cnt_centroid is not None:
                        cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

                        if prev_position is not None:
                            dx = cnt_centroid[0] - prev_position[0]
                            dy = cnt_centroid[1] - prev_position[1]
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

                                    if len(tap_times) == 4:
                                        print("已完成测试小节，开始计算 BPM。")
                                        intervals = [tap_times[i + 1] - tap_times[i] for i in range(len(tap_times) - 1)]
                                        average_interval = sum(intervals) / len(intervals)
                                        bpm = max(60, 60 / average_interval)  # 防止 BPM 过低
                                        print(f"最终计算的 BPM：{bpm:.2f}")
                                        return bpm
                            else:
                                stop_detected = False

                        prev_position = cnt_centroid

        # 显示提示信息
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
    Args:
        centroid: 当前手势的质心位置。
        current_time: 当前时间戳。
        frame: 当前摄像头帧，用于绘制动画。
    """
    global last_stop_time, current_beat, bpm, note_durations, last_pause_info

    # 停顿检测逻辑
    if last_stop_time is not None:
        interval = current_time - last_stop_time
        if interval > 0:
            bpm = max(60, min(200, 60 / interval))  # 防止 BPM 过高或过低
            note_durations = calculate_note_durations(bpm)  # 更新音符时长
            print(f"动态更新 BPM：{bpm:.2f}")

    last_stop_time = current_time  # 更新上次停顿时间

    # 更新最后的停顿信息
    if "tap_times" in last_pause_info:
        last_pause_info["tap_times"].append(current_time)
    else:
        last_pause_info["tap_times"] = [current_time]
    last_pause_info["bpm"] = bpm

    # 显示动画：绘制停顿次数和时间戳
    for idx, tap_time in enumerate(last_pause_info["tap_times"][-4:], 1):  # 显示最近 4 次停顿
        cv2.putText(frame, f"Stop {idx}: {tap_time:.2f}s", (10, 60 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.putText(frame, f"BPM: {bpm:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 播放当前节拍音符（异步播放）
    if current_beat < len(beats_notes[0]):  # 假设所有声部节拍数一致
        play_midi_beat_persistent(beats_notes, current_beat, bpm, volume, frame)
        current_beat += 1
    else:
        print("所有节拍已播放完毕。")





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
    Play specified beat notes asynchronously using pyfluidsynth and update the frame with played notes.
    Args:
        beats_notes: List of voice-specific beats, each containing tuples (note, note_value, velocity).
        beat_index: Index of the current beat to play.
        bpm: Current BPM for playback.
        volume: Default volume for the notes.
        frame: Current camera frame, used for displaying played notes.
    """
    global fs, playback_thread, stop_playback, fluid_lock

    if fs is None:
        print("FluidSynth 未启动。")
        return

    if beat_index >= len(beats_notes[0]):  # 假设所有声部节拍数一致
        print("所有节拍已播放完毕。")
        return

    # 停止当前播放线程（如果存在）
    if playback_thread and playback_thread.is_alive():
        stop_playback = True
        playback_thread.join()  # 等待线程结束

    # 播放音符的函数
    def play_notes():
        global stop_playback
        stop_playback = False

        notes_to_display = []  # 用于显示在右下角的音符列表

        for voice_notes in beats_notes:
            if beat_index < len(voice_notes):  # 确保当前声部有此节拍
                for note, note_value, velocity in voice_notes[beat_index]:
                    if stop_playback:  # 检查是否需要中断播放
                        print("Playback interrupted.")
                        return

                    playback_duration = (60 / bpm) * (4 / note_value)  # 计算音符时值

                    with fluid_lock:  # 确保线程安全访问 FluidSynth
                        fs.noteon(0, note, int(velocity))  # 开始播放音符
                        time.sleep(playback_duration)      # 播放音符持续时间
                        fs.noteoff(0, note)               # 停止播放音符

                    print(f"Played note {note} with value {note_value:.2f}, velocity {velocity}, duration {playback_duration:.2f}s.")
                    notes_to_display.append(f"Note: {note}, Val: {note_value:.2f}, Vel: {velocity}")

                    # 动态更新帧中的音符显示
                    y_offset = frame.shape[0] - 20  # 起始高度
                    for idx, note_info in enumerate(notes_to_display[-10:]):  # 显示最近 10 个音符
                        cv2.putText(frame, note_info, (10, y_offset - idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 创建并启动新线程
    playback_thread = threading.Thread(target=play_notes)
    playback_thread.start()




# 主！逻！辑！
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头！")
    cleanup_fluidsynth()
    exit()

print("按 'q' 键退出程序。")

# 初始化手部直方图
hand_hist = initialize_hand_histogram()

# 初始化 MIDI 预处理
note_durations = calculate_note_durations(bpm)
beats_notes, ticks_per_beat = preprocess_midi(midi_file_path)
total_beats = len(beats_notes)

bpm_initialized = False  # 标志：BPM 初始化是否完成

# 初始化全局变量
stop_detected = False  # 初始化停顿检测标志
prev_position = None   # 初始化上一帧位置

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧，退出程序。")
            break

        frame = cv2.flip(frame, 1)  # 镜像翻转以获得正确视角

        # 阶段 1: 初始化 BPM
        if not bpm_initialized:
            cv2.putText(frame, "Initializing BPM, wave and stop to record...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            bpm = detect_bpm(cap, hand_hist)  # 调用 BPM 检测函数
            if bpm is not None:
                note_durations = calculate_note_durations(bpm)  # 更新音符时长
                bpm_initialized = True
                print(f"BPM 初始化完成，值为 {bpm:.2f}，进入 MIDI 播放流程。")
            else:
                print("BPM 初始化失败，程序退出。")
                break
            continue

        # 阶段 2: MIDI 播放流程
        if hand_hist is not None:
            masked_frame = hist_masking(frame, hand_hist)
            contours_list = contours(masked_frame)

            if contours_list:
                max_contour = max(contours_list, key=cv2.contourArea)
                if cv2.contourArea(max_contour) > 500:
                    cnt_centroid = centroid(max_contour)
                    if cnt_centroid is not None:
                        current_time = time.time()

                        # 绘制手势中心点
                        cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

                        # 调用通用停顿检测逻辑
                        if prev_position is not None:
                            stop_detected_local, motion_amplitude, stop_detected = detect_pause(
                                motion_amplitude, prev_position, cnt_centroid, stop_detected
                            )
                            if stop_detected_local:
                                print(f"检测到停顿：时间戳 {current_time:.2f}")
                                detect_pause_and_calculate_bpm(cnt_centroid, current_time, frame)

                                # 播放当前节拍音符
                                if current_beat < total_beats:
                                    play_midi_beat_persistent(beats_notes, current_beat, bpm, volume, frame)
                                    current_beat += 1

                        prev_position = cnt_centroid

        # 显示当前 BPM 和最近停顿信息
        if last_pause_info["bpm"] is not None:
            cv2.putText(frame, f"BPM: {last_pause_info['bpm']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            for idx, tap_time in enumerate(last_pause_info["tap_times"][-4:], 1):  # 显示最近 4 次停顿
                cv2.putText(frame, f"Stop {idx}: {tap_time:.2f}s", (10, 60 + idx * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # 显示摄像头帧
        cv2.imshow("Hand Gesture MIDI Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    cleanup_fluidsynth()


