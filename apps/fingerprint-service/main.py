import os
import sys
import glob
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from psycopg2 import pool
import numpy as np
import librosa
from scipy.signal.windows import hamming
from numpy.fft import rfft
import io
import pyaudio
from collections import defaultdict

warnings.filterwarnings('ignore')
MAX_WORKERS = 4

'''
CREATE TABLE fingerprints (
    id SERIAL PRIMARY KEY,
    song_id INTEGER NOT NULL,
    hash_value BIGINT NOT NULL,
    anchor_time INTEGER NOT NULL
);
CREATE TABLE songs (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    filepath TEXT
);
'''


DB_USER = "postgres"
DB_NAME = "melodive"
DB_HOST = ""
DB_PORT = ""
DB_PASS = ""

SAMPLE_RATE = 11025
CHUNK_DURATION = 7
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1

AUDIO_EXTENSIONS = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac', '*.ogg', '*.wma']

FRAME_SIZE = 1024
HOP_SIZE = 512
NUM_BANDS = 6
TARGET_ZONE_FRAMES = 20

# New constants for recognition confidence thresholds
MIN_RECOGNITION_CONFIDENCE = 500 # Minimum number of consistent matches to consider it a strong candidate
MIN_RECOGNITION_TOTAL_MATCHES = 1000 # Minimum total matches for a strong candidate (prevents sparse high-confidence)


db_pool = pool.ThreadedConnectionPool(1, MAX_WORKERS, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT, dbname=DB_NAME)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def fingerprint_audio_file(file_path, file_name):
    conn = None
    try:
        print(f"Processing: {file_name}")

        y, sr = librosa.load(file_path)
        y = librosa.resample(y=y, orig_sr=sr, target_sr=SAMPLE_RATE)
        print(f"Resampled audio (min,max,type): {y.min():.3f}, {y.max():.3f}, {y.dtype}")

        hashes = generate_fingerprints(y)
        print(f"Total hashes generated: {len(hashes)}")

        if not hashes:
            print(f"No fingerprints generated for {file_name}")
            return False

        conn = db_pool.getconn()
        cur = conn.cursor()

        cur.execute("INSERT INTO songs (name, filepath) VALUES (%s, %s) RETURNING id",
                    (file_name, file_path))
        song_id = cur.fetchone()[0]

        fingerprints_data = [(song_id, int(hash_val), int(anchor_time))
                             for (hash_val, anchor_time) in hashes]

        output = io.StringIO()
        for fp in fingerprints_data:
            output.write(f"{fp[0]}\t{fp[1]}\t{fp[2]}\n")

        output.seek(0)
        cur.copy_from(output, 'fingerprints', columns=('song_id', 'hash_value', 'anchor_time'))
        conn.commit()

        print(f"Successfully fingerprinted: {file_name}")
        return True

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error processing {file_name}: {e}")
        return False
    finally:
        if conn:
            db_pool.putconn(conn)


def frame_signal(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::hop_size]
    return frames[:num_frames]


def find_audio_files(folder_path):
    audio_files = []
    for extension in AUDIO_EXTENSIONS:
        pattern = os.path.join(folder_path, '**', extension)
        files = glob.glob(pattern, recursive=True)
        audio_files.extend(files)
    return audio_files


def add_music_from_folder():
    folder_path = input("Enter the folder path containing music files: ").strip()

    if not os.path.exists(folder_path):
        print("Error: Folder path does not exist!")
        return

    if not os.path.isdir(folder_path):
        print("Error: Path is not a directory!")
        return

    print(f"Scanning for audio files in: {folder_path}")
    audio_files = find_audio_files(folder_path)

    if not audio_files:
        print("No audio files found in the specified folder!")
        return

    print(f"Found {len(audio_files)} audio files:")
    for file in audio_files:
        print(f"  - {os.path.basename(file)}")

    confirm = input(f"\nProceed to fingerprint {len(audio_files)} files? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return

    print(f"\nStarting multithreaded fingerprinting process with {MAX_WORKERS} workers...")

    tasks = []
    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        future = executor.submit(fingerprint_audio_file, file_path, file_name)
        tasks.append((future, file_name))

    successful = 0
    failed = 0

    for future, file_name in tasks:
        try:
            result = future.result()
            if result:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            failed += 1

    print(f"\nFingerprinting complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")


def generate_fingerprints(audio_data):
    try:
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        frames = frame_signal(audio_data, FRAME_SIZE, HOP_SIZE)
        if frames.shape[0] == 0:
            return []

        window = hamming(FRAME_SIZE, sym=False)
        windowed_frames = frames * window

        fft_frames = np.abs(rfft(windowed_frames, axis=1))

        BIN_COUNT = fft_frames.shape[1]
        BINS_PER_BAND = max(1, BIN_COUNT // NUM_BANDS)

        peaks = []
        for frame_idx, spectrum in enumerate(fft_frames):
            for band_idx in range(NUM_BANDS):
                start_bin = band_idx * BINS_PER_BAND
                end_bin = (band_idx + 1) * BINS_PER_BAND if band_idx < NUM_BANDS - 1 else BIN_COUNT

                band = spectrum[start_bin:end_bin]
                if len(band) > 0:
                    local_peak_idx = np.argmax(band)
                    peak_bin = start_bin + local_peak_idx
                    magnitude = band[local_peak_idx]
                    peaks.append((frame_idx, peak_bin, magnitude))

        hashes = []
        peaks.sort(key=lambda p: p[0])

        for i, (anchor_time, f1, _) in enumerate(peaks):
            for j in range(i + 1, len(peaks)):
                target_time, f2, _ = peaks[j]

                dt = target_time - anchor_time

                if dt > TARGET_ZONE_FRAMES:
                    break
                elif dt <= 0:
                    continue

                if f1 < 512 and f2 < 512 and dt < (1 << 14):
                    hash_val = (f1 << 23) | (f2 << 14) | dt
                    hashes.append((hash_val, anchor_time))

        return hashes
    except Exception as e:
        print(f"Error generating fingerprints: {e}")
        return []


def match_fingerprints(query_hashes):
    if not query_hashes:
        return {}

    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()

        query_hash_to_anchor = {int(h[0]): int(h[1]) for h in query_hashes}
        
        hash_values_to_query = list(query_hash_to_anchor.keys())
        
        if not hash_values_to_query:
            return {}

        cur.execute("""
            SELECT f.hash_value, f.song_id, f.anchor_time, s.name
            FROM fingerprints f
            JOIN songs s ON f.song_id = s.id
            WHERE f.hash_value = ANY(%s)
        """, (hash_values_to_query,))

        db_matches = cur.fetchall()

        song_raw_matches = defaultdict(list)
        for db_hash_val, song_id, db_anchor_time, song_name in db_matches:
            query_anchor_time = query_hash_to_anchor.get(db_hash_val)

            if query_anchor_time is not None:
                time_offset = db_anchor_time - query_anchor_time
                song_raw_matches[song_id].append({
                    'song_name': song_name,
                    'time_offset': time_offset,
                    'hash_value': db_hash_val
                })

        results = {}
        for song_id, matches_list in song_raw_matches.items():
            if len(matches_list) > 0:
                song_name = matches_list[0]['song_name']

                offset_counts = defaultdict(int)
                for match in matches_list:
                    offset_counts[match['time_offset']] += 1

                best_offset = max(offset_counts, key=offset_counts.get)
                confidence = offset_counts[best_offset]

                results[song_id] = {
                    'name': song_name,
                    'confidence': confidence,
                    'time_offset': best_offset,
                    'total_matches': len(matches_list)
                }

        return results

    except Exception as e:
        print(f"Database error during matching: {e}")
        return {}
    finally:
        if conn:
            db_pool.putconn(conn)


def scan_music():
    print("Music Recognition from Microphone")
    print("=" * 50)

    try:
        p = pyaudio.PyAudio()
    except Exception as e:
        print(f"Error initializing audio: {e}")
        print("Make sure you have PyAudio installed: pip install pyaudio")
        print("On some systems, you might also need portaudio development libraries.")
        return

    try:
        device_info = p.get_default_input_device_info()
        print(f"Using microphone: {device_info['name']}")
    except Exception as e:
        print(f"No microphone found: {e}")
        p.terminate()
        return

    print(f"Listening... (capturing {CHUNK_DURATION}s chunks)")
    print("Press Ctrl+C to stop")
    print("-" * 50)

    stream = p.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    try:
        chunk_count = 0
        while True:
            chunk_count += 1
            print(f"\nCapturing chunk #{chunk_count}...")

            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            print(f"Audio data shape: {audio_float.shape}, min: {audio_float.min():.3f}, max: {audio_float.max():.3f}")

            hashes = generate_fingerprints(audio_float)
            print(f"Generated {len(hashes)} fingerprints")

            if hashes:
                matches = match_fingerprints(hashes)

                if matches:
                    print("\nMATCHES FOUND:")
                    sorted_matches = sorted(matches.items(), key=lambda x: x[1]['confidence'], reverse=True)

                    rank_suffix = {
                        0: "1st probability",
                        1: "2nd probability",
                        2: "3rd probability"
                    }

                    for i, (song_id, match_info) in enumerate(sorted_matches[:3]):
                        confidence = match_info['confidence']
                        name = match_info['name']
                        total_matches = match_info['total_matches']

                        rank_text = rank_suffix.get(i, f"{i+1}th probability")

                        print(f"  {rank_text}: {name}")
                        print(f"    Confidence: {confidence} consistent matches")
                        print(f"    Total fingerprints matched: {total_matches}")

                        if confidence >= MIN_RECOGNITION_CONFIDENCE and total_matches >= MIN_RECOGNITION_TOTAL_MATCHES:
                            print(f"  HIGH CONFIDENCE MATCH!")
                            return
                else:
                    print("No matches found")
            else:
                print("No fingerprints generated from this audio chunk")

            print("-" * 50)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping recognition...")
    except Exception as e:
        print(f"Error during recognition: {e}")
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()
        print("Audio stream closed")


def show_menu():
    print("\n" + "="*50)
    print("MUSIC FINGERPRINTING SYSTEM")
    print("="*50)
    print("1. Scan Music (Recognition)")
    print("2. Add New Music (Fingerprint)")
    print("3. Exit")
    print("="*50)


def main():
    print("Welcome to Music Fingerprinting System!")

    while True:
        show_menu()
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            scan_music()
        elif choice == '2':
            add_music_from_folder()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    finally:
        if 'executor' in globals() and executor:
            executor.shutdown(wait=True)
        if 'db_pool' in globals() and db_pool:
            db_pool.closeall()
