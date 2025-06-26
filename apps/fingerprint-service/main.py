import pika
import io
import warnings
from concurrent.futures import ThreadPoolExecutor
from psycopg2 import pool
import os
import sys
import numpy as np
import psycopg2
import librosa
from scipy.signal.windows import hamming
from numpy.fft import rfft  

sys.path.insert(0, os.path.abspath("../../packages/proto/gen/py"))
from songs_pb2 import IndexingJob

warnings.filterwarnings('ignore')

# Constants
QUEUE_NAME = "songs"
MAX_WORKERS = 4

db_pool = pool.ThreadedConnectionPool(1, MAX_WORKERS,user="postgres.oktupobutqyfznkpnchf", dbname="postgres", host="aws-0-ap-south-1.pooler.supabase.com", port="6543", password="kPa0OVNUZfPA2NJe")
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def process_message(body):
    conn = None
    job: IndexingJob = IndexingJob()
    job.ParseFromString(body)

    print("Got a new job with fileName", job.fileName)

    y, sr = librosa.load(job.filePath)
    y = librosa.resample(y=y, orig_sr=sr, target_sr=11025)
    print("Resampled audio (min,max,type) :  ", y.min(), y.max(), y.dtype)

    FRAME_SIZE = 1024
    HOP_SIZE = 512

    frames = frame_signal(y, FRAME_SIZE, HOP_SIZE)
    print("Frame array shape before windowing:", frames.shape)

    # Apply Hamming window
    window = hamming(FRAME_SIZE, sym=False)  
    windowed_frames = frames * window
    print("Windowed frame shape =", windowed_frames.shape)

    # Apply FFT
    fft_frames = np.abs(rfft(windowed_frames, axis=1))
    print("FFT shape =", fft_frames.shape)

    # Peak picking
    NUM_BANDS = 6
    BIN_COUNT = fft_frames.shape[1]
    BINS_PER_BAND = BIN_COUNT // NUM_BANDS

    peaks = []

    for frame_idx, spectrum in enumerate(fft_frames):
        for band_idx in range(NUM_BANDS):
            start_bin = band_idx * BINS_PER_BAND
            end_bin = (band_idx + 1) * BINS_PER_BAND if band_idx < NUM_BANDS - 1 else BIN_COUNT

            band = spectrum[start_bin:end_bin]
            local_peak_idx = np.argmax(band)
            peak_bin = start_bin + local_peak_idx
            magnitude = band[local_peak_idx]

            peaks.append((frame_idx, peak_bin, magnitude))

    print("Total peaks extracted:", len(peaks))

    TARGET_ZONE_FRAMES = 20
    hashes = []
    peaks.sort(key=lambda p: p[0]) 

    for i, (anchor_time, f1, _) in enumerate(peaks):
        for j in range(i+1, len(peaks)):
            target_time, f2, _ = peaks[j]

            # Stop if the target peak is outside the target zone
            dt = target_time - anchor_time
            if dt > TARGET_ZONE_FRAMES:
                break
            elif dt <= 0:
                continue  # skip invalid

            # Encode the fingerprint into a 32-bit integer
            if f1 < 512 and f2 < 512 and dt < (1 << 14):  
                hash_val = (f1 << 23) | (f2 << 14) | dt
                hashes.append((hash_val, anchor_time)) 

    print("Total hashes generated:", len(hashes))
    
    try:
            conn = db_pool.getconn()
            cur = conn.cursor()

            cur.execute("INSERT INTO songs (name, filepath) VALUES (%s, %s) RETURNING id", (job.fileName, job.filePath))
            song_id = cur.fetchone()[0]
            fingerprints = [(song_id, int(hash_val), int(anchor_time)) for (hash_val, anchor_time) in hashes]
            
            output = io.StringIO()
            for fp in fingerprints:
                output.write(f"{fp[0]}\t{fp[1]}\t{fp[2]}\n") 
            
            output.seek(0) # Go to the beginning of the stream
            cur.copy_from(output, 'fingerprints', columns=('song_id', 'hash_value', 'anchor_time'))
            conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()
        print("DB error:", e)
    finally:
        if conn:
            db_pool.putconn(conn) 

    print("Fingerprinting complete for:", job.fileName)


def frame_signal(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::hop_size]
    return frames[:num_frames]


def callback(ch, method, properties, body):
    executor.submit(process_message, body)


def main():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters("localhost")
    )
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_consume(
        queue=QUEUE_NAME,
        auto_ack=True,
        on_message_callback=callback
    )

    print(" [*] Waiting for messages. To exit press CTRL+C")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Shutting down...")
        executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
