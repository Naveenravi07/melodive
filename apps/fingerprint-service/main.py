import pika
import warnings
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import numpy as np
import librosa
from scipy.signal.windows import hamming
from numpy.fft import rfft  

sys.path.insert(0, os.path.abspath("../../packages/proto/gen/py"))
from songs_pb2 import IndexingJob

warnings.filterwarnings('ignore')

# Constants
QUEUE_NAME = "songs"
MAX_WORKERS = 4

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def process_message(body):
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
    print("FFT example magnitudes for first frame:", fft_frames[0][:10]) 
    print("Done processing for", job.fileName)



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
