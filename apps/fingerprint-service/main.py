import sys, os
from concurrent.futures import ThreadPoolExecutor
import pika

sys.path.insert(0, os.path.abspath("../../packages/proto/gen/py"))
from songs_pb2 import IndexingJob

## Constants
QUEUE_NAME = "songs"
MAX_WORKERS = 4

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def process_message(body):
    job = IndexingJob()
    job.ParseFromString(body)
    print(job)


def callback(ch, method, properties, body):
    executor.submit(process_message,body)


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_consume(queue=QUEUE_NAME, auto_ack=True, on_message_callback=callback)
    
    print(" [*] Waiting for messages. To exit press CTRL+C")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Shutting down...")
        executor.shutdown(wait=True)



if __name__ == "__main__":
    main()
