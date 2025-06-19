import amqplib from 'amqplib'
import fs from 'fs/promises'
import {IndexingJob} from '@repo/proto/songs'

(async () => {
    const queue = 'songs';
    const songs_dir = '/home/shastri/Music'

    const conn = await amqplib.connect('amqp://localhost')
        .catch((err) => {
            console.error("Connection to rabbit mq failed", err)
            throw err
        })

    const ch1 = await conn.createChannel();
    await ch1.assertQueue(queue);

    try {
        const audioExt = [".mp3", ".m4a"]

        let dir = await fs.readdir(songs_dir, {
            withFileTypes: true,
            recursive: false,
        })

        for (const music of dir) {
            let name_split = music.name.split('.')
            let ext = '.'.concat(name_split[name_split.length - 1])

            if(!audioExt.includes(ext)) continue;
            console.log(music.name)

            let msg = IndexingJob.encode({fileName: music.name, filePath: music.parentPath}).finish()
            ch1.sendToQueue(queue,Buffer.from(msg))
            console.log(`Message Sent ${msg}`)
        }

    } catch (err) {
        console.error("Error occured during a file operation", err)
    }
})();
