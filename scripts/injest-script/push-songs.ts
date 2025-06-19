import amqplib from 'amqplib'
import fs from 'fs/promises'

(async () => {
    const queue = 'songs';
    const songs_dir = '/home/shastri/music'

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

            ch1.sendToQueue(queue,)
        }

    } catch (err) {
        console.error("Error occured during a file operation", err)
    }
})();
