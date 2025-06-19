import { createServer } from "./server";
import {logger} from "@repo/logger"

const port = process.env.PORT || 5001;
const server = createServer();

server.listen(port, () => {
    logger.info("API Gateway started")
});
