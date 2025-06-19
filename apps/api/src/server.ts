import { json, urlencoded } from "body-parser";
import express, { type Express } from "express";
import cors from "cors";

export const createServer = (): Express => {
  const app = express();
  app
    .disable("x-powered-by")
    .use(urlencoded({ extended: true }))
    .use(json())
    .use(cors({origin:"*"}))


  return app;
};
