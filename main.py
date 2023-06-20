"""Main entrypoint for the app."""
import logging
import pickle
import asyncio
from pathlib import Path
from typing import Optional, Any

from deta import Deta, _Base
from fastapi import FastAPI, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import (
    QuestionGenCallbackHandler,
    StreamingLLMCallbackHandler,
    AsyncCallbackHandler,
)
from query_data import get_chain
from schemas import ChatResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


class InstaRepCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_rep(self, message: str, **kwargs: Any) -> None:
        # Make a nice scrolling effect by sending it word by word
        all_words = message.split(" ")
        for word in all_words:
            resp = ChatResponse(sender="bot", message=word + " ", type="stream")
            await self.websocket.send_json(resp.dict())
            await asyncio.sleep(0.05)


def get_db():
    deta = Deta()
    db = deta.Base("docs")
    yield db


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket, db: _Base = Depends(get_db)):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    insta_rep_handler = InstaRepCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(
        vectorstore, question_handler, insta_rep_handler, stream_handler
    )
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
