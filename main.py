from app.schemas import ChatRequest
from fastapi import FastAPI
import uvicorn
import fastapi_cdn_host

app = FastAPI(title = "AI-Chat-Api")
fastapi_cdn_host.patch_docs(app)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/chat")
async def chat(data:ChatRequest):
    print(data)
    return {"reply": f"你好，用户{data.user_id}，你说的是: {data.message}"}

if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port = 8000)