from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv

from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
from botbuilder.schema import Activity

from config import Config
from bot import MCSHumanResourcesBot

_ = load_dotenv()

app = FastAPI()
bot = MCSHumanResourcesBot()

settings = BotFrameworkAdapterSettings(
    app_id=Config.APP_ID,
    app_password=Config.APP_PASSWORD,
    channel_auth_tenant=Config.TENANT_ID
)

adapter = BotFrameworkAdapter(settings)

@app.post("/")
async def messages(request: Request):
    body = await request.json()
    activity = Activity().deserialize(body)

    auth_header = request.headers.get("Authorization", "")

    async def call_bot(context):
        await bot.on_turn(context)
    
    await adapter.process_activity(activity, auth_header, call_bot)
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)