from main import app

handler = app

@app.get("/api/cron")
async def run_cron():
    return {"message": "Cron job executed successfully"}
