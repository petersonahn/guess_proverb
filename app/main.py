from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.core.config import settings
from app.models.base import Base
from app.includes.dbconn import engine
from app.routers import health, questions, score
from app.routers import users

app = FastAPI(title="Proverb Game API", version="1.1")

# create tables (leaderboard only; proverb is assumed to exist already)
Base.metadata.create_all(bind=engine)

# Routers
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(questions.router, prefix=settings.API_PREFIX)
app.include_router(score.router, prefix=settings.API_PREFIX)
app.include_router(users.router, prefix=settings.API_PREFIX)

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/css", StaticFiles(directory="css"), name="css")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/rankings", response_class=HTMLResponse, include_in_schema=False)
def rankings(request: Request):
    return templates.TemplateResponse("rankings.html", {"request": request})
