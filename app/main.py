from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.models.base import Base
from app.includes.dbconn import engine
from app.routers import health, questions, score, users

app = FastAPI(title="Proverb Game API", version="1.2")

# DB 테이블 생성 (SQLAlchemy 모델 기준)
Base.metadata.create_all(bind=engine)

# API Routers (예: /api/health, /api/questions, /api/score, /api/users)
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(questions.router, prefix=settings.API_PREFIX)
app.include_router(score.router, prefix=settings.API_PREFIX)
app.include_router(users.router, prefix=settings.API_PREFIX)

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/css", StaticFiles(directory="css"), name="css")
templates = Jinja2Templates(directory="templates")

# ===== Pages =====
# 메인 진입점: / -> main.html
@app.get("/", response_class=HTMLResponse, include_in_schema=False, name="main")
def main_page(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

# 게임 페이지: /play -> index.html
@app.get("/play", response_class=HTMLResponse, include_in_schema=False, name="index")
def play_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 랭킹 페이지: /rankings -> rankings.html
@app.get("/rankings", response_class=HTMLResponse, include_in_schema=False, name="rankings")
def rankings_page(request: Request):
    return templates.TemplateResponse("rankings.html", {"request": request})

# ===== Aliases (과거 링크 호환/직접 접근 대비) =====
@app.get("/index", include_in_schema=False)
def index_alias():
    return RedirectResponse(url="/play")

@app.get("/index.html", include_in_schema=False)
def index_html_alias():
    return RedirectResponse(url="/play")

@app.get("/ranking.html", include_in_schema=False)
def ranking_html_alias():
    return RedirectResponse(url="/rankings")
