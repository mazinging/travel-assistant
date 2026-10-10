"""启动脚本：python run.py 后浏览器访问 http://127.0.0.1:8000"""
import uvicorn


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
