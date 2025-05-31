from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

class BadRequestException(Exception):
    def __init__(self, message: str):
        self.message = message

class InternalServerException(Exception):
    def __init__(self, message: str):
        self.message = message

@app.exception_handler(BadRequestException)
async def bad_request_exception_handler(request: Request, exc: BadRequestException):
    return JSONResponse(
        status_code=400,
        content={"number": 400, "message": exc.message}
    )

@app.exception_handler(InternalServerException)
async def internal_server_exception_handler(request: Request, exc: InternalServerException):
    return JSONResponse(
        status_code=500,
        content={"number": 500, "message": exc.message}
    )