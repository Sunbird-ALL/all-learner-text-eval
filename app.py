import os
from fastapi import FastAPI
from routes import router
from middleware import JWTMiddleware
from routes.v1_routes import v1_router
from routes.v2_routes import v2_router

# Api doc url
app = FastAPI(
    title="Text Eval Service API",
    version="2.0",
    docs_url='/api/docs',
    openapi_url='/api/openapi.json'
)

# Add JWT middleware globally only for selected routes.
app.add_middleware(JWTMiddleware)

# v1 & v2 Routes
app.include_router(v1_router)
app.include_router(v2_router)

# Health check endpoint
@app.get("/ping")
async def health_check():
    return {
        "status": True,
        "message": "Text Eval Service is working"
    }

if __name__ == "__main__":
    import uvicorn
    num_workers = os.cpu_count() or 1
    uvicorn.run("app:app", host="0.0.0.0", port=5001, workers=num_workers)