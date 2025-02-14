from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt
import os

SECRET_KEY = os.getenv("JWT_SECRET_KEY")

class JWTMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        protected_routes = ["/uploadAudio"]

        if request.url.path in protected_routes:
            auth_header = request.headers.get("Authorization")

            try:
                if not auth_header:
                    raise HTTPException(status_code=400, detail="Authorization header is missing")
                
                if not auth_header.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="Invalid token format. It must start with 'Bearer '")

                token = auth_header.split(" ")[1]
        
                # Decode the token
                payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
                request.state.user = payload  # Store user data in request state
            
            except HTTPException as e:
                # Catch FastAPI exceptions and return as JSON response
                return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token expired. Please login again."})
            
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token. Authentication failed."})

            except Exception as e:
                return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error": str(e)})

        return await call_next(request)
