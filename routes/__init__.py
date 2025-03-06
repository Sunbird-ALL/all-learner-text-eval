from routes.v1_routes import v1_router
from routes.v2_routes import v2_router

from fastapi import APIRouter

router = APIRouter()
router.include_router(v1_router)
router.include_router(v2_router)
