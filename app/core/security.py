"""Utilitários de hash de senha e geração/validação de JWT."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from passlib.context import CryptContext

from app.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"


def hash_password(senha_plana: str) -> str:
    return pwd_context.hash(senha_plana)


def verify_password(senha_plana: str, senha_hash: str) -> bool:
    return pwd_context.verify(senha_plana, senha_hash)


def create_access_token(
    *,
    subject: str,
    extra_claims: dict[str, Any] | None = None,
    expires_minutes: int | None = None,
) -> str:
    settings = get_settings()
    expire_delta = timedelta(
        minutes=expires_minutes if expires_minutes is not None else settings.jwt_expire_minutes
    )
    expire = datetime.now(timezone.utc) + expire_delta

    payload: dict[str, Any] = {
        "sub": subject,
        "exp": expire,
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
