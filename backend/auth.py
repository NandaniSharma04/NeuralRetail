import os
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
import bcrypt
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Secret key to encode the JWT token
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-neural-retail-auth")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # bcrypt requires bytes; truncate to 72 bytes (bcrypt's hard limit)
    plain_bytes = plain_password.encode("utf-8")[:72]
    hashed_bytes = hashed_password.encode("utf-8") if isinstance(hashed_password, str) else hashed_password
    return bcrypt.checkpw(plain_bytes, hashed_bytes)

def get_password_hash(password: str) -> str:
    # Truncate to 72 bytes before hashing (bcrypt's hard limit)
    plain_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_bytes, salt).decode("utf-8")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
