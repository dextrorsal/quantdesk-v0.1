"""
Authentication and Authorization Module
Provides JWT-based authentication for the trading system
"""
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel):
    """User model"""
    id: str
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime
    last_login: Optional[datetime] = None

class UserCreate(BaseModel):
    """User creation model"""
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str

class Token(BaseModel):
    """Token model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class AuthManager:
    """Authentication manager"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256", access_token_expire_minutes: int = 30):
        """
        Initialize authentication manager
        
        Args:
            secret_key: JWT secret key
            algorithm: JWT algorithm
            access_token_expire_minutes: Token expiration time
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        
        # In-memory user storage (replace with database in production)
        self.users: Dict[str, User] = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_password = "admin123"  # Change this in production!
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@quantify.com",
            hashed_password=self.hash_password(admin_password),
            is_active=True,
            is_admin=True,
            created_at=datetime.now()
        )
        self.users["admin"] = admin_user
        logger.info("Default admin user created: admin/admin123")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create new user"""
        # Check if user already exists
        if user_data.username in self.users:
            raise ValueError("Username already exists")
        
        # Create user
        user = User(
            id=secrets.token_urlsafe(16),
            username=user_data.username,
            email=user_data.email,
            hashed_password=self.hash_password(user_data.password),
            is_active=True,
            is_admin=False,
            created_at=datetime.now()
        )
        
        self.users[user.username] = user
        logger.info(f"User created: {user.username}")
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user"""
        user = self.users.get(username)
        if not user:
            logger.warning(f"Authentication failed: user {username} not found")
            return None
        
        if not user.is_active:
            logger.warning(f"Authentication failed: user {username} is inactive")
            return None
        
        if not self.verify_password(password, user.hashed_password):
            logger.warning(f"Authentication failed: invalid password for {username}")
            return None
        
        # Update last login
        user.last_login = datetime.now()
        
        logger.info(f"User authenticated: {username}")
        return user
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token"""
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "is_admin": user.is_admin,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # JWT ID for token tracking
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from token"""
        payload = self.verify_token(token)
        if not payload:
            return None
        
        username = payload.get("username")
        if not username:
            return None
        
        return self.users.get(username)
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh JWT token"""
        user = self.get_user_from_token(token)
        if not user:
            return None
        
        return self.create_access_token(user)

# Global authentication manager
auth_manager = None

def initialize_auth(secret_key: str):
    """Initialize authentication manager"""
    global auth_manager
    auth_manager = AuthManager(secret_key)
    logger.info("Authentication manager initialized")

def get_auth_manager() -> AuthManager:
    """Get authentication manager"""
    if auth_manager is None:
        raise RuntimeError("Authentication manager not initialized")
    return auth_manager

# Authentication decorators
def require_auth(func):
    """Decorator to require authentication"""
    async def wrapper(*args, **kwargs):
        # Get token from request
        request = kwargs.get('request')
        if not request:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authentication header")
        
        token = auth_header.split(" ")[1]
        
        # Verify token
        user = get_auth_manager().get_user_from_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Add user to kwargs
        kwargs['current_user'] = user
        
        return await func(*args, **kwargs)
    return wrapper

def require_admin(func):
    """Decorator to require admin privileges"""
    async def wrapper(*args, **kwargs):
        # First check authentication
        await require_auth(func)(*args, **kwargs)
        
        # Check admin privileges
        user = kwargs.get('current_user')
        if not user or not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        return await func(*args, **kwargs)
    return wrapper

# Utility functions
def get_current_user(request) -> Optional[User]:
    """Get current user from request"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    return get_auth_manager().get_user_from_token(token)

def is_authenticated(request) -> bool:
    """Check if request is authenticated"""
    return get_current_user(request) is not None

def is_admin(request) -> bool:
    """Check if request is from admin user"""
    user = get_current_user(request)
    return user is not None and user.is_admin
