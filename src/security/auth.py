"""Authentication and authorization middleware."""

import time
import logging
import hashlib
import secrets
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import jwt
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class User:
    """Authenticated user."""
    id: str
    username: str
    email: Optional[str] = None
    permissions: List[Permission] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthConfig:
    """Authentication configuration."""
    secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    token_expiry_hours: int = 24
    refresh_token_days: int = 30
    algorithm: str = "HS256"
    require_https: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when authorization fails."""
    pass


class TokenValidator:
    """JWT token validation."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self._revoked_tokens: set = set()
    
    def generate_token(self, user: User) -> str:
        """Generate JWT token for user."""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "permissions": [p.value for p in user.permissions],
            "exp": datetime.now(timezone.utc) + timedelta(hours=self.config.token_expiry_hours),
            "iat": datetime.now(timezone.utc),
        }
        
        token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
        logger.info(f"Generated token for user: {user.username}")
        return token
    
    def validate_token(self, token: str) -> User:
        """Validate JWT token and return user."""
        if token in self._revoked_tokens:
            raise AuthenticationError("Token has been revoked")
        
        try:
            payload = jwt.decode(
                token, 
                self.config.secret_key, 
                algorithms=[self.config.algorithm]
            )
            
            user = User(
                id=payload["user_id"],
                username=payload["username"],
                email=payload.get("email"),
                permissions=[Permission(p) for p in payload.get("permissions", [])]
            )
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    def revoke_token(self, token: str) -> None:
        """Revoke a token."""
        self._revoked_tokens.add(token)
        logger.info("Token revoked")


class RateLimiter:
    """Simple rate limiter for API protection."""
    
    def __init__(self, requests: int, window_seconds: int):
        self.requests = requests
        self.window_seconds = window_seconds
        self._attempts: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        # Clean old attempts
        if identifier in self._attempts:
            self._attempts[identifier] = [
                t for t in self._attempts[identifier] 
                if now - t < self.window_seconds
            ]
        
        # Check rate limit
        attempts = self._attempts.get(identifier, [])
        if len(attempts) >= self.requests:
            return False
        
        # Record attempt
        if identifier not in self._attempts:
            self._attempts[identifier] = []
        self._attempts[identifier].append(now)
        
        return True


class AuthMiddleware:
    """Authentication middleware for request handling."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.validator = TokenValidator(config)
        self.rate_limiter = RateLimiter(
            config.rate_limit_requests,
            config.rate_limit_window_seconds
        )
    
    def authenticate(self, token: str, client_ip: Optional[str] = None) -> User:
        """Authenticate request with token."""
        # Check rate limit
        if client_ip and not self.rate_limiter.is_allowed(client_ip):
            raise AuthenticationError("Rate limit exceeded")
        
        # Validate token
        user = self.validator.validate_token(token)
        
        logger.debug(f"Authenticated user: {user.username}")
        return user
    
    def authorize(self, user: User, required_permissions: List[Permission]) -> None:
        """Check if user has required permissions."""
        user_perms = set(user.permissions)
        required_perms = set(required_permissions)
        
        # Admin has all permissions
        if Permission.ADMIN in user_perms:
            return
        
        # Check required permissions
        if not required_perms.issubset(user_perms):
            missing = required_perms - user_perms
            raise AuthorizationError(
                f"Missing required permissions: {[p.value for p in missing]}"
            )
    
    def create_user(
        self, 
        username: str, 
        password: str,
        email: Optional[str] = None,
        permissions: Optional[List[Permission]] = None
    ) -> User:
        """Create a new user with hashed password."""
        # Generate user ID
        user_id = hashlib.sha256(username.encode()).hexdigest()[:16]
        
        # Hash password (in production, use proper password hashing like bcrypt)
        password_hash = hashlib.sha256(
            (password + self.config.secret_key).encode()
        ).hexdigest()
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            permissions=permissions or [Permission.READ],
            metadata={"password_hash": password_hash}
        )
        
        logger.info(f"Created user: {username}")
        return user
    
    def verify_password(self, user: User, password: str) -> bool:
        """Verify user password."""
        password_hash = hashlib.sha256(
            (password + self.config.secret_key).encode()
        ).hexdigest()
        
        return user.metadata.get("password_hash") == password_hash


# Decorators for protecting functions

def require_auth(auth_middleware: AuthMiddleware) -> Callable:
    """Decorator to require authentication."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract token from kwargs or first arg
            token = kwargs.get("token") or (args[0] if args else None)
            if not token:
                raise AuthenticationError("No authentication token provided")
            
            # Authenticate
            user = auth_middleware.authenticate(token)
            
            # Add user to kwargs
            kwargs["authenticated_user"] = user
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_permission(*permissions: Permission) -> Callable:
    """Decorator to require specific permissions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get authenticated user
            user = kwargs.get("authenticated_user")
            if not user:
                raise AuthenticationError("No authenticated user")
            
            # Check permissions
            user_perms = set(user.permissions)
            required_perms = set(permissions)
            
            if Permission.ADMIN not in user_perms and not required_perms.issubset(user_perms):
                missing = required_perms - user_perms
                raise AuthorizationError(
                    f"Missing required permissions: {[p.value for p in missing]}"
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator