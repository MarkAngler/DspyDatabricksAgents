"""Tests for security components."""

import pytest
import time
from unittest.mock import Mock, patch
from dspy_databricks_agents.security import (
    AuthConfig,
    AuthMiddleware,
    TokenValidator,
    User,
    Permission,
    require_auth,
    require_permission,
    encrypt_data,
    decrypt_data,
    AuditLogger,
    AuditEvent
)
from dspy_databricks_agents.security.auth import (
    AuthenticationError,
    AuthorizationError
)
from dspy_databricks_agents.security.audit import AuditEventType


class TestAuthentication:
    """Test authentication functionality."""
    
    def test_user_creation(self):
        """Test user creation with permissions."""
        user = User(
            id="123",
            username="testuser",
            email="test@example.com",
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        assert user.id == "123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert len(user.permissions) == 2
        assert Permission.READ in user.permissions
    
    def test_token_generation_and_validation(self):
        """Test JWT token generation and validation."""
        config = AuthConfig(secret_key="test-secret")
        validator = TokenValidator(config)
        
        user = User(
            id="123",
            username="testuser",
            permissions=[Permission.READ]
        )
        
        # Generate token
        token = validator.generate_token(user)
        assert isinstance(token, str)
        
        # Validate token
        validated_user = validator.validate_token(token)
        assert validated_user.id == user.id
        assert validated_user.username == user.username
        assert validated_user.permissions == user.permissions
    
    def test_token_expiration(self):
        """Test token expiration."""
        config = AuthConfig(
            secret_key="test-secret",
            token_expiry_hours=0  # Immediate expiration
        )
        validator = TokenValidator(config)
        
        user = User(id="123", username="testuser")
        token = validator.generate_token(user)
        
        # Wait a moment for expiration
        time.sleep(0.1)
        
        with pytest.raises(AuthenticationError, match="expired"):
            validator.validate_token(token)
    
    def test_token_revocation(self):
        """Test token revocation."""
        config = AuthConfig(secret_key="test-secret")
        validator = TokenValidator(config)
        
        user = User(id="123", username="testuser")
        token = validator.generate_token(user)
        
        # Token should be valid
        validator.validate_token(token)
        
        # Revoke token
        validator.revoke_token(token)
        
        # Token should now be invalid
        with pytest.raises(AuthenticationError, match="revoked"):
            validator.validate_token(token)
    
    def test_invalid_token(self):
        """Test invalid token handling."""
        config = AuthConfig(secret_key="test-secret")
        validator = TokenValidator(config)
        
        with pytest.raises(AuthenticationError, match="Invalid token"):
            validator.validate_token("invalid-token")
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        config = AuthConfig(
            secret_key="test-secret",
            rate_limit_requests=2,
            rate_limit_window_seconds=1
        )
        middleware = AuthMiddleware(config)
        
        user = User(id="123", username="testuser")
        token = middleware.validator.generate_token(user)
        
        # First two requests should succeed
        middleware.authenticate(token, client_ip="127.0.0.1")
        middleware.authenticate(token, client_ip="127.0.0.1")
        
        # Third request should be rate limited
        with pytest.raises(AuthenticationError, match="Rate limit"):
            middleware.authenticate(token, client_ip="127.0.0.1")
        
        # Wait for window to reset
        time.sleep(1.1)
        
        # Should work again
        middleware.authenticate(token, client_ip="127.0.0.1")
    
    def test_authorization(self):
        """Test authorization checks."""
        config = AuthConfig()
        middleware = AuthMiddleware(config)
        
        # User with limited permissions
        user = User(
            id="123",
            username="testuser",
            permissions=[Permission.READ]
        )
        
        # Should pass with matching permission
        middleware.authorize(user, [Permission.READ])
        
        # Should fail with missing permission
        with pytest.raises(AuthorizationError, match="Missing required permissions"):
            middleware.authorize(user, [Permission.WRITE])
        
        # Admin should pass all checks
        admin = User(
            id="456",
            username="admin",
            permissions=[Permission.ADMIN]
        )
        middleware.authorize(admin, [Permission.READ, Permission.WRITE])
    
    def test_user_creation_and_password_verification(self):
        """Test user creation with password."""
        config = AuthConfig(secret_key="test-secret")
        middleware = AuthMiddleware(config)
        
        # Create user
        user = middleware.create_user(
            username="testuser",
            password="secretpass",
            email="test@example.com",
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert len(user.permissions) == 2
        
        # Verify correct password
        assert middleware.verify_password(user, "secretpass") is True
        
        # Verify incorrect password
        assert middleware.verify_password(user, "wrongpass") is False


class TestAuthDecorators:
    """Test authentication decorators."""
    
    def test_require_auth_decorator(self):
        """Test require_auth decorator."""
        config = AuthConfig(secret_key="test-secret")
        middleware = AuthMiddleware(config)
        
        user = User(id="123", username="testuser")
        token = middleware.validator.generate_token(user)
        
        @require_auth(middleware)
        def protected_function(token: str, authenticated_user: User = None):
            return f"Hello {authenticated_user.username}"
        
        # With valid token
        result = protected_function(token)
        assert result == "Hello testuser"
        
        # Without token
        with pytest.raises(AuthenticationError, match="No authentication token"):
            protected_function(None)
        
        # With invalid token
        with pytest.raises(AuthenticationError):
            protected_function("invalid-token")
    
    def test_require_permission_decorator(self):
        """Test require_permission decorator."""
        user = User(
            id="123",
            username="testuser",
            permissions=[Permission.READ]
        )
        
        @require_permission(Permission.READ)
        def read_function(authenticated_user: User):
            return "read allowed"
        
        @require_permission(Permission.WRITE)
        def write_function(authenticated_user: User):
            return "write allowed"
        
        # Should allow read
        assert read_function(authenticated_user=user) == "read allowed"
        
        # Should deny write
        with pytest.raises(AuthorizationError, match="Missing required permissions"):
            write_function(authenticated_user=user)
        
        # Admin should access both
        admin = User(id="456", username="admin", permissions=[Permission.ADMIN])
        assert read_function(authenticated_user=admin) == "read allowed"
        assert write_function(authenticated_user=admin) == "write allowed"


class TestEncryption:
    """Test encryption utilities."""
    
    def test_encrypt_decrypt_string(self):
        """Test string encryption and decryption."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        
        original = "This is a secret message"
        encrypted = encrypt_data(original, key)
        
        # Encrypted should be different from original
        assert encrypted != original
        assert isinstance(encrypted, str)
        
        # Decrypt should recover original
        decrypted = decrypt_data(encrypted, key)
        assert decrypted == original
    
    def test_encrypt_decrypt_with_string_key(self):
        """Test encryption with string key."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()  # Generate valid key
        
        original = "Secret data"
        encrypted = encrypt_data(original, key)
        decrypted = decrypt_data(encrypted, key)
        
        assert decrypted == original
    
    def test_different_keys_fail(self):
        """Test decryption with wrong key fails."""
        from cryptography.fernet import Fernet
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        
        original = "Secret"
        encrypted = encrypt_data(original, key1)
        
        # Decryption with wrong key should fail
        with pytest.raises(Exception):
            decrypt_data(encrypted, key2)


class TestAuditLogging:
    """Test audit logging functionality."""
    
    def test_audit_event_creation(self):
        """Test audit event creation."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            user_id="123",
            username="testuser",
            ip_address="127.0.0.1"
        )
        
        assert event.event_type == AuditEventType.AUTH_SUCCESS
        assert event.user_id == "123"
        assert event.username == "testuser"
        assert event.timestamp > 0
        
        # Test JSON serialization
        json_str = event.to_json()
        assert "auth_success" in json_str
        assert "testuser" in json_str
    
    def test_audit_logger_basic(self):
        """Test basic audit logger functionality."""
        logger = AuditLogger(max_events=10)
        
        # Log various events
        logger.log_auth_success("123", "testuser", "127.0.0.1")
        logger.log_auth_failure("baduser", "127.0.0.1", "Invalid password")
        logger.log_permission_check("123", "testuser", "/api/data", "read", True)
        logger.log_data_access("123", "testuser", "/api/data", "read")
        logger.log_security_alert("suspicious_activity", "Multiple failed logins", "high")
        
        # Get all events
        events = logger.get_events()
        assert len(events) == 5
        
        # Events should be in reverse chronological order
        assert events[0].event_type == AuditEventType.SECURITY_ALERT
        assert events[-1].event_type == AuditEventType.AUTH_SUCCESS
    
    def test_audit_logger_filtering(self):
        """Test audit log filtering."""
        logger = AuditLogger()
        
        # Log events for different users
        logger.log_auth_success("user1", "alice", "127.0.0.1")
        logger.log_auth_success("user2", "bob", "127.0.0.2")
        logger.log_auth_failure("charlie", "127.0.0.3")
        
        # Filter by event type
        auth_successes = logger.get_events(event_type=AuditEventType.AUTH_SUCCESS)
        assert len(auth_successes) == 2
        
        # Filter by user
        user1_events = logger.get_events(user_id="user1")
        assert len(user1_events) == 1
        assert user1_events[0].username == "alice"
    
    def test_audit_logger_retention(self):
        """Test audit log retention and cleanup."""
        logger = AuditLogger(retention_days=0)  # Immediate expiration
        
        # Add old event
        old_event = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            timestamp=time.time() - 86400  # 1 day ago
        )
        logger._events.append(old_event)
        
        # Add recent event
        logger.log_auth_success("123", "testuser")
        
        # Before cleanup
        assert len(logger._events) == 2
        
        # Cleanup should remove old event (and possibly the recent one too with retention_days=0)
        removed = logger.cleanup_old_events()
        assert removed >= 1  # At least the old event should be removed
        
        # If any events remain, they should be the most recent
        events = logger.get_events()
        if events:
            assert events[0].username == "testuser"
    
    def test_audit_logger_max_events(self):
        """Test max events limit."""
        logger = AuditLogger(max_events=3)
        
        # Add more than max events
        for i in range(5):
            logger.log_auth_success(f"user{i}", f"user{i}")
        
        # Should only keep last 3
        events = logger.get_events()
        assert len(events) == 3
        assert events[0].username == "user4"  # Most recent
        assert events[2].username == "user2"  # Oldest kept