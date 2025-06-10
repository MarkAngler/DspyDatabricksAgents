"""Data encryption utilities."""

import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Union


def generate_key(password: str, salt: bytes = None) -> bytes:
    """Generate encryption key from password."""
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


def encrypt_data(data: Union[str, bytes], key: Union[str, bytes]) -> str:
    """Encrypt data with key."""
    if isinstance(data, str):
        data = data.encode()
    
    if isinstance(key, str):
        key = key.encode()
    
    f = Fernet(key)
    encrypted = f.encrypt(data)
    
    return base64.urlsafe_b64encode(encrypted).decode()


def decrypt_data(encrypted_data: str, key: Union[str, bytes]) -> str:
    """Decrypt data with key."""
    if isinstance(key, str):
        key = key.encode()
    
    f = Fernet(key)
    encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
    decrypted = f.decrypt(encrypted_bytes)
    
    return decrypted.decode()