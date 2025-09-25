"""
Encryption utilities for secure wallet storage.
Uses Fernet (symmetric encryption) from the cryptography library.
"""

import base64
import os
import json
from typing import Dict, Any, Union, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class WalletEncryption:
    """
    Handles encryption and decryption of wallet configuration files.
    Uses Fernet symmetric encryption with a key derived from a password.
    """
    
    def __init__(self, password: str = None):
        """
        Initialize encryption with a password or generate a new key.
        
        Args:
            password: Optional password for key derivation
        """
        self.password = password
        self.salt = None
        self.cipher_suite = None
        
    def _init_cipher(self, salt: bytes = None):
        """
        Initialize cipher suite with provided salt or generate new one
        
        Args:
            salt: Optional salt bytes. If not provided, generates new salt.
        """
        if salt is None:
            # Generate new salt for encryption
            self.salt = os.urandom(16)
        else:
            # Use provided salt for decryption
            self.salt = salt
            
        self.key = self._derive_key(self.password, self.salt)
        self.cipher_suite = Fernet(self.key)
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive an encryption key from a password and salt using PBKDF2.
        
        Args:
            password: Password to derive key from
            salt: Salt bytes for key derivation
            
        Returns:
            bytes: 32-byte key suitable for Fernet
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # High iteration count for security
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_wallet_config(self, config: Union[Dict[str, Any], List[int]]) -> bytes:
        """
        Encrypt a wallet configuration.
        
        Args:
            config: Wallet configuration (either dict or byte array)
            
        Returns:
            bytes: Encrypted configuration with salt prepended
        """
        try:
            if not self.password:
                raise ValueError("Password not set")
                
            # Initialize cipher with new salt
            self._init_cipher()
            
            # If config is a list of integers (byte array), convert to bytes
            if isinstance(config, list):
                config_bytes = bytes(config)
            else:
                # Convert dict to JSON string and encode
                config_bytes = json.dumps(config).encode()
                
            # Encrypt the bytes
            encrypted_data = self.cipher_suite.encrypt(config_bytes)
            # Prepend salt to encrypted data
            return self.salt + encrypted_data
        except Exception as e:
            logger.error(f"Failed to encrypt wallet config: {e}")
            raise
    
    def decrypt_wallet_config(self, data: bytes) -> Union[Dict[str, Any], List[int]]:
        """
        Decrypt an encrypted wallet configuration.
        
        Args:
            data: Encrypted configuration bytes with salt prepended
            
        Returns:
            Union[Dict[str, Any], List[int]]: Decrypted wallet configuration
        """
        try:
            if len(data) < 16:
                raise ValueError(f"Invalid encrypted data length: {len(data)}")
                
            if not self.password:
                raise ValueError("Password not set")
                
            # Extract salt from data
            salt = data[:16]
            encrypted_data = data[16:]
            
            # Initialize cipher with extracted salt
            self._init_cipher(salt)
            
            # Log debug information
            logger.info(f"Salt: {salt.hex()}")
            logger.info(f"Encrypted data length: {len(encrypted_data)}")
            
            # Decrypt the data
            try:
                decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                logger.info(f"Decryption successful, data length: {len(decrypted_data)}")
            except Exception as e:
                logger.error(f"Fernet decryption failed: {str(e)}")
                # Check if data is valid Fernet token format
                if not encrypted_data.startswith(b'gAAAAA'):
                    logger.error("Data does not appear to be in Fernet token format")
                raise
            
            # Try to parse as JSON first
            try:
                return json.loads(decrypted_data.decode())
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                # If not JSON, return as byte array
                return list(decrypted_data)
        except Exception as e:
            # Check if running in test environment
            is_test_env = os.environ.get("PYTEST_CURRENT_TEST") is not None
            log_func = logger.debug if is_test_env else logger.error
            log_func(f"Failed to decrypt wallet config: {str(e)}")
            raise
    
    def save_encrypted_config(self, config: Union[Dict[str, Any], List[int]], filepath: str) -> bool:
        """
        Save an encrypted wallet configuration to a file.
        
        Args:
            config: Wallet configuration to encrypt and save
            filepath: Path to save the encrypted file
            
        Returns:
            bool: True if save was successful
        """
        try:
            # Encrypt the config
            encrypted_data = self.encrypt_wallet_config(config)
            
            # Write to file
            with open(filepath, 'wb') as f:
                f.write(encrypted_data)
                
            return True
        except Exception as e:
            logger.error(f"Failed to save encrypted config: {e}")
            return False
            
    def load_encrypted_config(self, filepath: str) -> Union[Dict[str, Any], List[int]]:
        """
        Load and decrypt a wallet configuration from a file.
        
        Args:
            filepath: Path to the encrypted file
            
        Returns:
            Union[Dict[str, Any], List[int]]: Decrypted wallet configuration
        """
        try:
            # Read the encrypted data
            with open(filepath, 'rb') as f:
                encrypted_data = f.read()
                
            # Decrypt the data
            return self.decrypt_wallet_config(encrypted_data)
        except Exception as e:
            # Check if running in test environment
            is_test_env = os.environ.get("PYTEST_CURRENT_TEST") is not None
            log_func = logger.debug if is_test_env else logger.error
            log_func(f"Failed to load encrypted config from {filepath}: {str(e)}")
            raise 