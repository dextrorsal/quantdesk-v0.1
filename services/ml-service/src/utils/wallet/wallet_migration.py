"""
Wallet configuration migration utility.
Helps transition existing wallet configurations to encrypted storage.
"""

import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from src.utils.wallet.encryption import WalletEncryption

logger = logging.getLogger(__name__)

class WalletMigration:
    """
    Handles migration of wallet configurations between formats
    and provides backup functionality.
    """
    
    def __init__(self, config_dir: str = "config/wallets"):
        """
        Initialize migration utility
        
        Args:
            config_dir: Directory containing wallet configurations
        """
        self.config_dir = Path(config_dir)
        self.backup_dir = self.config_dir / "backup"
        
    def backup_existing_configs(self) -> bool:
        """
        Create backup of all existing wallet configurations
        
        Returns:
            bool: True if backup was successful
        """
        try:
            # Create backup directory
            backup_timestamp = Path(f"backup_{int(time.time())}")
            backup_path = self.backup_dir / backup_timestamp
            os.makedirs(backup_path, exist_ok=True)
            
            # Copy all wallet configs to backup
            for config_file in self.config_dir.glob("*.json"):
                backup_file = backup_path / config_file.name
                shutil.copy2(config_file, backup_file)
                
            logger.info(f"Created backup at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def migrate_to_encrypted(self, password: str) -> Tuple[int, int]:
        """
        Migrate unencrypted wallet configs to encrypted format
        
        Args:
            password: Password for encryption
            
        Returns:
            Tuple[int, int]: (number of successful migrations, total attempted)
        """
        try:
            # Initialize encryption
            encryption = WalletEncryption(password)
            
            # Track migration stats
            successful = 0
            total = 0
            
            # Create backup first
            if not self.backup_existing_configs():
                raise Exception("Failed to create backup, aborting migration")
            
            # Migrate each unencrypted config
            for config_file in self.config_dir.glob("*.json"):
                total += 1
                try:
                    # Read existing config
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Create encrypted version
                    enc_path = config_file.with_suffix('.enc')
                    if encryption.save_encrypted_config(config, str(enc_path)):
                        # Remove old unencrypted file only after successful encryption
                        config_file.unlink()
                        successful += 1
                        logger.info(f"Migrated {config_file.name} to encrypted storage")
                    else:
                        logger.error(f"Failed to encrypt {config_file.name}")
                        
                except Exception as e:
                    logger.error(f"Failed to migrate {config_file.name}: {e}")
                    
            return successful, total
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return 0, 0
    
    def verify_migration(self, password: str) -> List[str]:
        """
        Verify all encrypted wallet configurations can be decrypted
        
        Args:
            password: Password for decryption
            
        Returns:
            List[str]: List of any problematic wallet names
        """
        encryption = WalletEncryption(password)
        problems = []
        
        for config_file in self.config_dir.glob("*.enc"):
            try:
                # Try to decrypt
                config = encryption.load_encrypted_config(str(config_file))
                if not all(k in config for k in ['keypair_path', 'public_key']):
                    problems.append(config_file.stem)
            except Exception:
                problems.append(config_file.stem)
                
        return problems
    
    def restore_backup(self, backup_timestamp: str) -> bool:
        """
        Restore wallet configurations from a backup
        
        Args:
            backup_timestamp: Timestamp of backup to restore
            
        Returns:
            bool: True if restore was successful
        """
        try:
            backup_path = self.backup_dir / f"backup_{backup_timestamp}"
            if not backup_path.exists():
                raise Exception(f"Backup {backup_timestamp} not found")
                
            # Remove current configs
            for config_file in self.config_dir.glob("*.json"):
                config_file.unlink()
            for config_file in self.config_dir.glob("*.enc"):
                config_file.unlink()
                
            # Restore from backup
            for backup_file in backup_path.glob("*.*"):
                shutil.copy2(backup_file, self.config_dir)
                
            logger.info(f"Restored backup from {backup_timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False 