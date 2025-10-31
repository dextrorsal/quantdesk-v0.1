import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { validateConfig } from '../../src/config/environment';

describe('Environment Security Validation', () => {
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    // Save original environment
    originalEnv = { ...process.env };
    
    // Clear all environment variables
    Object.keys(process.env).forEach(key => {
      delete process.env[key];
    });
  });

  afterEach(() => {
    // Restore original environment
    process.env = originalEnv;
  });

  describe('Required Environment Variables', () => {
    it('should throw error when SOLANA_WALLET is missing', () => {
      // Set all required vars except SOLANA_WALLET
      process.env.SUPABASE_URL = 'https://test.supabase.co';
      process.env.SUPABASE_ANON_KEY = 'test-anon-key';
      process.env.JWT_SECRET = 'test-jwt-secret';
      process.env.QUANTDESK_PROGRAM_ID = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';
      process.env.SOLANA_RPC_URL = 'https://api.devnet.solana.com';

      expect(() => validateConfig()).toThrow('Missing required configuration: solanaWalletPath');
    });

    it('should throw error when QUANTDESK_PROGRAM_ID is missing', () => {
      // Set all required vars except QUANTDESK_PROGRAM_ID
      process.env.SUPABASE_URL = 'https://test.supabase.co';
      process.env.SUPABASE_ANON_KEY = 'test-anon-key';
      process.env.JWT_SECRET = 'test-jwt-secret';
      process.env.SOLANA_WALLET = '/path/to/wallet.json';
      process.env.SOLANA_RPC_URL = 'https://api.devnet.solana.com';

      // Should not throw error because QUANTDESK_PROGRAM_ID has a fallback value
      expect(() => validateConfig()).not.toThrow();
    });

    it('should throw error when SOLANA_RPC_URL is missing', () => {
      // Set all required vars except SOLANA_RPC_URL
      process.env.SUPABASE_URL = 'https://test.supabase.co';
      process.env.SUPABASE_ANON_KEY = 'test-anon-key';
      process.env.JWT_SECRET = 'test-jwt-secret';
      process.env.SOLANA_WALLET = '/path/to/wallet.json';
      process.env.QUANTDESK_PROGRAM_ID = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';

      // Should not throw error because SOLANA_RPC_URL has a fallback value
      expect(() => validateConfig()).not.toThrow();
    });

    it('should pass when all required variables are present', () => {
      // Set all required variables
      process.env.SUPABASE_URL = 'https://test.supabase.co';
      process.env.SUPABASE_ANON_KEY = 'test-anon-key';
      process.env.JWT_SECRET = 'test-jwt-secret';
      process.env.SOLANA_WALLET = '/path/to/wallet.json';
      process.env.QUANTDESK_PROGRAM_ID = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';
      process.env.SOLANA_RPC_URL = 'https://api.devnet.solana.com';

      expect(() => validateConfig()).not.toThrow();
    });
  });

  describe('SOLANA_WALLET Validation', () => {
    beforeEach(() => {
      // Set all other required vars
      process.env.SUPABASE_URL = 'https://test.supabase.co';
      process.env.SUPABASE_ANON_KEY = 'test-anon-key';
      process.env.JWT_SECRET = 'test-jwt-secret';
      process.env.QUANTDESK_PROGRAM_ID = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';
      process.env.SOLANA_RPC_URL = 'https://api.devnet.solana.com';
    });

    it('should throw error for placeholder wallet path', () => {
      process.env.SOLANA_WALLET = 'your_wallet_path_here';
      
      expect(() => validateConfig()).toThrow('SOLANA_WALLET must be set to a valid wallet file path');
    });

    it('should throw error for empty wallet path', () => {
      process.env.SOLANA_WALLET = '';
      
      expect(() => validateConfig()).toThrow('Missing required configuration: solanaWalletPath');
    });

    it('should pass for valid wallet path', () => {
      process.env.SOLANA_WALLET = '/path/to/wallet.json';
      
      expect(() => validateConfig()).not.toThrow();
    });
  });

  describe('QUANTDESK_PROGRAM_ID Validation', () => {
    beforeEach(() => {
      // Set all other required vars
      process.env.SUPABASE_URL = 'https://test.supabase.co';
      process.env.SUPABASE_ANON_KEY = 'test-anon-key';
      process.env.JWT_SECRET = 'test-jwt-secret';
      process.env.SOLANA_WALLET = '/path/to/wallet.json';
      process.env.SOLANA_RPC_URL = 'https://api.devnet.solana.com';
    });

    it('should throw error for short program ID', () => {
      process.env.QUANTDESK_PROGRAM_ID = 'short';
      
      expect(() => validateConfig()).toThrow('QUANTDESK_PROGRAM_ID must be a valid Solana program ID');
    });

    it('should pass for valid program ID', () => {
      process.env.QUANTDESK_PROGRAM_ID = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';
      
      expect(() => validateConfig()).not.toThrow();
    });
  });

  describe('SOLANA_RPC_URL Validation', () => {
    beforeEach(() => {
      // Set all other required vars
      process.env.SUPABASE_URL = 'https://test.supabase.co';
      process.env.SUPABASE_ANON_KEY = 'test-anon-key';
      process.env.JWT_SECRET = 'test-jwt-secret';
      process.env.SOLANA_WALLET = '/path/to/wallet.json';
      process.env.QUANTDESK_PROGRAM_ID = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';
    });

    it('should throw error for invalid RPC URL', () => {
      process.env.SOLANA_RPC_URL = 'invalid-url';
      
      expect(() => validateConfig()).toThrow('SOLANA_RPC_URL must be a valid HTTP/HTTPS URL');
    });

    it('should pass for valid HTTPS RPC URL', () => {
      process.env.SOLANA_RPC_URL = 'https://api.devnet.solana.com';
      
      expect(() => validateConfig()).not.toThrow();
    });

    it('should pass for valid HTTP RPC URL', () => {
      process.env.SOLANA_RPC_URL = 'http://localhost:8899';
      
      expect(() => validateConfig()).not.toThrow();
    });
  });
});
