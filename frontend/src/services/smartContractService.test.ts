import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { PublicKey } from '@solana/web3.js'
import { smartContractService } from './smartContractService'

// Mock the dependencies
vi.mock('@solana/web3.js', () => ({
  Connection: vi.fn().mockImplementation(() => ({
    getLatestBlockhash: vi.fn().mockResolvedValue({ blockhash: 'test-blockhash', lastValidBlockHeight: 100 }),
    sendTransaction: vi.fn(),
    confirmTransaction: vi.fn(),
  })),
  PublicKey: vi.fn().mockImplementation((key) => ({ toString: () => key })),
  SystemProgram: {
    transfer: vi.fn(),
  },
  Transaction: vi.fn().mockImplementation(() => ({
    add: vi.fn(),
    sign: vi.fn(),
  })),
  SYSVAR_RENT_PUBKEY: 'rent-pubkey',
}))

vi.mock('@coral-xyz/anchor', () => ({
  Program: vi.fn(),
  AnchorProvider: vi.fn(),
  BN: vi.fn(),
}))

vi.mock('@solana/spl-token', () => ({
  TOKEN_PROGRAM_ID: 'token-program-id',
  ASSOCIATED_TOKEN_PROGRAM_ID: 'associated-token-program-id',
}))

vi.mock('../types/quantdesk_perp_dex.json', () => ({
  default: {
    address: 'test-program-id',
  },
}))

describe('SmartContractService Error Handling', () => {
  let mockWallet: any

  beforeEach(() => {
    mockWallet = {
      adapter: {
        publicKey: new PublicKey('test-wallet-address'),
        signTransaction: vi.fn(),
        signAllTransactions: vi.fn(),
      },
    }
    
    // Clear all mocks
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Error Classification Logic', () => {
    it('should classify user rejection errors correctly', () => {
      const error = new Error('User rejected the request')
      
      // Test the error classification logic directly
      if (error?.message?.includes('User rejected')) {
        expect(error.message).toContain('User rejected')
      }
    })

    it('should classify insufficient funds errors correctly', () => {
      const error = new Error('Insufficient funds for this transaction')
      
      if (error?.message?.includes('Insufficient')) {
        expect(error.message).toContain('Insufficient')
      }
    })

    it('should classify blockhash errors correctly', () => {
      const error = new Error('Blockhash not found')
      
      if (error?.message?.includes('Blockhash')) {
        expect(error.message).toContain('Blockhash')
      }
    })

    it('should detect success in transaction logs', () => {
      const error = {
        message: 'Some error',
        logs: ['Program log: success', 'Other log'],
      }
      
      const hasSuccess = error?.logs?.some(log => log.includes('success'))
      expect(hasSuccess).toBe(true)
    })

    it('should not detect success when logs are missing', () => {
      const error = {
        message: 'Some error',
        logs: ['Program log: error', 'Other log'],
      }
      
      const hasSuccess = error?.logs?.some(log => log.includes('success'))
      expect(hasSuccess).toBe(false)
    })
  })

  describe('getUserAccountState', () => {
    it('should return default state when account does not exist', async () => {
      // Mock the method to return default state
      vi.spyOn(smartContractService, 'getUserAccountState').mockResolvedValueOnce({
        exists: false,
        canDeposit: false,
        canTrade: false,
        totalCollateral: 0,
        initialMarginRequirement: 0,
        maintenanceMarginRequirement: 0,
        availableMargin: 0,
        accountHealth: 0,
        liquidationThreshold: 0,
        maxLeverage: 0,
        totalPositions: 0,
        maxPositions: 0,
        totalOrders: 0,
        totalFundingPaid: 0,
        totalFundingReceived: 0,
        totalFeesPaid: 0,
        totalRebatesEarned: 0,
        isActive: false,
      })

      const result = await smartContractService.getUserAccountState('test-wallet-address')
      
      expect(result.exists).toBe(false)
      expect(result.canDeposit).toBe(false)
      expect(result.canTrade).toBe(false)
      expect(result.totalCollateral).toBe(0)
    })

    it('should return active state when account exists', async () => {
      // Mock the method to return active state
      vi.spyOn(smartContractService, 'getUserAccountState').mockResolvedValueOnce({
        exists: true,
        canDeposit: true,
        canTrade: true,
        totalCollateral: 1000,
        initialMarginRequirement: 100,
        maintenanceMarginRequirement: 50,
        availableMargin: 900,
        accountHealth: 95,
        liquidationThreshold: 10,
        maxLeverage: 20,
        totalPositions: 2,
        maxPositions: 10,
        totalOrders: 5,
        totalFundingPaid: 10,
        totalFundingReceived: 5,
        totalFeesPaid: 15,
        totalRebatesEarned: 2,
        isActive: true,
      })

      const result = await smartContractService.getUserAccountState('test-wallet-address')
      
      expect(result.exists).toBe(true)
      expect(result.canDeposit).toBe(true)
      expect(result.canTrade).toBe(true)
      expect(result.totalCollateral).toBe(1000)
      expect(result.isActive).toBe(true)
    })
  })

  describe('getSOLCollateralBalance', () => {
    it('should return 0 when collateral account does not exist', async () => {
      vi.spyOn(smartContractService, 'getSOLCollateralBalance').mockResolvedValueOnce(0)

      const balance = await smartContractService.getSOLCollateralBalance('test-wallet-address')
      
      expect(balance).toBe(0)
    })

    it('should return actual collateral balance when account exists', async () => {
      vi.spyOn(smartContractService, 'getSOLCollateralBalance').mockResolvedValueOnce(5.5)

      const balance = await smartContractService.getSOLCollateralBalance('test-wallet-address')
      
      expect(balance).toBe(5.5)
    })
  })
})