/**
 * MOCK Deposit Service for Demo
 * Simulates successful deposits without actually hitting the blockchain
 */

export class MockDepositService {
  private mockBalance: number = 0;
  private mockTransactions: Array<{
    signature: string;
    amount: number;
    timestamp: number;
  }> = [];

  async deposit(amount: number): Promise<{ signature: string; balance: number }> {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Generate fake signature
    const signature = `mock_${Date.now()}_${Math.random().toString(36).substring(7)}`;
    
    // Update mock balance
    this.mockBalance += amount / 1e9; // Convert lamports to SOL

    // Store transaction
    this.mockTransactions.push({
      signature,
      amount: amount / 1e9,
      timestamp: Date.now(),
    });

    console.log('✅ MOCK DEPOSIT SUCCESS:', {
      signature,
      amount: `${amount / 1e9} SOL`,
      newBalance: `${this.mockBalance} SOL`,
    });

    return {
      signature,
      balance: this.mockBalance,
    };
  }

  getBalance(): number {
    return this.mockBalance;
  }

  getTransactions() {
    return this.mockTransactions;
  }

  // Reset for testing
  reset() {
    this.mockBalance = 0;
    this.mockTransactions = [];
  }
}

export const mockDepositService = new MockDepositService();
