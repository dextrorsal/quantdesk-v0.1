/**
 * Simple test to validate collateral retrieval fixes
 * Run this in the browser console or as part of the frontend
 */

// Test the improved getSOLCollateralBalance method
async function testCollateralFix() {
  console.log('🧪 Testing Collateral Retrieval Fixes');
  console.log('=====================================');
  
  try {
    // Import the service (this would work in the frontend context)
    const { smartContractService } = await import('./src/services/smartContractService.ts');
    
    // Test with a sample wallet address
    const testWalletAddress = '11111111111111111111111111111112';
    
    console.log('🔍 Testing wallet:', testWalletAddress);
    
    // Test the improved method
    const collateralBalance = await smartContractService.getSOLCollateralBalance(testWalletAddress);
    
    console.log('💰 Collateral balance retrieved:', collateralBalance, 'SOL');
    console.log('✅ Method executed successfully');
    
    // Test getUserAccountState as well
    const accountState = await smartContractService.getUserAccountState(testWalletAddress);
    
    console.log('📊 Account state:', accountState);
    console.log('✅ getUserAccountState executed successfully');
    
    return {
      success: true,
      collateralBalance,
      accountState
    };
    
  } catch (error) {
    console.error('❌ Test failed:', error);
    return {
      success: false,
      error: error.message
    };
  }
}

// Export for use in browser console
if (typeof window !== 'undefined') {
  (window as any).testCollateralFix = testCollateralFix;
}

export { testCollateralFix };
