// Solana Agent Kit Integration Tests
import { SolanaAgentKitTools } from '../src/services/SolanaAgentKitTools';
import { TradingAgent } from '../src/agents/TradingAgent';

describe('Solana Agent Kit Integration', () => {
  let solanaTools: SolanaAgentKitTools;
  let tradingAgent: TradingAgent;

  beforeEach(() => {
    solanaTools = new SolanaAgentKitTools();
    tradingAgent = new TradingAgent();
  });

  describe('SolanaAgentKitTools', () => {
    test('should initialize without private key', () => {
      const tools = new SolanaAgentKitTools();
      expect(tools.isInitialized()).toBe(false);
      
      const status = tools.getStatus();
      expect(status.initialized).toBe(false);
      expect(status.hasPrivateKey).toBe(false);
      expect(status.rpcUrl).toBeDefined();
    });

    test('should return all available tools', () => {
      const tools = SolanaAgentKitTools.getAllTools();
      expect(tools).toHaveLength(5);
      
      const toolNames = tools.map(tool => tool.name);
      expect(toolNames).toContain('get_wallet_balance');
      expect(toolNames).toContain('get_token_balance');
      expect(toolNames).toContain('get_swap_quote');
      expect(toolNames).toContain('get_token_data');
      expect(toolNames).toContain('get_wallet_info');
    });

    test('should handle wallet balance query', async () => {
      const tools = SolanaAgentKitTools.getAllTools();
      const walletTool = tools.find(tool => tool.name === 'get_wallet_balance');
      
      expect(walletTool).toBeDefined();
      
      const result = await walletTool!.func('test-wallet-address');
      const parsed = JSON.parse(result);
      
      expect(parsed.wallet).toBe('test-wallet-address');
      expect(parsed.status).toBeDefined();
    });

    test('should handle token balance query', async () => {
      const tools = SolanaAgentKitTools.getAllTools();
      const tokenTool = tools.find(tool => tool.name === 'get_token_balance');
      
      expect(tokenTool).toBeDefined();
      
      const result = await tokenTool!.func('test-wallet,EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v');
      const parsed = JSON.parse(result);
      
      expect(parsed.wallet).toBe('test-wallet');
      expect(parsed.tokenMint).toBe('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v');
      expect(parsed.status).toBeDefined();
    });

    test('should handle swap quote query', async () => {
      const tools = SolanaAgentKitTools.getAllTools();
      const swapTool = tools.find(tool => tool.name === 'get_swap_quote');
      
      expect(swapTool).toBeDefined();
      
      const result = await swapTool!.func('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v,So11111111111111111111111111111111111111112,100');
      const parsed = JSON.parse(result);
      
      expect(parsed.inputMint).toBe('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v');
      expect(parsed.outputMint).toBe('So11111111111111111111111111111111111111112');
      expect(parsed.inputAmount).toBe('100');
      expect(parsed.status).toBeDefined();
    });

    test('should handle token data query', async () => {
      const tools = SolanaAgentKitTools.getAllTools();
      const tokenDataTool = tools.find(tool => tool.name === 'get_token_data');
      
      expect(tokenDataTool).toBeDefined();
      
      const result = await tokenDataTool!.func('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v');
      const parsed = JSON.parse(result);
      
      expect(parsed.mint).toBe('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v');
      expect(parsed.status).toBeDefined();
    });

    test('should handle wallet info query', async () => {
      const tools = SolanaAgentKitTools.getAllTools();
      const walletInfoTool = tools.find(tool => tool.name === 'get_wallet_info');
      
      expect(walletInfoTool).toBeDefined();
      
      const result = await walletInfoTool!.func('test-wallet-address');
      const parsed = JSON.parse(result);
      
      expect(parsed.wallet).toBe('test-wallet-address');
      expect(parsed.status).toBeDefined();
    });

    test('should handle invalid input gracefully', async () => {
      const tools = SolanaAgentKitTools.getAllTools();
      const walletTool = tools.find(tool => tool.name === 'get_wallet_balance');
      
      const result = await walletTool!.func('');
      const parsed = JSON.parse(result);
      
      expect(parsed.error).toBeDefined();
    });
  });

  describe('TradingAgent Integration', () => {
    test('should detect Solana Agent Kit queries', () => {
      // Access private method for testing
      const needsSolanaAgentKit = (tradingAgent as any).needsSolanaAgentKit.bind(tradingAgent);
      
      expect(needsSolanaAgentKit('What is my SOL balance?')).toBe(true);
      expect(needsSolanaAgentKit('Get me a Jupiter swap quote')).toBe(true);
      expect(needsSolanaAgentKit('Check my token balance')).toBe(true);
      expect(needsSolanaAgentKit('Show wallet info')).toBe(true);
      expect(needsSolanaAgentKit('What is the current SOL price?')).toBe(false);
    });

    test('should process Solana Agent Kit queries', async () => {
      const query = {
        query: 'What is my SOL balance?',
        userId: 'test-user',
        timestamp: new Date()
      };

      const response = await tradingAgent.processQuery(query);
      
      expect(response).toBeDefined();
      expect(response.response).toContain('Solana Agent Kit');
      expect(response.sources).toContain('Solana Agent Kit');
      expect(response.provider).toBe('SolanaAgentKitTools');
    });

    test('should handle token balance queries', async () => {
      const query = {
        query: 'What is my USDC token balance?',
        userId: 'test-user',
        timestamp: new Date()
      };

      const response = await tradingAgent.processQuery(query);
      
      expect(response).toBeDefined();
      expect(response.response).toContain('Solana Agent Kit');
      expect(response.confidence).toBeGreaterThan(0);
    });

    test('should handle swap quote queries', async () => {
      const query = {
        query: 'Get me a swap quote for 100 USDC to SOL',
        userId: 'test-user',
        timestamp: new Date()
      };

      const response = await tradingAgent.processQuery(query);
      
      expect(response).toBeDefined();
      expect(response.response).toContain('Solana Agent Kit');
      expect(response.timestamp).toBeInstanceOf(Date);
    });
  });

  describe('Error Handling', () => {
    test('should handle tool errors gracefully', async () => {
      const tools = SolanaAgentKitTools.getAllTools();
      const walletTool = tools.find(tool => tool.name === 'get_wallet_balance');
      
      // Test with invalid input that might cause errors
      const result = await walletTool!.func('invalid-input');
      const parsed = JSON.parse(result);
      
      // Should return error message instead of throwing
      expect(parsed).toBeDefined();
      expect(typeof parsed).toBe('object');
    });

    test('should handle missing tools gracefully', async () => {
      const query = {
        query: 'Some random query that should not match',
        userId: 'test-user',
        timestamp: new Date()
      };

      const response = await tradingAgent.processQuery(query);
      
      // Should not crash and return a response
      expect(response).toBeDefined();
      expect(response.response).toBeDefined();
    });
  });

  describe('Performance', () => {
    test('should respond within reasonable time', async () => {
      const startTime = Date.now();
      
      const query = {
        query: 'What is my SOL balance?',
        userId: 'test-user',
        timestamp: new Date()
      };

      const response = await tradingAgent.processQuery(query);
      const duration = Date.now() - startTime;
      
      expect(response).toBeDefined();
      expect(duration).toBeLessThan(5000); // Should respond within 5 seconds
    });

    test('should handle multiple concurrent queries', async () => {
      const queries = [
        { query: 'What is my SOL balance?', userId: 'test-user', timestamp: new Date() },
        { query: 'Get token balance', userId: 'test-user', timestamp: new Date() },
        { query: 'Show wallet info', userId: 'test-user', timestamp: new Date() }
      ];

      const promises = queries.map(q => tradingAgent.processQuery(q));
      const responses = await Promise.all(promises);
      
      expect(responses).toHaveLength(3);
      responses.forEach(response => {
        expect(response).toBeDefined();
        expect(response.response).toBeDefined();
      });
    });
  });
});
