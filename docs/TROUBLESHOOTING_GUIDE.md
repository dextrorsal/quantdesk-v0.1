# QuantDesk Contract Integration Troubleshooting Guide

## 🚨 Emergency Procedures

### Critical Issues
If you encounter critical issues that affect trading:

1. **Stop all trading operations**
2. **Document the issue** with screenshots and logs
3. **Contact support immediately**: support@quantdesk.com
4. **Monitor for further issues**

## 🔧 Smart Contract Issues

### Contract Deployment Problems

#### Issue: Contract Deployment Fails
**Symptoms:**
- Deployment command fails
- Transaction errors
- Program not found

**Diagnosis:**
```bash
# Check Solana configuration
solana config get

# Verify network connectivity
solana cluster-version

# Check wallet balance
solana balance

# Verify program build
anchor build
```

**Solutions:**
1. **Insufficient SOL Balance**
   ```bash
   # Request devnet SOL
   solana airdrop 2
   
   # Check balance
   solana balance
   ```

2. **Network Issues**
   ```bash
   # Switch to devnet
   solana config set --url devnet
   
   # Verify connection
   solana cluster-version
   ```

3. **Build Issues**
   ```bash
   # Clean build
   anchor clean
   anchor build
   
   # Check for errors
   cargo check
   ```

#### Issue: Program ID Mismatch
**Symptoms:**
- Frontend can't find program
- Transaction failures
- "Program not found" errors

**Diagnosis:**
```bash
# Check deployed program
solana program show C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw

# Verify Anchor.toml
cat contracts/Anchor.toml
```

**Solutions:**
1. **Update Program ID**
   ```bash
   # Update Anchor.toml
   # Set correct program ID in [programs.devnet]
   quantdesk_perp_dex = "C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw"
   ```

2. **Redeploy Contract**
   ```bash
   anchor deploy --provider.cluster devnet
   ```

3. **Update Frontend Configuration**
   ```typescript
   // Update in frontend/src/config/solana.ts
   export const QUANTDESK_PROGRAM_ID = "C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw";
   ```

### Contract Interaction Issues

#### Issue: Transaction Failures
**Symptoms:**
- Transactions rejected
- "Insufficient funds" errors
- "Account not found" errors

**Diagnosis:**
```bash
# Check account status
solana account C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw

# Check transaction logs
solana logs --url devnet C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw
```

**Solutions:**
1. **Account Not Found**
   ```bash
   # Create user account
   anchor test --skip-local-validator
   ```

2. **Insufficient Funds**
   ```bash
   # Check wallet balance
   solana balance
   
   # Request more SOL
   solana airdrop 2
   ```

3. **Transaction Size Limit**
   ```bash
   # Check transaction size
   solana transaction <transaction-id> --url devnet
   
   # Optimize transaction
   # Reduce instruction count
   # Use smaller data payloads
   ```

#### Issue: Oracle Price Issues
**Symptoms:**
- Stale price data
- Price deviation errors
- Oracle connection failures

**Diagnosis:**
```bash
# Check oracle program
solana program show FsJ3A3u2vn5cTVofAjvy6y5kwABJAqYWpe4975bi2epH

# Test oracle connection
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getAccountInfo","params":["FsJ3A3u2vn5cTVofAjvy6y5kwABJAqYWpe4975bi2epH"]}' \
  https://api.devnet.solana.com
```

**Solutions:**
1. **Oracle Staleness**
   ```typescript
   // Check oracle staleness threshold
   const stalenessThreshold = 300; // 5 minutes
   const currentTime = Date.now() / 1000;
   const priceAge = currentTime - priceData.timestamp;
   
   if (priceAge > stalenessThreshold) {
     // Handle stale price
     throw new Error('Oracle price is stale');
   }
   ```

2. **Price Deviation**
   ```typescript
   // Check price deviation
   const maxDeviation = 0.05; // 5%
   const priceChange = Math.abs(newPrice - oldPrice) / oldPrice;
   
   if (priceChange > maxDeviation) {
     // Trigger circuit breaker
     throw new Error('Price deviation too high');
   }
   ```

3. **Oracle Connection Issues**
   ```bash
   # Test Pyth Network
   curl https://hermes.pyth.network/v2/updates/price/latest
   
   # Check network connectivity
   ping api.devnet.solana.com
   ```

## 🔗 Backend Integration Issues

### API Connection Problems

#### Issue: Backend API Unreachable
**Symptoms:**
- Frontend can't connect to backend
- API timeout errors
- CORS errors

**Diagnosis:**
```bash
# Check backend status
curl http://localhost:3002/api/health

# Check port availability
netstat -tulpn | grep 3002

# Check backend logs
tail -f backend/logs/app.log
```

**Solutions:**
1. **Backend Not Running**
   ```bash
   # Start backend
   cd backend
   pnpm run dev
   
   # Check for errors
   pnpm run type-check
   ```

2. **Port Conflicts**
   ```bash
   # Find process using port
   lsof -i :3002
   
   # Kill conflicting process
   kill -9 <pid>
   
   # Start backend
   pnpm run dev
   ```

3. **CORS Issues**
   ```typescript
   // Update backend/src/server.ts
   app.use(cors({
     origin: ['http://localhost:3001', 'https://quantdesk.app'],
     credentials: true
   }));
   ```

#### Issue: Database Connection Failures
**Symptoms:**
- Database query failures
- Connection timeout errors
- Authentication errors

**Diagnosis:**
```bash
# Test database connection
curl -X POST \
  -H "Content-Type: application/json" \
  -H "apikey: YOUR_SUPABASE_ANON_KEY" \
  -d '{"query": "SELECT 1"}' \
  https://your-project.supabase.co/rest/v1/rpc/execute_sql
```

**Solutions:**
1. **Connection String Issues**
   ```bash
   # Check environment variables
   echo $DATABASE_URL
   echo $SUPABASE_URL
   echo $SUPABASE_ANON_KEY
   ```

2. **Network Issues**
   ```bash
   # Test Supabase connectivity
   ping your-project.supabase.co
   
   # Check firewall settings
   telnet your-project.supabase.co 443
   ```

3. **Authentication Issues**
   ```typescript
   // Verify Supabase configuration
   const supabase = createClient(
     process.env.SUPABASE_URL!,
     process.env.SUPABASE_ANON_KEY!
   );
   
   // Test connection
   const { data, error } = await supabase.from('users').select('count');
   ```

### WebSocket Connection Issues

#### Issue: WebSocket Connection Failures
**Symptoms:**
- Real-time updates not working
- Connection drops frequently
- Authentication failures

**Diagnosis:**
```bash
# Check WebSocket server
curl -i -N -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" \
  http://localhost:3002/ws
```

**Solutions:**
1. **Authentication Issues**
   ```typescript
   // Check JWT token
   const token = localStorage.getItem('auth_token');
   if (!token) {
     // Redirect to login
     window.location.href = '/login';
   }
   
   // Verify token validity
   const decoded = jwt.verify(token, process.env.JWT_SECRET!);
   ```

2. **Connection Drops**
   ```typescript
   // Implement reconnection logic
   const reconnectInterval = 5000; // 5 seconds
   
   socket.on('disconnect', () => {
     setTimeout(() => {
       socket.connect();
     }, reconnectInterval);
   });
   ```

3. **Network Issues**
   ```bash
   # Check WebSocket port
   netstat -tulpn | grep 3002
   
   # Test WebSocket connection
   wscat -c ws://localhost:3002/ws
   ```

## 🎨 Frontend Integration Issues

### Wallet Connection Problems

#### Issue: Wallet Not Detected
**Symptoms:**
- "No wallet found" error
- Wallet extension not recognized
- Connection fails

**Diagnosis:**
```typescript
// Check wallet availability
if (typeof window !== 'undefined' && window.solana) {
  console.log('Wallet detected:', window.solana);
} else {
  console.log('No wallet detected');
}
```

**Solutions:**
1. **Wallet Extension Not Installed**
   ```html
   <!-- Add wallet detection script -->
   <script>
     if (!window.solana) {
       alert('Please install a Solana wallet extension');
     }
   </script>
   ```

2. **Wrong Network**
   ```typescript
   // Check network
   const network = await wallet.connect();
   if (network !== 'devnet') {
     await wallet.switchNetwork('devnet');
   }
   ```

3. **Permission Issues**
   ```typescript
   // Request permissions
   try {
     await wallet.connect();
   } catch (error) {
     console.error('Connection failed:', error);
   }
   ```

### Transaction Signing Issues

#### Issue: Transaction Rejection
**Symptoms:**
- User rejects transaction
- Transaction fails to sign
- Insufficient funds error

**Diagnosis:**
```typescript
// Check transaction details
console.log('Transaction:', transaction);
console.log('Required SOL:', requiredSol);
console.log('Available SOL:', availableSol);
```

**Solutions:**
1. **Insufficient Funds**
   ```typescript
   // Check balance before transaction
   const balance = await connection.getBalance(publicKey);
   if (balance < requiredSol) {
     throw new Error('Insufficient SOL balance');
   }
   ```

2. **Transaction Too Large**
   ```typescript
   // Optimize transaction
   const transaction = new Transaction();
   transaction.add(instruction);
   
   // Check transaction size
   const serialized = transaction.serialize();
   if (serialized.length > 1232) {
     throw new Error('Transaction too large');
   }
   ```

3. **User Rejection**
   ```typescript
   // Handle user rejection gracefully
   try {
     await wallet.signTransaction(transaction);
   } catch (error) {
     if (error.message.includes('User rejected')) {
       // Handle rejection
       return;
     }
     throw error;
   }
   ```

## 📊 Performance Issues

### Slow Transaction Processing

#### Issue: High Transaction Latency
**Symptoms:**
- Transactions take > 30 seconds
- Frequent timeouts
- Poor user experience

**Diagnosis:**
```bash
# Check network congestion
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' \
  https://api.devnet.solana.com

# Check RPC performance
time curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getAccountInfo","params":["C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw"]}' \
  https://api.devnet.solana.com
```

**Solutions:**
1. **Network Congestion**
   ```typescript
   // Add priority fee
   const transaction = new Transaction();
   transaction.add(ComputeBudgetProgram.setComputeUnitPrice({
     microLamports: 1000 // 0.001 SOL
   }));
   ```

2. **RPC Issues**
   ```typescript
   // Use multiple RPC endpoints
   const rpcEndpoints = [
     'https://api.devnet.solana.com',
     'https://devnet.helius-rpc.com',
     'https://devnet.rpcpool.com'
   ];
   
   // Implement failover
   for (const endpoint of rpcEndpoints) {
     try {
       const connection = new Connection(endpoint);
       await connection.getAccountInfo(programId);
       break;
     } catch (error) {
       continue;
     }
   }
   ```

3. **Transaction Optimization**
   ```typescript
   // Batch operations
   const transaction = new Transaction();
   transaction.add(instruction1);
   transaction.add(instruction2);
   transaction.add(instruction3);
   
   // Send batch transaction
   await connection.sendTransaction(transaction, [wallet]);
   ```

### Memory and Resource Issues

#### Issue: High Memory Usage
**Symptoms:**
- Browser becomes slow
- Memory leaks
- Application crashes

**Diagnosis:**
```typescript
// Monitor memory usage
console.log('Memory usage:', performance.memory);

// Check for memory leaks
setInterval(() => {
  console.log('Memory:', performance.memory.usedJSHeapSize);
}, 5000);
```

**Solutions:**
1. **Memory Leaks**
   ```typescript
   // Clean up event listeners
   useEffect(() => {
     const handleUpdate = (data) => {
       // Handle update
     };
     
     socket.on('update', handleUpdate);
     
     return () => {
       socket.off('update', handleUpdate);
     };
   }, []);
   ```

2. **Large Data Sets**
   ```typescript
   // Implement pagination
   const ITEMS_PER_PAGE = 50;
   const [page, setPage] = useState(0);
   
   const items = data.slice(
     page * ITEMS_PER_PAGE,
     (page + 1) * ITEMS_PER_PAGE
   );
   ```

3. **Resource Cleanup**
   ```typescript
   // Clean up connections
   useEffect(() => {
     return () => {
       connection.removeAllListeners();
       socket.disconnect();
     };
   }, []);
   ```

## 🔍 Debugging Tools

### Development Tools

#### Solana CLI Commands
```bash
# Check cluster status
solana cluster-version

# Check account info
solana account <account-address>

# Check program info
solana program show <program-id>

# Check transaction
solana transaction <transaction-id>

# Check logs
solana logs <program-id>
```

#### Browser DevTools
```typescript
// Enable Solana debugging
window.solana.on('connect', () => {
  console.log('Wallet connected');
});

window.solana.on('disconnect', () => {
  console.log('Wallet disconnected');
});

// Log all transactions
window.solana.on('transaction', (transaction) => {
  console.log('Transaction:', transaction);
});
```

#### Network Monitoring
```bash
# Monitor network traffic
tcpdump -i any port 3002

# Check WebSocket connections
netstat -an | grep 3002

# Monitor Solana RPC calls
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' \
  https://api.devnet.solana.com
```

### Logging and Monitoring

#### Backend Logging
```typescript
// Structured logging
import { Logger } from './utils/logger';

const logger = new Logger();

logger.info('Transaction started', {
  userId,
  transactionId,
  amount
});

logger.error('Transaction failed', {
  error: error.message,
  stack: error.stack,
  transactionId
});
```

#### Frontend Logging
```typescript
// Client-side logging
const logTransaction = (transaction) => {
  console.log('Transaction:', {
    signature: transaction.signature,
    slot: transaction.slot,
    timestamp: new Date().toISOString()
  });
};
```

## 📞 Getting Help

### Support Channels
- **Email**: support@quantdesk.com
- **Discord**: [QuantDesk Community](https://discord.gg/quantdesk)
- **GitHub**: [Report Issues](https://github.com/quantdesk/quantdesk/issues)
- **Documentation**: [QuantDesk Docs](https://docs.quantdesk.com)

### When Reporting Issues
Include the following information:
1. **Environment**: Browser, OS, wallet type
2. **Network**: Devnet/Testnet/Mainnet
3. **Steps**: How to reproduce the issue
4. **Logs**: Relevant error messages and logs
5. **Screenshots**: Visual evidence of the issue
6. **Transaction IDs**: If applicable

### Emergency Contacts
- **Critical Issues**: support@quantdesk.com (urgent)
- **Security Issues**: security@quantdesk.com
- **Technical Issues**: tech@quantdesk.com

---

**Remember**: Always test on devnet first and never use real funds for testing!
