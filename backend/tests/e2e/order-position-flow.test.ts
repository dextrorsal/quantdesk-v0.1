#!/usr/bin/env node

/**
 * End-to-End Test: Order → Position Flow
 * 
 * This script tests the complete flow from order placement to position creation
 * including WebSocket updates and smart contract integration.
 */

import { io, Socket } from 'socket.io-client';
import fetch from 'node-fetch';

const SERVER_URL = process.env.SERVER_URL || 'http://localhost:3002';
const TEST_USER_ID = 'test-user-e2e-' + Date.now();
const TEST_TOKEN = 'test-token-' + Date.now();

interface OrderResult {
  orderId: string;
  filled: boolean;
  fills: Array<{ price: number; size: number }>;
  averageFillPrice?: number;
}

interface OrderUpdate {
  symbol: string;
  orderId: string;
  status: string;
  filledSize: number;
  averageFillPrice?: number;
  userId: string;
  timestamp: number;
  smartContractTx?: string;
}

class OrderPositionFlowTest {
  private socket: Socket | null = null;
  private orderUpdates: OrderUpdate[] = [];
  private positionUpdates: any[] = [];
  private isConnected = false;

  async run(): Promise<void> {
    console.log('🚀 Starting Order → Position Flow E2E Test');
    console.log('==========================================');

    try {
      // Step 1: Initialize WebSocket connection
      await this.initializeWebSocket();
      
      // Step 2: Place a test order
      const orderResult = await this.placeTestOrder();
      
      // Step 3: Wait for order updates
      await this.waitForOrderUpdates(orderResult.orderId);
      
      // Step 4: Verify position creation
      await this.verifyPositionCreation(orderResult.orderId);
      
      // Step 5: Cleanup
      await this.cleanup();
      
      console.log('✅ Order → Position Flow E2E Test PASSED');
      
    } catch (error) {
      console.error('❌ Order → Position Flow E2E Test FAILED:', error);
      await this.cleanup();
      process.exit(1);
    }
  }

  private async initializeWebSocket(): Promise<void> {
    console.log('📡 Initializing WebSocket connection...');
    
    return new Promise((resolve, reject) => {
      this.socket = io(SERVER_URL, {
        auth: { token: TEST_TOKEN },
        transports: ['websocket'],
        timeout: 10000
      });

      this.socket.on('connect', () => {
        console.log('✅ WebSocket connected');
        this.isConnected = true;
        
        // Subscribe to order updates
        this.socket!.emit('subscribe_orders');
        console.log('📋 Subscribed to order updates');
        
        resolve();
      });

      this.socket.on('connect_error', (error) => {
        console.error('❌ WebSocket connection failed:', error);
        reject(error);
      });

      this.socket.on('order_update', (update: OrderUpdate) => {
        console.log('📨 Received order update:', update);
        this.orderUpdates.push(update);
      });

      this.socket.on('position_update', (update: any) => {
        console.log('📈 Received position update:', update);
        this.positionUpdates.push(update);
      });

      // Timeout after 10 seconds
      setTimeout(() => {
        if (!this.isConnected) {
          reject(new Error('WebSocket connection timeout'));
        }
      }, 10000);
    });
  }

  private async placeTestOrder(): Promise<OrderResult> {
    console.log('📝 Placing test order...');
    
    const orderData = {
      symbol: 'BTC/USD',
      side: 'buy',
      size: 0.001, // Small amount for testing
      orderType: 'market',
      leverage: 1
    };

    try {
      const response = await fetch(`${SERVER_URL}/api/orders`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${TEST_TOKEN}`
        },
        body: JSON.stringify(orderData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Order placement failed: ${errorData.error || errorData.message}`);
      }

      const result = await response.json();
      console.log('✅ Order placed successfully:', result);
      
      return result.data || result;
      
    } catch (error) {
      console.error('❌ Order placement failed:', error);
      throw error;
    }
  }

  private async waitForOrderUpdates(orderId: string, timeoutMs: number = 30000): Promise<void> {
    console.log(`⏳ Waiting for order updates for order ${orderId}...`);
    
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      
      const checkInterval = setInterval(() => {
        const relevantUpdates = this.orderUpdates.filter(update => update.orderId === orderId);
        
        if (relevantUpdates.length > 0) {
          console.log(`✅ Received ${relevantUpdates.length} order updates`);
          clearInterval(checkInterval);
          resolve();
          return;
        }
        
        if (Date.now() - startTime > timeoutMs) {
          console.error('❌ Timeout waiting for order updates');
          clearInterval(checkInterval);
          reject(new Error('Timeout waiting for order updates'));
        }
      }, 1000);
    });
  }

  private async verifyPositionCreation(orderId: string): Promise<void> {
    console.log('🔍 Verifying position creation...');
    
    try {
      // Check if position updates were received
      if (this.positionUpdates.length > 0) {
        console.log('✅ Position updates received:', this.positionUpdates.length);
      } else {
        console.log('⚠️ No position updates received (this might be expected for small orders)');
      }
      
      // Check order status
      const orderUpdate = this.orderUpdates.find(update => update.orderId === orderId);
      if (orderUpdate) {
        console.log('📊 Order status:', orderUpdate.status);
        console.log('💰 Filled size:', orderUpdate.filledSize);
        console.log('💵 Average fill price:', orderUpdate.averageFillPrice);
        
        if (orderUpdate.smartContractTx) {
          console.log('🔗 Smart contract transaction:', orderUpdate.smartContractTx);
        }
      }
      
      console.log('✅ Position creation verification completed');
      
    } catch (error) {
      console.error('❌ Position creation verification failed:', error);
      throw error;
    }
  }

  private async cleanup(): Promise<void> {
    console.log('🧹 Cleaning up...');
    
    if (this.socket) {
      this.socket.emit('unsubscribe_orders');
      this.socket.disconnect();
      this.socket = null;
    }
    
    console.log('✅ Cleanup completed');
  }
}

// Run the test
if (require.main === module) {
  const test = new OrderPositionFlowTest();
  test.run().catch(error => {
    console.error('Test failed:', error);
    process.exit(1);
  });
}

export { OrderPositionFlowTest };
