import { useCallback } from 'react';
import { useProgram } from '../contexts/ProgramContext';
import * as anchor from '@coral-xyz/anchor';

export interface TradeParams {
    marketSymbol: string;
    side: 'long' | 'short';
    size: number;
    leverage: number;
    price?: number; // For limit orders
    stopPrice?: number; // For stop orders
}

export interface OrderParams {
    marketSymbol: string;
    side: 'long' | 'short';
    size: number;
    price: number;
    orderType: 'limit' | 'stop' | 'stop_limit';
    leverage: number;
}

/**
 * Hook for trading operations with your Anchor program
 * Provides type-safe methods for opening/closing positions and managing orders
 */
export const useTrading = () => {
    const { 
        program, 
        markets, 
        userPositions, 
        openPosition, 
        closePosition,
        loadUserPositions 
    } = useProgram();

    // Open a position
    const trade = useCallback(async (params: TradeParams): Promise<string> => {
        if (!program) throw new Error('Program not initialized');
        
        const { marketSymbol, side, size, leverage } = params;
        
        // Validate market exists
        const market = markets.find(m => m.symbol === marketSymbol);
        if (!market) throw new Error(`Market ${marketSymbol} not found`);
        
        // Validate leverage
        if (leverage > market.maxLeverage) {
            throw new Error(`Leverage ${leverage}x exceeds maximum ${market.maxLeverage}x`);
        }
        
        // Validate minimum size (you might want to add this to your market data)
        if (size <= 0) {
            throw new Error('Position size must be greater than 0');
        }
        
        try {
            const txSignature = await openPosition(marketSymbol, side, size, leverage);
            
            // Reload positions to get updated data
            await loadUserPositions();
            
            return txSignature;
        } catch (error) {
            console.error('Trade failed:', error);
            throw error;
        }
    }, [program, markets, openPosition, loadUserPositions]);

    // Close a position
    const closeTrade = useCallback(async (marketSymbol: string): Promise<string> => {
        if (!program) throw new Error('Program not initialized');
        
        // Check if user has an open position
        const position = userPositions.find(p => 
            p.market.equals(markets.find(m => m.symbol === marketSymbol)?.marketPDA!)
        );
        
        if (!position) {
            throw new Error(`No open position found for ${marketSymbol}`);
        }
        
        try {
            const txSignature = await closePosition(marketSymbol);
            
            // Reload positions
            await loadUserPositions();
            
            return txSignature;
        } catch (error) {
            console.error('Close position failed:', error);
            throw error;
        }
    }, [program, userPositions, markets, closePosition, loadUserPositions]);

    // Place an order (limit, stop, etc.) - Now uses backend API with enhanced error handling
    const placeOrder = useCallback(async (params: OrderParams): Promise<string> => {
        const { marketSymbol, side, size, price, orderType, leverage } = params;
        
        // Enhanced validation
        if (!marketSymbol || !side || !size || size <= 0) {
            throw new Error('Invalid order parameters: marketSymbol, side, and positive size required');
        }
        
        if (orderType === 'limit' && (!price || price <= 0)) {
            throw new Error('Limit orders require a positive price');
        }
        
        if (leverage <= 0 || leverage > 100) {
            throw new Error('Leverage must be between 1 and 100');
        }
        
        try {
            // Map frontend side to backend side format
            const backendSide = side === 'long' ? 'buy' : 'sell';
            
            // Call backend API to place order
            const response = await fetch('/api/orders', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}` // Add auth if needed
                },
                body: JSON.stringify({
                    symbol: marketSymbol,
                    side: backendSide,
                    size: size,
                    orderType: orderType,
                    price: price,
                    leverage: leverage
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                const errorMessage = errorData.error || errorData.message || 'Failed to place order';
                const errorCode = errorData.code || 'UNKNOWN_ERROR';
                
                console.error('❌ Order placement failed:', {
                    status: response.status,
                    error: errorMessage,
                    code: errorCode,
                    details: errorData.details
                });
                
                throw new Error(`${errorMessage} (${errorCode})`);
            }

            const result = await response.json();
            console.log('✅ Order placed via backend:', result);
            
            // Validate response has required fields
            if (!result.orderId && !result.id) {
                throw new Error('Invalid response: missing order ID');
            }
            
            // Return order ID instead of transaction signature
            return result.orderId || result.id || 'order-placed';
        } catch (error) {
            console.error('❌ Place order failed:', error);
            
            // Re-throw with enhanced error context
            if (error instanceof Error) {
                throw new Error(`Order placement failed: ${error.message}`);
            } else {
                throw new Error('Order placement failed: Unknown error');
            }
        }
    }, []);

    // Cancel an order - Now uses backend API
    const cancelOrder = useCallback(async (orderId: string): Promise<string> => {
        try {
            // Call backend API to cancel order
            const response = await fetch(`/api/orders/${orderId}/cancel`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}` // Add auth if needed
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to cancel order');
            }

            const result = await response.json();
            console.log('✅ Order cancelled via backend:', result);
            
            return result.orderId || orderId;
        } catch (error) {
            console.error('❌ Cancel order failed:', error);
            throw error;
        }
    }, []);

    // Get user's position for a specific market
    const getUserPosition = useCallback((marketSymbol: string) => {
        const market = markets.find(m => m.symbol === marketSymbol);
        if (!market) return null;
        
        return userPositions.find(p => p.market.equals(market.marketPDA!));
    }, [markets, userPositions]);

    // Calculate PnL for a position
    const calculatePnL = useCallback((marketSymbol: string): number => {
        const position = getUserPosition(marketSymbol);
        if (!position) return 0;
        
        const market = markets.find(m => m.symbol === marketSymbol);
        if (!market) return 0;
        
        // Calculate unrealized PnL
        const currentPrice = market.currentPrice;
        const entryPrice = position.entryPrice.toNumber();
        const size = position.size.toNumber();
        
        let pnl = 0;
        if (position.side === 'long') {
            pnl = (currentPrice - entryPrice) * size;
        } else {
            pnl = (entryPrice - currentPrice) * size;
        }
        
        return pnl;
    }, [getUserPosition, markets]);

    // Check if user can open a position (margin requirements)
    const canOpenPosition = useCallback((marketSymbol: string, size: number, leverage: number): boolean => {
        const market = markets.find(m => m.symbol === marketSymbol);
        if (!market) return false;
        
        // Check leverage limits
        if (leverage > market.maxLeverage) return false;
        
        // Check margin requirements (you'll need to implement this based on your collateral system)
        // This is a simplified check
        return true;
    }, [markets]);

    return {
        // Trading functions
        trade,
        closeTrade,
        placeOrder,
        cancelOrder,
        
        // Position management
        getUserPosition,
        calculatePnL,
        canOpenPosition,
        
        // Data
        markets,
        userPositions,
        program,
    };
};

export default useTrading;
