import React, { createContext, useState, useEffect, useContext, useMemo, useCallback } from 'react';
import * as anchor from '@coral-xyz/anchor';
import { Connection, PublicKey, SystemProgram, SYSVAR_RENT_PUBKEY } from '@solana/web3.js';
import { QuantdeskPerpDex } from '../types/quantdesk_perp_dex';
import IDL from '../types/quantdesk_perp_dex.json';
import {
    getOrCreateCollateralAccount,
    getPythPriceFeed,
    deriveUserAccountPDA,
    derivePositionPDA,
    deriveMarketPDA
} from '../utils/accountHelpers';

// Your program ID - MUST match deployed program on devnet
const PROGRAM_ID = new PublicKey("C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw");

interface MarketData {
    symbol: string;
    baseAsset: string;
    quoteAsset: string;
    maxLeverage: number;
    currentPrice: number;
    category: string;
    metadata: any;
    marketPDA: PublicKey | null;
    // On-chain data from your Anchor program
    baseReserve?: anchor.BN;
    quoteReserve?: anchor.BN;
    fundingRate?: anchor.BN;
    lastFundingTime?: anchor.BN;
    fundingInterval?: anchor.BN;
    authority?: PublicKey;
    initialMarginRatio?: number;
    maintenanceMarginRatio?: number;
    isActive?: boolean;
}

interface PositionData {
    user: PublicKey;
    market: PublicKey;
    side: 'long' | 'short';
    size: anchor.BN;
    entryPrice: anchor.BN;
    leverage: number;
    margin: anchor.BN;
    pnl: anchor.BN;
    isActive: boolean;
}

interface ProgramContextType {
    program: anchor.Program<QuantdeskPerpDex> | null;
    markets: MarketData[];
    userPositions: PositionData[];
    userAccount: any | null;
    loadMarkets: () => Promise<void>;
    loadUserPositions: () => Promise<void>;
    loadUserAccount: () => Promise<void>;
    createUserAccount: (accountIndex?: number, maxLeverage?: number) => Promise<PublicKey>;
    subscribeToMarket: (marketPDA: PublicKey) => void;
    unsubscribeFromMarket: (marketPDA: PublicKey) => void;
    openPosition: (marketSymbol: string, side: 'long' | 'short', size: number, leverage: number) => Promise<string>;
    closePosition: (marketSymbol: string) => Promise<string>;
    updateOraclePrice: (marketSymbol: string, newPrice: number) => Promise<string>;
}

interface ProgramProviderProps {
    children: React.ReactNode;
    connection: Connection;
    wallet: anchor.Wallet;
}

const ProgramContext = createContext<ProgramContextType>({
    program: null,
    markets: [],
    userPositions: [],
    userAccount: null,
    loadMarkets: async () => {},
    loadUserPositions: async () => {},
    loadUserAccount: async () => {},
    createUserAccount: async () => new PublicKey(0),
    subscribeToMarket: () => {},
    unsubscribeFromMarket: () => {},
    openPosition: async () => '',
    closePosition: async () => '',
    updateOraclePrice: async () => '',
});

export const useProgram = () => useContext(ProgramContext);

export const ProgramProvider: React.FC<ProgramProviderProps> = ({
    children,
    connection,
    wallet,
}) => {
    const [program, setProgram] = useState<anchor.Program<QuantdeskPerpDex> | null>(null);
    const [markets, setMarkets] = useState<MarketData[]>([]);
    const [userPositions, setUserPositions] = useState<PositionData[]>([]);
    const [userAccount, setUserAccount] = useState<any>(null);
    const [subscriptions, setSubscriptions] = useState<{[key: string]: number}>({});

    // Initialize program instance
    useEffect(() => {
        if (connection && wallet) {
            const provider = new anchor.AnchorProvider(
                connection, 
                wallet, 
                anchor.AnchorProvider.defaultOptions()
            );
            anchor.setProvider(provider);
            
            const program = new anchor.Program<QuantdeskPerpDex>(
                IDL as QuantdeskPerpDex, 
                PROGRAM_ID
            );
            setProgram(program);
        }
    }, [connection, wallet]);

    // Load markets from Supabase and derive PDAs
    const loadMarkets = useCallback(async () => {
        if (!program) return;

        try {
            // Fetch markets from your Supabase API
            const response = await fetch('/api/markets');
            const data = await response.json();
            
            if (data.success && data.markets) {
                const marketsWithPDAs = await Promise.all(
                    data.markets.map(async (market: any) => {
                        // Derive PDA using helper
                        const [marketPDA] = deriveMarketPDA(program, market.baseAsset, market.quoteAsset);

                        return {
                            ...market,
                            marketPDA,
                        };
                    })
                );

                setMarkets(marketsWithPDAs);
            }
        } catch (error) {
            console.error('Error loading markets:', error);
        }
    }, [program]);

    // Load user positions - OPTIMIZED: Batch RPC calls
    const loadUserPositions = useCallback(async () => {
        if (!program || !wallet.publicKey) return;

        try {
            // Derive all position PDAs first (no RPC calls)
            const positionPDAs = markets
                .filter(m => m.marketPDA)
                .map(market => ({
                    market,
                    pda: derivePositionPDA(program, wallet.publicKey, market.marketPDA!)[0]
                }));

            if (positionPDAs.length === 0) {
                setUserPositions([]);
                return;
            }

            // OPTIMIZATION: Batch fetch all position accounts in ONE RPC call
            const accountInfos = await program.provider.connection.getMultipleAccountsInfo(
                positionPDAs.map(p => p.pda)
            );

            // Decode positions that exist
            const positions: PositionData[] = [];
            accountInfos.forEach((accountInfo, index) => {
                if (!accountInfo) return; // Position doesn't exist
                
                try {
                    const position = program.coder.accounts.decode('position', accountInfo.data);
                    positions.push({
                        user: position.user,
                        market: position.market,
                        side: position.side.long ? 'long' : 'short',
                        size: position.size,
                        entryPrice: position.entryPrice,
                        leverage: position.leverage,
                        margin: position.margin,
                        pnl: position.pnl,
                        isActive: position.isActive,
                    });
                } catch (error) {
                    console.log(`Failed to decode position for ${positionPDAs[index].market.symbol}`);
                }
            });

            setUserPositions(positions);
            console.log(`Loaded ${positions.length} positions in 1 RPC call (${accountInfos.length} markets checked)`);
        } catch (error) {
            console.error('Error loading user positions:', error);
        }
    }, [program, wallet.publicKey, markets]);

    // Memoize user account PDA FIRST (before loadUserAccount uses it)
    const userAccountPDA = useMemo(() => {
        if (!program || !wallet.publicKey) return null;
        const [pda] = deriveUserAccountPDA(program, wallet.publicKey, 0);
        return pda;
    }, [program, wallet.publicKey]);

    // Load user account
    const loadUserAccount = useCallback(async () => {
        if (!program || !userAccountPDA) return;

        try {
            const account = await program.account.userAccount.fetch(userAccountPDA);
            setUserAccount(account);
            console.log('User account loaded:', userAccountPDA.toBase58());
        } catch (error) {
            console.log('User account not found, may need to create one');
            setUserAccount(null);
        }
    }, [program, userAccountPDA]);

    // Create user account (defined AFTER loadUserAccount to avoid TDZ issues)
    const createUserAccount = useCallback(async (
        accountIndex: number = 0,
        maxLeverage: number = 10
    ): Promise<PublicKey> => {
        if (!program || !wallet.publicKey) {
            throw new Error('Program or wallet not connected');
        }

        try {
            const [pda] = deriveUserAccountPDA(program, wallet.publicKey, accountIndex);

            console.log('Creating user account at PDA:', pda.toBase58());

            const tx = await program.methods
                .createUserAccount(accountIndex, maxLeverage)
                .accounts({
                    userAccount: pda,
                    authority: wallet.publicKey,
                    systemProgram: SystemProgram.programId,
                    rent: SYSVAR_RENT_PUBKEY,
                })
                .rpc();

            console.log('User account created:', tx);
            await program.provider.connection.confirmTransaction(tx, 'confirmed');
            
            await loadUserAccount();
            return pda;
        } catch (error) {
            console.error('Error creating user account:', error);
            throw error;
        }
    }, [program, wallet.publicKey, loadUserAccount]);

    // Subscribe to market account changes
    const subscribeToMarket = useCallback((marketPDA: PublicKey) => {
        if (!program || subscriptions[marketPDA.toBase58()]) return;

        const subscriptionId = connection.onAccountChange(
            marketPDA,
            async (accountInfo) => {
                if (accountInfo) {
                    try {
                        const marketData = await program.account.market.fetch(marketPDA);
                        setMarkets((prevMarkets) =>
                            prevMarkets.map((market) =>
                                market.marketPDA?.equals(marketPDA) 
                                    ? { 
                                        ...market, 
                                        baseReserve: marketData.baseReserve,
                                        quoteReserve: marketData.quoteReserve,
                                        fundingRate: marketData.fundingRate,
                                        lastFundingTime: marketData.lastFundingTime,
                                        fundingInterval: marketData.fundingInterval,
                                        authority: marketData.authority,
                                        initialMarginRatio: marketData.initialMarginRatio,
                                        maintenanceMarginRatio: marketData.maintenanceMarginRatio,
                                        isActive: marketData.isActive,
                                    } 
                                    : market
                            )
                        );
                    } catch (error) {
                        console.error(`Error updating market ${marketPDA.toBase58()}:`, error);
                    }
                }
            },
            'confirmed'
        );

        setSubscriptions((prevSubscriptions) => ({
            ...prevSubscriptions,
            [marketPDA.toBase58()]: subscriptionId,
        }));
    }, [program, connection, subscriptions]);

    const unsubscribeFromMarket = useCallback((marketPDA: PublicKey) => {
        const subscriptionId = subscriptions[marketPDA.toBase58()];
        if (subscriptionId) {
            connection.removeAccountChangeListener(subscriptionId);
            setSubscriptions((prevSubscriptions) => {
                const newSubscriptions = { ...prevSubscriptions };
                delete newSubscriptions[marketPDA.toBase58()];
                return newSubscriptions;
            });
        }
    }, [connection, subscriptions]);

    // Open position function
    const openPosition = useCallback(async (
        marketSymbol: string, 
        side: 'long' | 'short', 
        size: number, 
        leverage: number
    ): Promise<string> => {
        if (!program || !wallet.publicKey) throw new Error('Program or wallet not available');

        const market = markets.find(m => m.symbol === marketSymbol);
        if (!market || !market.marketPDA) throw new Error('Market not found');

        try {
            // Derive position PDA using helper
            const [positionPDA] = derivePositionPDA(program, wallet.publicKey, market.marketPDA);

            // Get or create user's collateral account
            const userCollateral = await getOrCreateCollateralAccount(program, wallet.publicKey);

            const tx = await program.methods
                .openPosition(
                    0, // position_index
                    side === 'long' ? { long: {} } : { short: {} },
                    new anchor.BN(size),
                    new anchor.BN(leverage),
                    new anchor.BN(market.currentPrice) // entry_price
                )
                .accounts({
                    position: positionPDA,
                    userAccount: userAccount ? userAccount.publicKey : wallet.publicKey,
                    market: market.marketPDA,
                    user: wallet.publicKey,
                    systemProgram: SystemProgram.programId,
                    clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
                })
                .rpc();

            console.log('Position opened:', tx);
            await program.provider.connection.confirmTransaction(tx, 'confirmed');
            
            // Reload positions
            await loadUserPositions();
            
            return tx;
        } catch (error) {
            console.error('Error opening position:', error);
            throw error;
        }
    }, [program, wallet.publicKey, markets, userAccount, loadUserPositions]);

    // Close position function
    const closePosition = useCallback(async (marketSymbol: string): Promise<string> => {
        if (!program || !wallet.publicKey) throw new Error('Program or wallet not available');

        const market = markets.find(m => m.symbol === marketSymbol);
        if (!market || !market.marketPDA) throw new Error('Market not found');

        try {
            // Derive position PDA using helper
            const [positionPDA] = derivePositionPDA(program, wallet.publicKey, market.marketPDA);

            const tx = await program.methods
                .closePosition()
                .accounts({
                    position: positionPDA,
                    userAccount: userAccount ? userAccount.publicKey : wallet.publicKey,
                    user: wallet.publicKey,
                    clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
                })
                .rpc();

            console.log('Position closed:', tx);
            await program.provider.connection.confirmTransaction(tx, 'confirmed');
            
            // Reload positions
            await loadUserPositions();
            
            return tx;
        } catch (error) {
            console.error('Error closing position:', error);
            throw error;
        }
    }, [program, wallet.publicKey, markets, userAccount, loadUserPositions]);

    // Update oracle price (for admin/keeper bots)
    const updateOraclePrice = useCallback(async (
        marketSymbol: string, 
        newPrice: number
    ): Promise<string> => {
        if (!program || !wallet.publicKey) throw new Error('Program or wallet not available');

        const market = markets.find(m => m.symbol === marketSymbol);
        if (!market || !market.marketPDA) throw new Error('Market not found');

        try {
            // Get Pyth price feed for the market
            const priceFeed = getPythPriceFeed(marketSymbol);

            const tx = await program.methods
                .updateOraclePrice(new anchor.BN(newPrice))
                .accounts({
                    market: market.marketPDA,
                    priceFeed: priceFeed,
                    authority: wallet.publicKey,
                    clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
                })
                .rpc();

            console.log('Oracle price updated:', tx);
            await program.provider.connection.confirmTransaction(tx, 'confirmed');
            return tx;
        } catch (error) {
            console.error('Error updating oracle price:', error);
            throw error;
        }
    }, [program, wallet.publicKey, markets]);

    // Initialize subscriptions when program is ready
    useEffect(() => {
        if (program && markets.length > 0) {
            markets.forEach((market) => {
                if (market.marketPDA) {
                    subscribeToMarket(market.marketPDA);
                }
            });

            return () => {
                markets.forEach((market) => {
                    if (market.marketPDA) {
                        unsubscribeFromMarket(market.marketPDA);
                    }
                });
            };
        }
    }, [program, markets, subscribeToMarket, unsubscribeFromMarket]);

    // Load data when program is ready
    useEffect(() => {
        if (program) {
            loadMarkets();
            loadUserAccount();
        }
    }, [program, loadMarkets, loadUserAccount]);

    useEffect(() => {
        if (program && markets.length > 0) {
            loadUserPositions();
        }
    }, [program, markets, loadUserPositions]);

    const contextValue: ProgramContextType = useMemo(() => ({
        program,
        markets,
        userPositions,
        userAccount,
        loadMarkets,
        loadUserPositions,
        loadUserAccount,
        createUserAccount,
        subscribeToMarket,
        unsubscribeFromMarket,
        openPosition,
        closePosition,
        updateOraclePrice,
    }), [
        program,
        markets,
        userPositions,
        userAccount,
        loadMarkets,
        loadUserPositions,
        loadUserAccount,
        createUserAccount,
        subscribeToMarket,
        unsubscribeFromMarket,
        openPosition,
        closePosition,
        updateOraclePrice,
    ]);

    return (
        <ProgramContext.Provider value={contextValue}>
            {children}
        </ProgramContext.Provider>
    );
};

export default ProgramContext;
