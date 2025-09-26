import { Logger } from '../utils/logger';

const logger = new Logger();

// JIT Liquidity interfaces
export interface LiquidityAuction {
  id: string;
  marketId: string;
  side: 'buy' | 'sell';
  size: number;
  price: number;
  minPrice?: number;
  maxPrice?: number;
  deadline: number;
  status: AuctionStatus;
  participants: AuctionParticipant[];
  winningBid?: WinningBid;
  createdAt: number;
  updatedAt: number;
}

export interface AuctionParticipant {
  makerId: string;
  bidPrice: number;
  bidSize: number;
  timestamp: number;
  status: 'active' | 'withdrawn' | 'won' | 'lost';
}

export interface WinningBid {
  makerId: string;
  price: number;
  size: number;
  executionPrice: number;
  timestamp: number;
}

export enum AuctionStatus {
  OPEN = 'open',
  CLOSED = 'closed',
  EXECUTED = 'executed',
  CANCELLED = 'cancelled',
  EXPIRED = 'expired'
}

export interface MarketMaker {
  id: string;
  address: string;
  totalVolume: number;
  totalFees: number;
  activeAuctions: number;
  winRate: number;
  averageSpread: number;
  reputation: number;
  tier: MarketMakerTier;
  incentives: MarketMakerIncentive[];
  createdAt: number;
  lastActive: number;
}

export enum MarketMakerTier {
  BRONZE = 'bronze',
  SILVER = 'silver',
  GOLD = 'gold',
  PLATINUM = 'platinum',
  DIAMOND = 'diamond'
}

export interface MarketMakerIncentive {
  id: string;
  type: IncentiveType;
  amount: number;
  currency: string;
  conditions: IncentiveCondition[];
  status: 'active' | 'completed' | 'expired';
  startDate: number;
  endDate: number;
  claimed: boolean;
}

export enum IncentiveType {
  VOLUME_BONUS = 'volume_bonus',
  SPREAD_REWARD = 'spread_reward',
  PARTICIPATION_REWARD = 'participation_reward',
  QUALITY_BONUS = 'quality_bonus',
  RETENTION_BONUS = 'retention_bonus'
}

export interface IncentiveCondition {
  type: 'min_volume' | 'max_spread' | 'min_participation' | 'min_quality';
  value: number;
  period: number; // in days
}

export interface LiquidityMiningProgram {
  id: string;
  name: string;
  description: string;
  marketIds: string[];
  totalRewards: number;
  currency: string;
  startDate: number;
  endDate: number;
  status: 'active' | 'paused' | 'completed';
  participants: LiquidityMiningParticipant[];
  rules: LiquidityMiningRule[];
}

export interface LiquidityMiningParticipant {
  makerId: string;
  totalVolume: number;
  totalRewards: number;
  currentTier: number;
  lastClaim: number;
  joinedAt: number;
}

export interface LiquidityMiningRule {
  tier: number;
  minVolume: number;
  maxVolume?: number;
  rewardRate: number; // rewards per volume unit
  bonusMultiplier: number;
}

export interface PriceImprovement {
  id: string;
  auctionId: string;
  originalPrice: number;
  improvedPrice: number;
  improvementAmount: number;
  improvementPercentage: number;
  makerId: string;
  timestamp: number;
}

export interface MarketMakingStrategy {
  id: string;
  makerId: string;
  marketId: string;
  strategyType: StrategyType;
  parameters: StrategyParameters;
  isActive: boolean;
  performance: StrategyPerformance;
  createdAt: number;
  updatedAt: number;
}

export enum StrategyType {
  GRID_TRADING = 'grid_trading',
  MEAN_REVERSION = 'mean_reversion',
  MOMENTUM = 'momentum',
  ARBITRAGE = 'arbitrage',
  CUSTOM = 'custom'
}

export interface StrategyParameters {
  gridSize: number;
  gridCount: number;
  minSpread: number;
  maxSpread: number;
  rebalanceThreshold: number;
  stopLoss: number;
  takeProfit: number;
}

export interface StrategyPerformance {
  totalPnL: number;
  winRate: number;
  averageSpread: number;
  totalVolume: number;
  totalFees: number;
  sharpeRatio: number;
  maxDrawdown: number;
  lastUpdated: number;
}

class JITLiquidityService {
  private static instance: JITLiquidityService;
  private auctions: Map<string, LiquidityAuction> = new Map();
  private marketMakers: Map<string, MarketMaker> = new Map();
  private liquidityMiningPrograms: Map<string, LiquidityMiningProgram> = new Map();
  private priceImprovements: Map<string, PriceImprovement> = new Map();
  private marketMakingStrategies: Map<string, MarketMakingStrategy> = new Map();
  
  private constructor() {
    this.initializeDefaultPrograms();
  }
  
  public static getInstance(): JITLiquidityService {
    if (!JITLiquidityService.instance) {
      JITLiquidityService.instance = new JITLiquidityService();
    }
    return JITLiquidityService.instance;
  }
  
  // Initialize default liquidity mining programs
  private initializeDefaultPrograms(): void {
    const defaultProgram: LiquidityMiningProgram = {
      id: 'default-program',
      name: 'QuantDesk Liquidity Mining',
      description: 'Earn rewards by providing liquidity to QuantDesk markets',
      marketIds: ['BTC-PERP', 'ETH-PERP', 'SOL-PERP'],
      totalRewards: 1000000, // 1M tokens
      currency: 'QDK',
      startDate: Date.now(),
      endDate: Date.now() + (90 * 24 * 60 * 60 * 1000), // 90 days
      status: 'active',
      participants: [],
      rules: [
        {
          tier: 1,
          minVolume: 10000,
          maxVolume: 50000,
          rewardRate: 0.001,
          bonusMultiplier: 1.0
        },
        {
          tier: 2,
          minVolume: 50000,
          maxVolume: 100000,
          rewardRate: 0.0015,
          bonusMultiplier: 1.2
        },
        {
          tier: 3,
          minVolume: 100000,
          maxVolume: 500000,
          rewardRate: 0.002,
          bonusMultiplier: 1.5
        },
        {
          tier: 4,
          minVolume: 500000,
          rewardRate: 0.0025,
          bonusMultiplier: 2.0
        }
      ]
    };
    
    this.liquidityMiningPrograms.set('default-program', defaultProgram);
  }
  
  // Create a new liquidity auction
  public createAuction(
    marketId: string,
    side: 'buy' | 'sell',
    size: number,
    price: number,
    deadline: number,
    minPrice?: number,
    maxPrice?: number
  ): LiquidityAuction {
    const auction: LiquidityAuction = {
      id: `auction_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      marketId,
      side,
      size,
      price,
      minPrice,
      maxPrice,
      deadline,
      status: AuctionStatus.OPEN,
      participants: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    };
    
    this.auctions.set(auction.id, auction);
    logger.info(`Created liquidity auction ${auction.id} for ${marketId}`);
    
    return auction;
  }
  
  // Submit a bid to an auction
  public submitBid(
    auctionId: string,
    makerId: string,
    bidPrice: number,
    bidSize: number
  ): boolean {
    const auction = this.auctions.get(auctionId);
    if (!auction) {
      logger.error(`Auction ${auctionId} not found`);
      return false;
    }
    
    if (auction.status !== AuctionStatus.OPEN) {
      logger.error(`Auction ${auctionId} is not open`);
      return false;
    }
    
    if (Date.now() > auction.deadline) {
      logger.error(`Auction ${auctionId} has expired`);
      auction.status = AuctionStatus.EXPIRED;
      return false;
    }
    
    // Validate bid
    if (bidSize <= 0 || bidPrice <= 0) {
      logger.error(`Invalid bid parameters: price=${bidPrice}, size=${bidSize}`);
      return false;
    }
    
    // Check if maker already has a bid
    const existingBidIndex = auction.participants.findIndex(p => p.makerId === makerId);
    if (existingBidIndex >= 0) {
      // Update existing bid
      auction.participants[existingBidIndex] = {
        makerId,
        bidPrice,
        bidSize,
        timestamp: Date.now(),
        status: 'active'
      };
    } else {
      // Add new bid
      auction.participants.push({
        makerId,
        bidPrice,
        bidSize,
        timestamp: Date.now(),
        status: 'active'
      });
    }
    
    auction.updatedAt = Date.now();
    logger.info(`Bid submitted to auction ${auctionId} by maker ${makerId}`);
    
    return true;
  }
  
  // Close an auction and determine winner
  public closeAuction(auctionId: string): WinningBid | null {
    const auction = this.auctions.get(auctionId);
    if (!auction) {
      logger.error(`Auction ${auctionId} not found`);
      return null;
    }
    
    if (auction.status !== AuctionStatus.OPEN) {
      logger.error(`Auction ${auctionId} is not open`);
      return null;
    }
    
    // Filter active participants
    const activeParticipants = auction.participants.filter(p => p.status === 'active');
    
    if (activeParticipants.length === 0) {
      auction.status = AuctionStatus.CANCELLED;
      logger.warn(`Auction ${auctionId} cancelled - no active participants`);
      return null;
    }
    
    // Determine winner based on auction type
    let winner: AuctionParticipant;
    
    if (auction.side === 'buy') {
      // For buy orders, highest price wins
      winner = activeParticipants.reduce((best, current) => 
        current.bidPrice > best.bidPrice ? current : best
      );
    } else {
      // For sell orders, lowest price wins
      winner = activeParticipants.reduce((best, current) => 
        current.bidPrice < best.bidPrice ? current : best
      );
    }
    
    // Calculate execution price (improved price)
    const executionPrice = this.calculateExecutionPrice(auction, winner);
    
    const winningBid: WinningBid = {
      makerId: winner.makerId,
      price: winner.bidPrice,
      size: Math.min(winner.bidSize, auction.size),
      executionPrice,
      timestamp: Date.now()
    };
    
    auction.winningBid = winningBid;
    auction.status = AuctionStatus.CLOSED;
    auction.updatedAt = Date.now();
    
    // Record price improvement
    this.recordPriceImprovement(auctionId, auction.price, executionPrice, winner.makerId);
    
    // Update market maker stats
    this.updateMarketMakerStats(winner.makerId, winningBid);
    
    logger.info(`Auction ${auctionId} closed. Winner: ${winner.makerId}, Execution Price: ${executionPrice}`);
    
    return winningBid;
  }
  
  // Calculate execution price with improvement
  private calculateExecutionPrice(auction: LiquidityAuction, winner: AuctionParticipant): number {
    const originalPrice = auction.price;
    const bidPrice = winner.bidPrice;
    
    if (auction.side === 'buy') {
      // For buy orders, execution price is the better of original or bid
      return Math.min(originalPrice, bidPrice);
    } else {
      // For sell orders, execution price is the better of original or bid
      return Math.max(originalPrice, bidPrice);
    }
  }
  
  // Record price improvement
  private recordPriceImprovement(
    auctionId: string,
    originalPrice: number,
    improvedPrice: number,
    makerId: string
  ): void {
    const improvement = Math.abs(originalPrice - improvedPrice);
    const improvementPercentage = (improvement / originalPrice) * 100;
    
    const priceImprovement: PriceImprovement = {
      id: `improvement_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      auctionId,
      originalPrice,
      improvedPrice,
      improvementAmount: improvement,
      improvementPercentage,
      makerId,
      timestamp: Date.now()
    };
    
    this.priceImprovements.set(priceImprovement.id, priceImprovement);
  }
  
  // Update market maker statistics
  private updateMarketMakerStats(makerId: string, winningBid: WinningBid): void {
    let marketMaker = this.marketMakers.get(makerId);
    
    if (!marketMaker) {
      marketMaker = {
        id: makerId,
        address: makerId,
        totalVolume: 0,
        totalFees: 0,
        activeAuctions: 0,
        winRate: 0,
        averageSpread: 0,
        reputation: 100,
        tier: MarketMakerTier.BRONZE,
        incentives: [],
        createdAt: Date.now(),
        lastActive: Date.now()
      };
    }
    
    // Update stats
    marketMaker.totalVolume += winningBid.size;
    marketMaker.totalFees += winningBid.size * 0.001; // 0.1% fee
    marketMaker.lastActive = Date.now();
    
    // Calculate win rate
    const totalAuctions = marketMaker.activeAuctions + 1;
    const wins = marketMaker.winRate * marketMaker.activeAuctions + 1;
    marketMaker.winRate = wins / totalAuctions;
    marketMaker.activeAuctions = totalAuctions;
    
    // Update tier based on volume
    this.updateMarketMakerTier(marketMaker);
    
    this.marketMakers.set(makerId, marketMaker);
  }
  
  // Update market maker tier
  private updateMarketMakerTier(marketMaker: MarketMaker): void {
    const volume = marketMaker.totalVolume;
    
    if (volume >= 1000000) {
      marketMaker.tier = MarketMakerTier.DIAMOND;
    } else if (volume >= 500000) {
      marketMaker.tier = MarketMakerTier.PLATINUM;
    } else if (volume >= 100000) {
      marketMaker.tier = MarketMakerTier.GOLD;
    } else if (volume >= 50000) {
      marketMaker.tier = MarketMakerTier.SILVER;
    } else {
      marketMaker.tier = MarketMakerTier.BRONZE;
    }
  }
  
  // Get active auctions
  public getActiveAuctions(): LiquidityAuction[] {
    return Array.from(this.auctions.values())
      .filter(auction => auction.status === AuctionStatus.OPEN);
  }
  
  // Get auction by ID
  public getAuction(auctionId: string): LiquidityAuction | null {
    return this.auctions.get(auctionId) || null;
  }
  
  // Get market maker by ID
  public getMarketMaker(makerId: string): MarketMaker | null {
    return this.marketMakers.get(makerId) || null;
  }
  
  // Get all market makers
  public getAllMarketMakers(): MarketMaker[] {
    return Array.from(this.marketMakers.values());
  }
  
  // Get liquidity mining program
  public getLiquidityMiningProgram(programId: string): LiquidityMiningProgram | null {
    return this.liquidityMiningPrograms.get(programId) || null;
  }
  
  // Get all liquidity mining programs
  public getAllLiquidityMiningPrograms(): LiquidityMiningProgram[] {
    return Array.from(this.liquidityMiningPrograms.values());
  }
  
  // Join liquidity mining program
  public joinLiquidityMiningProgram(programId: string, makerId: string): boolean {
    const program = this.liquidityMiningPrograms.get(programId);
    if (!program) {
      logger.error(`Liquidity mining program ${programId} not found`);
      return false;
    }
    
    if (program.status !== 'active') {
      logger.error(`Liquidity mining program ${programId} is not active`);
      return false;
    }
    
    // Check if already participating
    const existingParticipant = program.participants.find(p => p.makerId === makerId);
    if (existingParticipant) {
      logger.warn(`Maker ${makerId} already participating in program ${programId}`);
      return true;
    }
    
    // Add new participant
    const participant: LiquidityMiningParticipant = {
      makerId,
      totalVolume: 0,
      totalRewards: 0,
      currentTier: 1,
      lastClaim: Date.now(),
      joinedAt: Date.now()
    };
    
    program.participants.push(participant);
    logger.info(`Maker ${makerId} joined liquidity mining program ${programId}`);
    
    return true;
  }
  
  // Calculate liquidity mining rewards
  public calculateLiquidityMiningRewards(programId: string, makerId: string): number {
    const program = this.liquidityMiningPrograms.get(programId);
    if (!program) return 0;
    
    const participant = program.participants.find(p => p.makerId === makerId);
    if (!participant) return 0;
    
    // Find appropriate tier
    const tier = program.rules.find(rule => 
      participant.totalVolume >= rule.minVolume && 
      (!rule.maxVolume || participant.totalVolume <= rule.maxVolume)
    );
    
    if (!tier) return 0;
    
    // Calculate rewards based on volume and tier
    const baseRewards = participant.totalVolume * tier.rewardRate;
    const bonusRewards = baseRewards * tier.bonusMultiplier;
    
    return bonusRewards;
  }
  
  // Get price improvements
  public getPriceImprovements(limit: number = 100): PriceImprovement[] {
    return Array.from(this.priceImprovements.values())
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, limit);
  }
  
  // Get market making strategies
  public getMarketMakingStrategies(makerId?: string): MarketMakingStrategy[] {
    let strategies = Array.from(this.marketMakingStrategies.values());
    
    if (makerId) {
      strategies = strategies.filter(s => s.makerId === makerId);
    }
    
    return strategies;
  }
  
  // Create market making strategy
  public createMarketMakingStrategy(
    makerId: string,
    marketId: string,
    strategyType: StrategyType,
    parameters: StrategyParameters
  ): MarketMakingStrategy {
    const strategy: MarketMakingStrategy = {
      id: `strategy_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      makerId,
      marketId,
      strategyType,
      parameters,
      isActive: true,
      performance: {
        totalPnL: 0,
        winRate: 0,
        averageSpread: 0,
        totalVolume: 0,
        totalFees: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        lastUpdated: Date.now()
      },
      createdAt: Date.now(),
      updatedAt: Date.now()
    };
    
    this.marketMakingStrategies.set(strategy.id, strategy);
    logger.info(`Created market making strategy ${strategy.id} for maker ${makerId}`);
    
    return strategy;
  }
  
  // Get JIT liquidity statistics
  public getJITLiquidityStats(): any {
    const activeAuctions = this.getActiveAuctions();
    const totalMarketMakers = this.marketMakers.size;
    const totalVolume = Array.from(this.marketMakers.values())
      .reduce((sum, mm) => sum + mm.totalVolume, 0);
    const totalFees = Array.from(this.marketMakers.values())
      .reduce((sum, mm) => sum + mm.totalFees, 0);
    const priceImprovements = this.getPriceImprovements(1000);
    const avgPriceImprovement = priceImprovements.length > 0 
      ? priceImprovements.reduce((sum, pi) => sum + pi.improvementPercentage, 0) / priceImprovements.length
      : 0;
    
    return {
      activeAuctions: activeAuctions.length,
      totalMarketMakers,
      totalVolume,
      totalFees,
      totalPriceImprovements: priceImprovements.length,
      averagePriceImprovement: avgPriceImprovement,
      liquidityMiningPrograms: this.liquidityMiningPrograms.size,
      marketMakingStrategies: this.marketMakingStrategies.size,
      timestamp: Date.now()
    };
  }
}

export const jitLiquidityService = JITLiquidityService.getInstance();
export default jitLiquidityService;
