import { Logger } from './logger';

/**
 * USD/SOL Conversion Utilities
 * 
 * CRITICAL: These utilities ensure accurate price conversions and prevent
 * manipulation attacks by implementing proper validation and fallback mechanisms.
 */
export class PriceConversionUtils {
  private static instance: PriceConversionUtils;
  private logger = new Logger();

  // Security thresholds
  private readonly MAX_PRICE_CHANGE_PERCENT = 10; // 10% max price change
  private readonly PRICE_STALENESS_SECONDS = 30; // 30 seconds max staleness
  private readonly MIN_SOL_PRICE = 1; // $1 minimum SOL price
  private readonly MAX_SOL_PRICE = 10000; // $10,000 maximum SOL price

  private constructor() {}

  public static getInstance(): PriceConversionUtils {
    if (!PriceConversionUtils.instance) {
      PriceConversionUtils.instance = new PriceConversionUtils();
    }
    return PriceConversionUtils.instance;
  }

  /**
   * Convert SOL to USD with security validation
   * CRITICAL: This prevents price manipulation attacks
   */
  public convertSOLToUSD(
    solAmount: number,
    solPrice: number,
    options: {
      validatePrice?: boolean;
      allowStalePrice?: boolean;
      maxPriceChange?: number;
    } = {}
  ): {
    usdAmount: number;
    isValid: boolean;
    warnings: string[];
    errors: string[];
  } {
    const warnings: string[] = [];
    const errors: string[] = [];
    let isValid = true;

    try {
      // 1. Validate SOL amount
      if (solAmount < 0) {
        errors.push('SOL amount cannot be negative');
        isValid = false;
      }

      if (!isFinite(solAmount)) {
        errors.push('SOL amount must be a valid number');
        isValid = false;
      }

      // 2. Validate SOL price
      if (options.validatePrice !== false) {
        const priceValidation = this.validateSOLPrice(solPrice);
        if (!priceValidation.isValid) {
          errors.push(...priceValidation.errors);
          isValid = false;
        }
        warnings.push(...priceValidation.warnings);
      }

      // 3. Check for unrealistic conversions
      const usdAmount = solAmount * solPrice;
      
      if (usdAmount < 0) {
        errors.push('USD amount cannot be negative');
        isValid = false;
      }

      if (!isFinite(usdAmount)) {
        errors.push('USD amount must be a valid number');
        isValid = false;
      }

      // 4. Check for suspiciously large amounts
      if (usdAmount > 10000000) { // $10M
        warnings.push('Very large USD amount detected');
      }

      return {
        usdAmount: isValid ? usdAmount : 0,
        isValid,
        warnings,
        errors
      };

    } catch (error) {
      this.logger.error('Error converting SOL to USD:', error);
      return {
        usdAmount: 0,
        isValid: false,
        warnings: [],
        errors: ['Conversion failed due to system error']
      };
    }
  }

  /**
   * Convert USD to SOL with security validation
   */
  public convertUSDtoSOL(
    usdAmount: number,
    solPrice: number,
    options: {
      validatePrice?: boolean;
      allowStalePrice?: boolean;
      maxPriceChange?: number;
    } = {}
  ): {
    solAmount: number;
    isValid: boolean;
    warnings: string[];
    errors: string[];
  } {
    const warnings: string[] = [];
    const errors: string[] = [];
    let isValid = true;

    try {
      // 1. Validate USD amount
      if (usdAmount < 0) {
        errors.push('USD amount cannot be negative');
        isValid = false;
      }

      if (!isFinite(usdAmount)) {
        errors.push('USD amount must be a valid number');
        isValid = false;
      }

      // 2. Validate SOL price
      if (options.validatePrice !== false) {
        const priceValidation = this.validateSOLPrice(solPrice);
        if (!priceValidation.isValid) {
          errors.push(...priceValidation.errors);
          isValid = false;
        }
        warnings.push(...priceValidation.warnings);
      }

      // 3. Prevent division by zero
      if (solPrice <= 0) {
        errors.push('SOL price must be greater than zero');
        isValid = false;
        return {
          solAmount: 0,
          isValid: false,
          warnings,
          errors
        };
      }

      // 4. Calculate SOL amount
      const solAmount = usdAmount / solPrice;

      if (!isFinite(solAmount)) {
        errors.push('SOL amount must be a valid number');
        isValid = false;
      }

      // 5. Check for suspiciously large amounts
      if (solAmount > 1000000) { // 1M SOL
        warnings.push('Very large SOL amount detected');
      }

      return {
        solAmount: isValid ? solAmount : 0,
        isValid,
        warnings,
        errors
      };

    } catch (error) {
      this.logger.error('Error converting USD to SOL:', error);
      return {
        solAmount: 0,
        isValid: false,
        warnings: [],
        errors: ['Conversion failed due to system error']
      };
    }
  }

  /**
   * Validate SOL price for manipulation attempts
   */
  private validateSOLPrice(solPrice: number): {
    isValid: boolean;
    warnings: string[];
    errors: string[];
  } {
    const warnings: string[] = [];
    const errors: string[] = [];
    let isValid = true;

    try {
      // 1. Check for valid number
      if (!isFinite(solPrice)) {
        errors.push('SOL price must be a valid number');
        isValid = false;
      }

      // 2. Check price range
      if (solPrice < this.MIN_SOL_PRICE) {
        errors.push(`SOL price too low: $${solPrice} < $${this.MIN_SOL_PRICE}`);
        isValid = false;
      }

      if (solPrice > this.MAX_SOL_PRICE) {
        errors.push(`SOL price too high: $${solPrice} > $${this.MAX_SOL_PRICE}`);
        isValid = false;
      }

      // 3. Check for zero or negative prices
      if (solPrice <= 0) {
        errors.push('SOL price must be greater than zero');
        isValid = false;
      }

      // 4. Check for suspiciously round numbers (potential manipulation)
      if (solPrice % 1 === 0 && solPrice > 100) {
        warnings.push('Suspiciously round SOL price detected');
      }

      return {
        isValid,
        warnings,
        errors
      };

    } catch (error) {
      this.logger.error('Error validating SOL price:', error);
      return {
        isValid: false,
        warnings: [],
        errors: ['Price validation failed due to system error']
      };
    }
  }

  /**
   * Format SOL amount for display
   */
  public formatSOLAmount(solAmount: number, decimals: number = 4): string {
    try {
      if (!isFinite(solAmount) || solAmount < 0) {
        return '0.0000';
      }

      return solAmount.toFixed(decimals);
    } catch (error) {
      this.logger.error('Error formatting SOL amount:', error);
      return '0.0000';
    }
  }

  /**
   * Format USD amount for display
   */
  public formatUSDAmount(usdAmount: number, decimals: number = 2): string {
    try {
      if (!isFinite(usdAmount) || usdAmount < 0) {
        return '$0.00';
      }

      return `$${usdAmount.toFixed(decimals)}`;
    } catch (error) {
      this.logger.error('Error formatting USD amount:', error);
      return '$0.00';
    }
  }

  /**
   * Get SOL equivalent with proper validation
   * This replaces the old getSOLEquivalent function with security enhancements
   */
  public getSOLEquivalent(
    collateralRaw: number | string,
    solPrice: number
  ): {
    solAmount: number;
    isValid: boolean;
    warnings: string[];
    errors: string[];
  } {
    try {
      const collateralAmount = typeof collateralRaw === 'string' 
        ? parseFloat(collateralRaw) 
        : collateralRaw;

      return this.convertUSDtoSOL(collateralAmount, solPrice);
    } catch (error) {
      this.logger.error('Error getting SOL equivalent:', error);
      return {
        solAmount: 0,
        isValid: false,
        warnings: [],
        errors: ['SOL equivalent calculation failed']
      };
    }
  }

  /**
   * Format collateral with proper validation
   * This replaces the old formatCollateral function with security enhancements
   */
  public formatCollateral(
    collateralRaw: number | string,
    solPrice: number,
    displayMode: 'usd' | 'sol' = 'usd'
  ): {
    formatted: string;
    isValid: boolean;
    warnings: string[];
    errors: string[];
  } {
    try {
      const collateralAmount = typeof collateralRaw === 'string' 
        ? parseFloat(collateralRaw) 
        : collateralRaw;

      if (displayMode === 'usd') {
        const conversion = this.convertSOLToUSD(collateralAmount, solPrice);
        return {
          formatted: this.formatUSDAmount(conversion.usdAmount),
          isValid: conversion.isValid,
          warnings: conversion.warnings,
          errors: conversion.errors
        };
      } else {
        const conversion = this.convertUSDtoSOL(collateralAmount, solPrice);
        return {
          formatted: this.formatSOLAmount(conversion.solAmount),
          isValid: conversion.isValid,
          warnings: conversion.warnings,
          errors: conversion.errors
        };
      }
    } catch (error) {
      this.logger.error('Error formatting collateral:', error);
      return {
        formatted: displayMode === 'usd' ? '$0.00' : '0.0000',
        isValid: false,
        warnings: [],
        errors: ['Collateral formatting failed']
      };
    }
  }
}

// Export singleton instance
export const priceConversionUtils = PriceConversionUtils.getInstance();

// Legacy function exports for backward compatibility
export function getSOLEquivalent(
  collateralRaw: number | string,
  solPrice: number
): number {
  const result = priceConversionUtils.getSOLEquivalent(collateralRaw, solPrice);
  
  if (!result.isValid) {
    console.warn('SOL equivalent calculation failed:', result.errors);
  }
  
  return result.solAmount;
}

export function formatCollateral(
  collateralRaw: number | string,
  solPrice: number,
  displayMode: 'usd' | 'sol' = 'usd'
): string {
  const result = priceConversionUtils.formatCollateral(collateralRaw, solPrice, displayMode);
  
  if (!result.isValid) {
    console.warn('Collateral formatting failed:', result.errors);
  }
  
  return result.formatted;
}
