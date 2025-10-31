/**
 * Comprehensive Form Validation Utility
 * Provides validation rules and helpers for trading forms
 */

export interface ValidationRule {
  required?: boolean;
  min?: number;
  max?: number;
  pattern?: RegExp;
  custom?: (value: any) => string | null;
  message?: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

export interface FieldValidation {
  value: any;
  rules: ValidationRule[];
  label?: string;
}

export class FormValidator {
  /**
   * Validate a single field
   */
  static validateField(field: FieldValidation): ValidationResult {
    const { value, rules, label = 'Field' } = field;
    const errors: string[] = [];
    const warnings: string[] = [];

    for (const rule of rules) {
      const error = this.validateRule(value, rule, label);
      if (error) {
        if (rule.message) {
          errors.push(rule.message);
        } else {
          errors.push(error);
        }
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validate multiple fields
   */
  static validateFields(fields: Record<string, FieldValidation>): Record<string, ValidationResult> {
    const results: Record<string, ValidationResult> = {};

    for (const [fieldName, field] of Object.entries(fields)) {
      results[fieldName] = this.validateField(field);
    }

    return results;
  }

  /**
   * Validate a single rule
   */
  private static validateRule(value: any, rule: ValidationRule, label: string): string | null {
    // Required validation
    if (rule.required && (value === null || value === undefined || value === '')) {
      return `${label} is required`;
    }

    // Skip other validations if value is empty and not required
    if (value === null || value === undefined || value === '') {
      return null;
    }

    // Min validation
    if (rule.min !== undefined) {
      const numValue = Number(value);
      if (!isNaN(numValue) && numValue < rule.min) {
        return `${label} must be at least ${rule.min}`;
      }
    }

    // Max validation
    if (rule.max !== undefined) {
      const numValue = Number(value);
      if (!isNaN(numValue) && numValue > rule.max) {
        return `${label} must be no more than ${rule.max}`;
      }
    }

    // Pattern validation
    if (rule.pattern && typeof value === 'string') {
      if (!rule.pattern.test(value)) {
        return `${label} format is invalid`;
      }
    }

    // Custom validation
    if (rule.custom) {
      const customError = rule.custom(value);
      if (customError) {
        return customError;
      }
    }

    return null;
  }

  /**
   * Common validation rules for trading forms
   */
  static getTradingRules() {
    return {
      // Order size validation
      size: [
        { required: true, message: 'Order size is required' },
        { min: 0.0001, message: 'Order size must be greater than 0' },
        { max: 1000000, message: 'Order size is too large' },
        {
          custom: (value: number) => {
            // Check for reasonable precision (max 8 decimal places)
            const str = value.toString();
            const decimalPlaces = str.includes('.') ? str.split('.')[1].length : 0;
            if (decimalPlaces > 8) {
              return 'Order size has too many decimal places (max 8)';
            }
            return null;
          }
        }
      ],

      // Price validation
      price: [
        { required: true, message: 'Price is required for limit orders' },
        { min: 0.0001, message: 'Price must be greater than 0' },
        { max: 1000000, message: 'Price is too high' },
        {
          custom: (value: number) => {
            // Check for reasonable precision (max 8 decimal places)
            const str = value.toString();
            const decimalPlaces = str.includes('.') ? str.split('.')[1].length : 0;
            if (decimalPlaces > 8) {
              return 'Price has too many decimal places (max 8)';
            }
            return null;
          }
        }
      ],

      // Leverage validation
      leverage: [
        { required: true, message: 'Leverage is required' },
        { min: 1, message: 'Leverage must be at least 1x' },
        { max: 100, message: 'Leverage cannot exceed 100x' },
        {
          custom: (value: number) => {
            // Leverage should be whole numbers
            if (!Number.isInteger(value)) {
              return 'Leverage must be a whole number';
            }
            return null;
          }
        }
      ],

      // Stop price validation
      stopPrice: [
        { required: true, message: 'Stop price is required' },
        { min: 0.0001, message: 'Stop price must be greater than 0' },
        { max: 1000000, message: 'Stop price is too high' }
      ],

      // Trailing distance validation
      trailingDistance: [
        { required: true, message: 'Trailing distance is required' },
        { min: 0.01, message: 'Trailing distance must be at least 0.01' },
        { max: 1000, message: 'Trailing distance is too large' }
      ],

      // Email validation
      email: [
        { required: true, message: 'Email is required' },
        {
          pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
          message: 'Please enter a valid email address'
        }
      ],

      // Password validation
      password: [
        { required: true, message: 'Password is required' },
        { min: 8, message: 'Password must be at least 8 characters' },
        {
          pattern: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
          message: 'Password must contain at least one lowercase letter, one uppercase letter, and one number'
        }
      ]
    };
  }

  /**
   * Validate trading order form
   */
  static validateTradingOrder(formData: {
    size?: number;
    price?: number;
    leverage?: number;
    orderType?: string;
    side?: string;
    stopPrice?: number;
    trailingDistance?: number;
  }): ValidationResult {
    const rules = this.getTradingRules();
    const errors: string[] = [];
    const warnings: string[] = [];

    // Validate size
    if (formData.size !== undefined) {
      const sizeResult = this.validateField({
        value: formData.size,
        rules: rules.size,
        label: 'Order size'
      });
      errors.push(...sizeResult.errors);
      warnings.push(...sizeResult.warnings);
    }

    // Validate price for limit orders
    if (formData.orderType === 'limit' && formData.price !== undefined) {
      const priceResult = this.validateField({
        value: formData.price,
        rules: rules.price,
        label: 'Price'
      });
      errors.push(...priceResult.errors);
      warnings.push(...priceResult.warnings);
    }

    // Validate leverage
    if (formData.leverage !== undefined) {
      const leverageResult = this.validateField({
        value: formData.leverage,
        rules: rules.leverage,
        label: 'Leverage'
      });
      errors.push(...leverageResult.errors);
      warnings.push(...leverageResult.warnings);
    }

    // Validate stop price for stop orders
    if (formData.orderType === 'stop' && formData.stopPrice !== undefined) {
      const stopPriceResult = this.validateField({
        value: formData.stopPrice,
        rules: rules.stopPrice,
        label: 'Stop price'
      });
      errors.push(...stopPriceResult.errors);
      warnings.push(...stopPriceResult.warnings);
    }

    // Validate trailing distance for trailing stop orders
    if (formData.orderType === 'trailing_stop' && formData.trailingDistance !== undefined) {
      const trailingResult = this.validateField({
        value: formData.trailingDistance,
        rules: rules.trailingDistance,
        label: 'Trailing distance'
      });
      errors.push(...trailingResult.errors);
      warnings.push(...trailingResult.warnings);
    }

    // Cross-field validations
    if (formData.orderType === 'limit' && formData.price && formData.side) {
      // For buy orders, price should be reasonable compared to market
      if (formData.side === 'buy' && formData.price > 0) {
        // This would need current market price for proper validation
        // For now, just check if price is not extremely high
        if (formData.price > 1000000) {
          warnings.push('Buy price seems unusually high. Please verify.');
        }
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Get user-friendly error messages
   */
  static getErrorMessage(error: string, context?: { field?: string; value?: any }): string {
    const { field = 'Field', value } = context || {};

    // Common error patterns
    if (error.includes('required')) {
      return `${field} is required`;
    }

    if (error.includes('must be at least')) {
      return `${field} ${error}`;
    }

    if (error.includes('cannot exceed')) {
      return `${field} ${error}`;
    }

    if (error.includes('too many decimal places')) {
      return `${field} has too many decimal places`;
    }

    if (error.includes('format is invalid')) {
      return `${field} format is invalid`;
    }

    // Return original error if no pattern matches
    return error;
  }

  /**
   * Format validation errors for display
   */
  static formatErrors(errors: string[]): string[] {
    return errors.map(error => this.getErrorMessage(error));
  }

  /**
   * Check if form has any validation errors
   */
  static hasErrors(validationResults: Record<string, ValidationResult>): boolean {
    return Object.values(validationResults).some(result => !result.isValid);
  }

  /**
   * Get all errors from validation results
   */
  static getAllErrors(validationResults: Record<string, ValidationResult>): string[] {
    const allErrors: string[] = [];
    Object.values(validationResults).forEach(result => {
      allErrors.push(...result.errors);
    });
    return allErrors;
  }

  /**
   * Get all warnings from validation results
   */
  static getAllWarnings(validationResults: Record<string, ValidationResult>): string[] {
    const allWarnings: string[] = [];
    Object.values(validationResults).forEach(result => {
      allWarnings.push(...result.warnings);
    });
    return allWarnings;
  }
}

export default FormValidator;
