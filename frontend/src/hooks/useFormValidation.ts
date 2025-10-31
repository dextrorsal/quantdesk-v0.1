import { useState, useCallback, useMemo } from 'react';
import { FormValidator, ValidationResult, FieldValidation } from '../utils/formValidation';

export interface UseFormValidationOptions {
  validateOnChange?: boolean;
  validateOnBlur?: boolean;
  showWarnings?: boolean;
}

export interface FormField {
  value: any;
  error: string | null;
  warning: string | null;
  touched: boolean;
  dirty: boolean;
}

export interface UseFormValidationReturn<T extends Record<string, any>> {
  fields: Record<keyof T, FormField>;
  errors: string[];
  warnings: string[];
  isValid: boolean;
  isDirty: boolean;
  setValue: (field: keyof T, value: any) => void;
  setTouched: (field: keyof T, touched?: boolean) => void;
  validateField: (field: keyof T) => void;
  validateForm: () => boolean;
  reset: () => void;
  resetField: (field: keyof T) => void;
}

/**
 * Hook for form validation with real-time feedback
 */
export function useFormValidation<T extends Record<string, any>>(
  initialValues: T,
  validationRules: Record<keyof T, FieldValidation['rules']>,
  options: UseFormValidationOptions = {}
): UseFormValidationReturn<T> {
  const {
    validateOnChange = true,
    validateOnBlur = true,
    showWarnings = true
  } = options;

  // Initialize form fields
  const [fields, setFields] = useState<Record<keyof T, FormField>>(() => {
    const initialFields = {} as Record<keyof T, FormField>;
    Object.keys(initialValues).forEach(key => {
      initialFields[key as keyof T] = {
        value: initialValues[key as keyof T],
        error: null,
        warning: null,
        touched: false,
        dirty: false
      };
    });
    return initialFields;
  });

  // Memoized validation results
  const validationResults = useMemo(() => {
    const fieldValidations: Record<string, FieldValidation> = {};
    
    Object.keys(fields).forEach(key => {
      const field = fields[key as keyof T];
      fieldValidations[key] = {
        value: field.value,
        rules: validationRules[key as keyof T] || [],
        label: key.toString()
      };
    });

    return FormValidator.validateFields(fieldValidations);
  }, [fields, validationRules]);

  // Update field errors and warnings
  const updateFieldValidation = useCallback((fieldKey: keyof T) => {
    const result = validationResults[fieldKey as string];
    if (result) {
      setFields(prev => ({
        ...prev,
        [fieldKey]: {
          ...prev[fieldKey],
          error: result.errors[0] || null,
          warning: showWarnings ? (result.warnings[0] || null) : null
        }
      }));
    }
  }, [validationResults, showWarnings]);

  // Set field value
  const setValue = useCallback((field: keyof T, value: any) => {
    setFields(prev => ({
      ...prev,
      [field]: {
        ...prev[field],
        value,
        dirty: true
      }
    }));

    // Validate on change if enabled
    if (validateOnChange) {
      setTimeout(() => updateFieldValidation(field), 0);
    }
  }, [validateOnChange, updateFieldValidation]);

  // Set field touched state
  const setTouched = useCallback((field: keyof T, touched: boolean = true) => {
    setFields(prev => ({
      ...prev,
      [field]: {
        ...prev[field],
        touched
      }
    }));

    // Validate on blur if enabled
    if (validateOnBlur && touched) {
      setTimeout(() => updateFieldValidation(field), 0);
    }
  }, [validateOnBlur, updateFieldValidation]);

  // Validate single field
  const validateField = useCallback((field: keyof T) => {
    updateFieldValidation(field);
  }, [updateFieldValidation]);

  // Validate entire form
  const validateForm = useCallback((): boolean => {
    let isValid = true;
    
    Object.keys(fields).forEach(key => {
      const fieldKey = key as keyof T;
      updateFieldValidation(fieldKey);
      
      const result = validationResults[key];
      if (result && !result.isValid) {
        isValid = false;
      }
    });

    return isValid;
  }, [fields, validationResults, updateFieldValidation]);

  // Reset form
  const reset = useCallback(() => {
    setFields(prev => {
      const resetFields = {} as Record<keyof T, FormField>;
      Object.keys(prev).forEach(key => {
        resetFields[key as keyof T] = {
          value: initialValues[key as keyof T],
          error: null,
          warning: null,
          touched: false,
          dirty: false
        };
      });
      return resetFields;
    });
  }, [initialValues]);

  // Reset single field
  const resetField = useCallback((field: keyof T) => {
    setFields(prev => ({
      ...prev,
      [field]: {
        value: initialValues[field],
        error: null,
        warning: null,
        touched: false,
        dirty: false
      }
    }));
  }, [initialValues]);

  // Computed values
  const errors = useMemo(() => {
    return FormValidator.getAllErrors(validationResults);
  }, [validationResults]);

  const warnings = useMemo(() => {
    return showWarnings ? FormValidator.getAllWarnings(validationResults) : [];
  }, [validationResults, showWarnings]);

  const isValid = useMemo(() => {
    return !FormValidator.hasErrors(validationResults);
  }, [validationResults]);

  const isDirty = useMemo(() => {
    return Object.values(fields).some(field => field.dirty);
  }, [fields]);

  return {
    fields,
    errors,
    warnings,
    isValid,
    isDirty,
    setValue,
    setTouched,
    validateField,
    validateForm,
    reset,
    resetField
  };
}

/**
 * Hook for trading form validation
 */
export function useTradingFormValidation(initialValues: {
  size?: number;
  price?: number;
  leverage?: number;
  orderType?: string;
  side?: string;
  stopPrice?: number;
  trailingDistance?: number;
}) {
  const validationRules = {
    size: FormValidator.getTradingRules().size,
    price: FormValidator.getTradingRules().price,
    leverage: FormValidator.getTradingRules().leverage,
    stopPrice: FormValidator.getTradingRules().stopPrice,
    trailingDistance: FormValidator.getTradingRules().trailingDistance
  };

  return useFormValidation(initialValues, validationRules, {
    validateOnChange: true,
    validateOnBlur: true,
    showWarnings: true
  });
}

export default useFormValidation;
