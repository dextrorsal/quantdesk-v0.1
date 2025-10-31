import React from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';
import { FormField } from '../hooks/useFormValidation';

export interface ValidatedFormFieldProps {
  field: FormField;
  label: string;
  type?: 'text' | 'number' | 'email' | 'password';
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  className?: string;
  inputClassName?: string;
  labelClassName?: string;
  onChange: (value: any) => void;
  onBlur?: () => void;
  onFocus?: () => void;
  showValidation?: boolean;
}

export const ValidatedFormField: React.FC<ValidatedFormFieldProps> = ({
  field,
  label,
  type = 'text',
  placeholder,
  min,
  max,
  step,
  disabled = false,
  className = '',
  inputClassName = '',
  labelClassName = '',
  onChange,
  onBlur,
  onFocus,
  showValidation = true
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value;
    onChange(value);
  };

  const getInputClasses = () => {
    let baseClasses = 'w-full py-2 px-3 rounded-lg border focus:outline-none transition-colors';
    
    if (disabled) {
      baseClasses += ' bg-gray-700 text-gray-400 cursor-not-allowed';
    } else {
      baseClasses += ' bg-gray-700 text-white';
    }

    if (showValidation && field.touched) {
      if (field.error) {
        baseClasses += ' border-red-500 focus:border-red-400';
      } else if (field.warning) {
        baseClasses += ' border-yellow-500 focus:border-yellow-400';
      } else {
        baseClasses += ' border-green-500 focus:border-green-400';
      }
    } else {
      baseClasses += ' border-gray-600 focus:border-blue-500';
    }

    return `${baseClasses} ${inputClassName}`;
  };

  const getLabelClasses = () => {
    let baseClasses = 'block text-sm font-medium mb-2';
    
    if (disabled) {
      baseClasses += ' text-gray-500';
    } else {
      baseClasses += ' text-gray-300';
    }

    return `${baseClasses} ${labelClassName}`;
  };

  return (
    <div className={`space-y-1 ${className}`}>
      <label className={getLabelClasses()}>
        {label}
        {showValidation && field.touched && !field.error && (
          <CheckCircle className="inline w-4 h-4 text-green-400 ml-1" />
        )}
      </label>
      
      <div className="relative">
        <input
          type={type}
          value={field.value || ''}
          onChange={handleChange}
          onBlur={onBlur}
          onFocus={onFocus}
          placeholder={placeholder}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          className={getInputClasses()}
        />
        
        {showValidation && field.touched && (field.error || field.warning) && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            <AlertCircle className="w-4 h-4 text-red-400" />
          </div>
        )}
      </div>
      
      {showValidation && field.touched && (
        <>
          {field.error && (
            <p className="text-red-400 text-xs flex items-center">
              <AlertCircle className="w-3 h-3 mr-1" />
              {field.error}
            </p>
          )}
          {field.warning && !field.error && (
            <p className="text-yellow-400 text-xs flex items-center">
              <AlertCircle className="w-3 h-3 mr-1" />
              {field.warning}
            </p>
          )}
        </>
      )}
    </div>
  );
};

export interface ValidatedSelectFieldProps {
  field: FormField;
  label: string;
  options: Array<{ value: any; label: string }>;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  onChange: (value: any) => void;
  onBlur?: () => void;
  showValidation?: boolean;
}

export const ValidatedSelectField: React.FC<ValidatedSelectFieldProps> = ({
  field,
  label,
  options,
  placeholder,
  disabled = false,
  className = '',
  onChange,
  onBlur,
  showValidation = true
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    onChange(value);
  };

  const getSelectClasses = () => {
    let baseClasses = 'w-full py-2 px-3 rounded-lg border focus:outline-none transition-colors';
    
    if (disabled) {
      baseClasses += ' bg-gray-700 text-gray-400 cursor-not-allowed';
    } else {
      baseClasses += ' bg-gray-700 text-white';
    }

    if (showValidation && field.touched) {
      if (field.error) {
        baseClasses += ' border-red-500 focus:border-red-400';
      } else if (field.warning) {
        baseClasses += ' border-yellow-500 focus:border-yellow-400';
      } else {
        baseClasses += ' border-green-500 focus:border-green-400';
      }
    } else {
      baseClasses += ' border-gray-600 focus:border-blue-500';
    }

    return baseClasses;
  };

  const getLabelClasses = () => {
    let baseClasses = 'block text-sm font-medium mb-2';
    
    if (disabled) {
      baseClasses += ' text-gray-500';
    } else {
      baseClasses += ' text-gray-300';
    }

    return baseClasses;
  };

  return (
    <div className={`space-y-1 ${className}`}>
      <label className={getLabelClasses()}>
        {label}
        {showValidation && field.touched && !field.error && (
          <CheckCircle className="inline w-4 h-4 text-green-400 ml-1" />
        )}
      </label>
      
      <select
        value={field.value || ''}
        onChange={handleChange}
        onBlur={onBlur}
        disabled={disabled}
        className={getSelectClasses()}
      >
        {placeholder && (
          <option value="" disabled>
            {placeholder}
          </option>
        )}
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      
      {showValidation && field.touched && (
        <>
          {field.error && (
            <p className="text-red-400 text-xs flex items-center">
              <AlertCircle className="w-3 h-3 mr-1" />
              {field.error}
            </p>
          )}
          {field.warning && !field.error && (
            <p className="text-yellow-400 text-xs flex items-center">
              <AlertCircle className="w-3 h-3 mr-1" />
              {field.warning}
            </p>
          )}
        </>
      )}
    </div>
  );
};

export default ValidatedFormField;
