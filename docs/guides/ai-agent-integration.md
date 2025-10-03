# AI Agent Integration Guide: ML Trading Strategy Implementation

## Overview
This guide provides specific prompting strategies and workflows for integrating the ML trading strategy specification into your existing codebase using AI coding agents (Claude Code, Cursor, and the new Amazon Kiro agentic IDE).

## Pre-Integration Preparation

### 1. Project Context Documentation
Create a comprehensive context file for the AI agents:

```markdown
# PROJECT_CONTEXT.md

## Current Architecture
- Language: Python
- ML Framework: PyTorch (GPU-enabled)
- Data Sources: [List your data feeds]
- Trading Framework: [Your current framework]
- Existing Indicators: RSI, CCI, Wave Trend
- Database: [Your database setup]
- Infrastructure: [Cloud/local setup]

## Key Files Structure
```
project/
├── data/
│   ├── feeds.py          # Data ingestion
│   └── preprocessing.py  # Current feature engineering
├── strategies/
│   ├── base_strategy.py  # Your base strategy class
│   └── indicators.py     # RSI, CCI, Wave Trend
├── execution/
│   ├── trader.py         # Current trading logic
│   └── risk_manager.py   # Existing risk management
└── config/
    └── settings.py       # Configuration
```

## Dependencies
- torch==2.0.0
- pandas==1.5.3
- numpy==1.24.0
- [Add your current dependencies]

## Trading Logic Flow
[Describe your current trading flow]
```

### 2. Existing Code Samples
Prepare representative samples of your key files that the AI agent should understand and integrate with.

## Agent-Specific Integration Strategies

### Claude Code Integration

#### Initial Setup Prompt
```bash
# Use this prompt when starting with Claude Code
claude-code --prompt "I need to integrate advanced ML models into my existing crypto trading system. Here's my current project structure and a detailed ML specification. Please analyze my existing codebase and create an integration plan that:

1. Preserves all existing functionality
2. Adds the new ML components incrementally
3. Maintains my current coding style and patterns
4. Provides clear migration steps

Focus on backward compatibility and gradual deployment."
```

#### Iterative Development Prompts
```bash
# For feature engineering integration
claude-code --prompt "Looking at my existing indicators.py file, integrate the new ML feature engineering classes while maintaining the same interface pattern. Ensure the new features can be toggled on/off via configuration."

# For model training pipeline
claude-code --prompt "Create the model training pipeline that integrates with my existing data preprocessing. Use my current data loading patterns and add the new ML training components as separate modules."

# For signal integration
claude-code --prompt "Integrate ML predictions into my existing trading signal generation without breaking current strategy logic. The integration should be additive and configurable."
```

### Cursor IDE Integration

#### Workspace Setup
1. **Open your project in Cursor**
2. **Add the ML specification as a reference document**
3. **Use Cursor's composer feature with context**

#### Effective Cursor Prompts

##### For Code Analysis
```
@codebase Analyze my current trading system and identify the best integration points for the ML components from the specification. Focus on:
- Where to add new feature engineering
- How to integrate model predictions
- Minimal changes to existing code
- Configuration-driven approach
```

##### For Implementation
```
@codebase Looking at my existing [filename], implement the [specific ML component] while:
- Following my current code patterns
- Using my existing error handling
- Maintaining the same configuration style
- Adding comprehensive logging like my other modules
```

##### For Testing Integration
```
@codebase Create unit tests for the new ML integration that:
- Mock the ML model predictions
- Test backward compatibility
- Validate configuration options
- Follow my existing test patterns
```

### Amazon Kiro Agentic IDE Integration

#### Understanding Kiro's Unique Features
Kiro is an agentic IDE that helps you do your best work with features such as specs, steering, and hooks. Amazon Kiro AI IDE, an advanced tool for spec-driven development and rapid code prototyping was just released in preview and offers intelligent agentic workflows for automation.

#### Spec-Driven Development with Kiro
Since Kiro excels at spec-driven development, leverage your ML specification directly:

```python
# Create a Kiro Spec file: ml_integration.kspec
"""
KIRO SPECIFICATION: ML Trading Strategy Integration

TARGET: Existing crypto trading system
GOAL: Integrate ML models while preserving existing functionality
CONSTRAINTS: 
- Backward compatibility required
- Gradual rollout approach
- PyTorch GPU optimization
- Configuration-driven features

INTEGRATION_POINTS:
1. Feature engineering enhancement
2. Model training pipeline
3. Prediction engine integration
4. Signal blending logic

EXISTING_PATTERNS:
- Class-based architecture
- Configuration management via settings.py
- Logging with structured format
- Error handling with fallbacks
"""
```

#### Kiro Steering Prompts
Use Kiro's steering feature to guide the implementation:

```python
# KIRO_STEERING: Feature Engineering Integration
"""
Analyze existing indicators.py and extend with ML features.
Maintain same interface pattern.
Add configuration toggles.
Preserve existing RSI, CCI, Wave Trend calculations.
"""

# KIRO_STEERING: Model Integration  
"""
Create ml/ module structure compatible with existing data pipeline.
Use existing data loading patterns.
Implement training/prediction separation.
Add model versioning and management.
"""

# KIRO_STEERING: Signal Blending
"""
Extend current SignalGenerator class.
Add ML prediction input option.
Implement confidence-based blending.
Maintain fallback to traditional signals.
"""
```

#### Kiro Hooks for Integration Points
Leverage Kiro's hooks feature for key integration points:

```python
# KIRO_HOOK: pre_signal_generation
def integrate_ml_prediction(traditional_signal, market_data):
    """Hook to add ML prediction before final signal generation"""
    if config.enable_ml and ml_engine.is_ready():
        ml_prediction = ml_engine.predict(market_data)
        return blend_signals(traditional_signal, ml_prediction)
    return traditional_signal

# KIRO_HOOK: post_feature_calculation  
def enhance_features_with_ml(base_features, market_data):
    """Hook to add ML-derived features after basic calculation"""
    if config.enable_ml_features:
        ml_features = ml_feature_engine.calculate(market_data)
        return {**base_features, **ml_features}
    return base_features

# KIRO_HOOK: pre_position_sizing
def ml_position_adjustment(base_position, ml_confidence):
    """Hook to adjust position size based on ML confidence"""
    if ml_confidence > config.high_confidence_threshold:
        return base_position * config.ml_confidence_multiplier
    return base_position
```

#### Kiro Agentic Workflow Setup
Configure Kiro's agentic automation for your integration:

```yaml
# .kiro/workflows/ml_integration.yml
name: "ML Trading Integration Workflow"
description: "Automated integration of ML components"

agents:
  - name: "code_analyzer"
    task: "Analyze existing codebase patterns"
    
  - name: "feature_integrator"  
    task: "Integrate ML feature engineering"
    depends_on: ["code_analyzer"]
    
  - name: "model_integrator"
    task: "Add model training and prediction"
    depends_on: ["feature_integrator"]
    
  - name: "signal_integrator"
    task: "Integrate ML signals with existing logic"
    depends_on: ["model_integrator"]
    
  - name: "test_generator"
    task: "Generate comprehensive test suite"
    depends_on: ["signal_integrator"]

hooks:
  pre_integration:
    - validate_existing_functionality
    - backup_current_codebase
    
  post_integration:
    - run_backward_compatibility_tests
    - validate_performance_benchmarks
```

## Strategic Integration Workflow

### Phase 1: Analysis and Planning
```markdown
**AI Agent Prompt Template:**

"I have an existing crypto trading system and want to integrate advanced ML capabilities using [Claude Code/Cursor/Kiro]. Please:

1. **Code Review**: Analyze my existing codebase and identify:
   - Current architecture patterns
   - Extension points for ML integration
   - Potential conflicts or dependencies
   - Best practices I'm already following

2. **Integration Plan**: Create a step-by-step plan that:
   - Preserves existing functionality
   - Minimizes code changes
   - Allows gradual rollout
   - Maintains testability

3. **Risk Assessment**: Identify potential issues:
   - Breaking changes
   - Performance impacts
   - Integration complexity
   - Rollback strategies

Please provide specific file changes and code examples."
```

### Phase 2: Incremental Implementation

#### Step 1: Feature Engineering Integration
```markdown
**Prompt for AI Agent:**

"Integrate the advanced feature engineering from the ML specification into my existing feature pipeline. Requirements:

- Extend my current FeatureBase class
- Add new features as optional modules
- Maintain existing feature calculation methods
- Use configuration flags to enable/disable new features
- Follow my current naming conventions
- Add the same level of documentation

Show me the exact code changes needed for [specific file]."
```

#### Step 2: Model Integration
```markdown
**Prompt for AI Agent:**

"Add the ML model training and prediction pipeline to my project. Requirements:

- Create new ml/ directory with proper structure
- Integrate with my existing data loading
- Use my current configuration management
- Add proper error handling and logging
- Make model predictions optional (fallback to existing logic)
- Follow my current class inheritance patterns

Focus on the ModelTrainer class first, then PredictionEngine."
```

#### Step 3: Signal Integration
```markdown
**Prompt for AI Agent:**

"Integrate ML predictions into my existing trading signal generation. Requirements:

- Modify my SignalGenerator class minimally
- Add ML predictions as additional input
- Maintain existing signal logic as fallback
- Use confidence-based blending
- Add comprehensive logging
- Ensure backward compatibility

Show me exactly how to modify [my_signal_file.py]."
```

### Phase 3: Testing and Validation

#### Testing Strategy Prompt
```markdown
**AI Agent Prompt:**

"Create a comprehensive testing strategy for the ML integration:

1. **Unit Tests**: For each new ML component
2. **Integration Tests**: End-to-end with existing system
3. **Backward Compatibility Tests**: Ensure nothing breaks
4. **Performance Tests**: Measure impact on execution speed
5. **Mock Tests**: For ML models during development

Use my existing test framework and patterns. Show me specific test implementations."
```

## Advanced Prompting Techniques

### Context-Aware Prompting
```markdown
**Multi-Context Prompt:**

"I'm integrating ML into my trading system. Here's my context:

**Current System:**
- [Paste key existing code snippets]
- [Current performance metrics]
- [Existing architecture decisions]

**ML Integration Goal:**
- [Specific ML specification sections]
- [Performance targets]
- [Integration constraints]

**Specific Request:**
[Detailed implementation request]

Please provide code that seamlessly integrates with my existing patterns."
```

### Iterative Refinement Prompts
```markdown
**Refinement Prompt Template:**

"The previous implementation needs adjustment. Here's what I found:

**Issues Encountered:**
- [Specific problems]
- [Performance concerns]
- [Integration conflicts]

**My Existing Code Style:**
- [Show examples of your patterns]

**Requested Changes:**
- [Specific modifications needed]

Please revise the implementation to address these issues while maintaining integration with my existing codebase."
```

## Common Integration Patterns

### 1. Configuration-Driven Integration
```python
# Prompt: "Make all ML features configurable"
class MLConfig:
    def __init__(self):
        self.enable_ml_features = True
        self.enable_ml_predictions = True
        self.ml_confidence_threshold = 0.7
        self.fallback_to_traditional = True

# Usage in existing code
if config.enable_ml_predictions and ml_confidence > config.ml_confidence_threshold:
    signal = ml_prediction
elif config.fallback_to_traditional:
    signal = traditional_strategy.get_signal()
```

### 2. Decorator-Based Integration
```python
# Prompt: "Use decorators to add ML functionality to existing methods"
def with_ml_enhancement(confidence_threshold=0.7):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get traditional signal
            traditional_signal = func(*args, **kwargs)
            
            # Get ML enhancement
            if ml_engine.is_available():
                ml_signal = ml_engine.predict(*args)
                if ml_signal.confidence > confidence_threshold:
                    return blend_signals(traditional_signal, ml_signal)
            
            return traditional_signal
        return wrapper
    return decorator
```

### 3. Factory Pattern Integration
```python
# Prompt: "Use factory pattern to integrate ML models"
class PredictionEngineFactory:
    @staticmethod
    def create_engine(config):
        if config.use_ml and ml_models_available():
            return MLPredictionEngine(config.ml_models)
        else:
            return TraditionalPredictionEngine(config.traditional_indicators)
```

## Debugging and Troubleshooting Prompts

### Performance Issues
```markdown
**Performance Debugging Prompt:**

"The ML integration is causing performance issues. Help me optimize:

**Current Issues:**
- [Specific performance problems]
- [Timing measurements]

**My Existing Code:**
- [Show problematic code]

**Requirements:**
- Maintain prediction accuracy
- Reduce latency to <100ms
- Use my existing caching patterns

Please provide optimized implementation with profiling suggestions."
```

### Integration Conflicts
```markdown
**Conflict Resolution Prompt:**

"I'm getting conflicts between my existing code and the new ML integration:

**Error Messages:**
- [Specific errors]

**Conflicting Code:**
- [Show conflicting sections]

**My Constraints:**
- Cannot modify [specific existing code]
- Must maintain [specific functionality]

Please provide a solution that works around these constraints."
```

## Validation and Testing Prompts

### Comprehensive Testing
```markdown
**Testing Strategy Prompt:**

"Create a complete testing suite for the ML integration:

1. **Functional Tests**: Verify ML predictions work correctly
2. **Performance Tests**: Ensure acceptable speed
3. **Integration Tests**: Test with existing system
4. **Edge Case Tests**: Handle missing data, model failures
5. **Regression Tests**: Ensure existing functionality intact

Use my existing test framework [pytest/unittest] and follow my test patterns."
```

### Gradual Rollout Strategy
```markdown
**Rollout Planning Prompt:**

"Design a safe rollout strategy for the ML integration:

1. **Development Phase**: Local testing with historical data
2. **Staging Phase**: Paper trading with live data
3. **Production Phase**: Gradual capital allocation

Include:
- Feature flags for quick rollback
- Monitoring and alerting
- A/B testing framework
- Performance comparison tools

Show me the implementation for each phase."
```

## Best Practices for AI Agent Collaboration

### 1. Always Provide Context
- Share your existing code patterns
- Explain your constraints and requirements
- Include performance expectations
- Mention your coding standards

### 2. Iterate and Refine
- Start with basic integration
- Test thoroughly at each step
- Refine based on results
- Use agent feedback to improve

### 3. Maintain Documentation
- Document all changes made by AI agents
- Keep track of what works and what doesn't
- Create integration guides for future reference
- Update your project documentation

### 4. Validate Everything
- Never blindly accept AI-generated code
- Test all integrations thoroughly
- Review for security implications
- Ensure performance meets requirements

### 5. Leverage Kiro's Unique Capabilities
- Use spec-driven development for complex integrations
- Implement steering for guided code generation
- Set up hooks for clean integration points
- Configure agentic workflows for automation

This guide will help you effectively collaborate with AI coding agents to integrate the ML trading strategy into your existing system while maintaining stability and performance.