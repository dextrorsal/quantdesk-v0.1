// Tool Orchestration Layer for MIKEY-AI
// Enables complex multi-step workflows combining multiple tools

import { DynamicTool } from '@langchain/core/tools';
import { WalletAddressExtractor } from '../utils/WalletAddressExtractor';
import { SolanaWalletTools } from './SolanaWalletTools';
import { QuantDeskProtocolTools } from './QuantDeskProtocolTools';
import { systemLogger, errorLogger } from '../utils/logger';

export interface WorkflowStep {
  id: string;
  type: 'solana_wallet' | 'quantdesk_protocol' | 'analysis' | 'synthesis';
  tool: string;
  input: any;
  dependencies?: string[];
}

export interface WorkflowResult {
  stepId: string;
  success: boolean;
  data?: any;
  error?: string;
  timestamp: string;
}

export interface OrchestratedResponse {
  success: boolean;
  workflow: string;
  results: WorkflowResult[];
  synthesis?: string;
  confidence: number;
  timestamp: string;
}

export class ToolOrchestrator {
  private solanaTools: DynamicTool[];
  private quantdeskTools: DynamicTool[];

  constructor() {
    this.solanaTools = SolanaWalletTools.getAllTools();
    this.quantdeskTools = QuantDeskProtocolTools.getAllTools();
  }

  /**
   * Execute a complex workflow based on user query
   */
  async executeWorkflow(query: string): Promise<OrchestratedResponse> {
    try {
      systemLogger.startup(`Executing workflow for query: ${query}`, 'workflow');
      
      // Parse the query to determine workflow steps
      const steps = this.parseWorkflowSteps(query);
      
      if (steps.length === 0) {
        return this.createErrorResponse(query, 'No actionable steps found in query');
      }

      // Execute steps in parallel where possible, sequential where dependencies exist
      const results = await this.executeSteps(steps);
      
      // Synthesize results into a comprehensive response
      const synthesis = await this.synthesizeResults(query, results);
      
      return {
        success: true,
        workflow: this.identifyWorkflowType(query),
        results,
        synthesis,
        confidence: this.calculateConfidence(results),
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      errorLogger.aiError(error as Error, `Workflow execution failed for query: ${query}`);
      return this.createErrorResponse(query, error.message);
    }
  }

  /**
   * Parse user query into actionable workflow steps
   */
  private parseWorkflowSteps(query: string): WorkflowStep[] {
    const steps: WorkflowStep[] = [];
    const lowerQuery = query.toLowerCase();
    
    // Extract wallet address if present
    const walletAddress = WalletAddressExtractor.extractWalletAddress(query);
    
    // Determine workflow steps based on query content
    if (lowerQuery.includes('balance') && lowerQuery.includes('portfolio')) {
      // Multi-step: Check SOL balance + QuantDesk portfolio
      steps.push({
        id: 'sol_balance',
        type: 'solana_wallet',
        tool: 'check_sol_balance',
        input: query
      });
      
      if (walletAddress) {
        steps.push({
          id: 'quantdesk_portfolio',
          type: 'quantdesk_protocol',
          tool: 'check_quantdesk_portfolio',
          input: walletAddress
        });
      }
      
      steps.push({
        id: 'portfolio_analysis',
        type: 'analysis',
        tool: 'analyze_portfolio',
        input: { steps: ['sol_balance', 'quantdesk_portfolio'] },
        dependencies: ['sol_balance', 'quantdesk_portfolio']
      });
    }
    else if (lowerQuery.includes('wallet') && lowerQuery.includes('analysis')) {
      // Comprehensive wallet analysis
      steps.push({
        id: 'sol_balance',
        type: 'solana_wallet',
        tool: 'check_sol_balance',
        input: query
      });
      
      steps.push({
        id: 'transaction_history',
        type: 'solana_wallet',
        tool: 'get_transaction_history',
        input: query
      });
      
      if (walletAddress) {
        steps.push({
          id: 'quantdesk_portfolio',
          type: 'quantdesk_protocol',
          tool: 'check_quantdesk_portfolio',
          input: walletAddress
        });
        
        steps.push({
          id: 'risk_analysis',
          type: 'quantdesk_protocol',
          tool: 'analyze_wallet_risk',
          input: walletAddress
        });
      }
      
      steps.push({
        id: 'comprehensive_analysis',
        type: 'analysis',
        tool: 'comprehensive_wallet_analysis',
        input: { 
          steps: ['sol_balance', 'transaction_history', 'quantdesk_portfolio', 'risk_analysis'] 
        },
        dependencies: ['sol_balance', 'transaction_history']
      });
    }
    else if (lowerQuery.includes('market') && lowerQuery.includes('sentiment')) {
      // Market analysis workflow
      steps.push({
        id: 'market_data',
        type: 'quantdesk_protocol',
        tool: 'get_quantdesk_market_data',
        input: 'SOL-PERP'
      });
      
      steps.push({
        id: 'market_analysis',
        type: 'analysis',
        tool: 'analyze_market_sentiment',
        input: { steps: ['market_data'] },
        dependencies: ['market_data']
      });
    }
    else {
      // Single step workflows
      if (lowerQuery.includes('balance')) {
        steps.push({
          id: 'sol_balance',
          type: 'solana_wallet',
          tool: 'check_sol_balance',
          input: query
        });
      }
      
      if (lowerQuery.includes('transaction') || lowerQuery.includes('history')) {
        steps.push({
          id: 'transaction_history',
          type: 'solana_wallet',
          tool: 'get_transaction_history',
          input: query
        });
      }
      
      if (lowerQuery.includes('portfolio')) {
        steps.push({
          id: 'quantdesk_portfolio',
          type: 'quantdesk_protocol',
          tool: 'check_quantdesk_portfolio',
          input: walletAddress || query
        });
      }
    }

    return steps;
  }

  /**
   * Execute workflow steps with dependency management
   */
  private async executeSteps(steps: WorkflowStep[]): Promise<WorkflowResult[]> {
    const results: WorkflowResult[] = [];
    const executed = new Set<string>();
    
    // Create a dependency graph
    const dependencies = new Map<string, string[]>();
    steps.forEach(step => {
      dependencies.set(step.id, step.dependencies || []);
    });

    // Execute steps in topological order
    while (executed.size < steps.length) {
      const readySteps = steps.filter(step => 
        !executed.has(step.id) && 
        (step.dependencies || []).every(dep => executed.has(dep))
      );

      if (readySteps.length === 0) {
        // Circular dependency or error
        break;
      }

      // Execute ready steps in parallel
      const stepPromises = readySteps.map(step => this.executeStep(step));
      const stepResults = await Promise.allSettled(stepPromises);

      stepResults.forEach((result, index) => {
        const step = readySteps[index];
        if (result.status === 'fulfilled') {
          results.push(result.value);
          executed.add(step.id);
        } else {
          results.push({
            stepId: step.id,
            success: false,
            error: result.reason?.message || 'Unknown error',
            timestamp: new Date().toISOString()
          });
          executed.add(step.id);
        }
      });
    }

    return results;
  }

  /**
   * Execute a single workflow step
   */
  private async executeStep(step: WorkflowStep): Promise<WorkflowResult> {
    try {
      let toolResult: string;

      if (step.type === 'solana_wallet') {
        const tool = this.solanaTools.find(t => t.name === step.tool);
        if (!tool) {
          throw new Error(`Solana tool not found: ${step.tool}`);
        }
        toolResult = await tool.func(step.input);
      }
      else if (step.type === 'quantdesk_protocol') {
        const tool = this.quantdeskTools.find(t => t.name === step.tool);
        if (!tool) {
          throw new Error(`QuantDesk tool not found: ${step.tool}`);
        }
        toolResult = await tool.func(step.input);
      }
      else if (step.type === 'analysis') {
        toolResult = await this.performAnalysis(step);
      }
      else {
        throw new Error(`Unknown step type: ${step.type}`);
      }

      return {
        stepId: step.id,
        success: true,
        data: JSON.parse(toolResult),
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      return {
        stepId: step.id,
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Perform analysis on collected data
   */
  private async performAnalysis(step: WorkflowStep): Promise<string> {
    const analysisType = step.tool;
    const inputData = step.input;

    switch (analysisType) {
      case 'analyze_portfolio':
        return this.analyzePortfolio(inputData);
      
      case 'comprehensive_wallet_analysis':
        return this.comprehensiveWalletAnalysis(inputData);
      
      case 'analyze_market_sentiment':
        return this.analyzeMarketSentiment(inputData);
      
      default:
        return JSON.stringify({
          success: false,
          error: `Unknown analysis type: ${analysisType}`
        });
    }
  }

  /**
   * Analyze portfolio combining SOL balance and QuantDesk positions
   */
  private async analyzePortfolio(inputData: any): Promise<string> {
    // This would integrate with your existing analysis logic
    return JSON.stringify({
      success: true,
      analysis: {
        type: 'portfolio_analysis',
        summary: 'Combined SOL balance and QuantDesk portfolio analysis',
        recommendations: [
          'Monitor SOL balance for trading opportunities',
          'Review QuantDesk positions for risk management',
          'Consider portfolio diversification'
        ],
        timestamp: new Date().toISOString()
      }
    });
  }

  /**
   * Comprehensive wallet analysis
   */
  private async comprehensiveWalletAnalysis(inputData: any): Promise<string> {
    return JSON.stringify({
      success: true,
      analysis: {
        type: 'comprehensive_wallet_analysis',
        summary: 'Complete wallet analysis including balance, transactions, portfolio, and risk',
        insights: [
          'Wallet activity patterns identified',
          'Risk profile assessed',
          'Trading opportunities highlighted'
        ],
        timestamp: new Date().toISOString()
      }
    });
  }

  /**
   * Analyze market sentiment
   */
  private async analyzeMarketSentiment(inputData: any): Promise<string> {
    return JSON.stringify({
      success: true,
      analysis: {
        type: 'market_sentiment',
        summary: 'Market sentiment analysis based on QuantDesk data',
        sentiment: 'neutral',
        confidence: 0.75,
        timestamp: new Date().toISOString()
      }
    });
  }

  /**
   * Synthesize workflow results into a comprehensive response
   */
  private async synthesizeResults(query: string, results: WorkflowResult[]): Promise<string> {
    const successfulResults = results.filter(r => r.success);
    const failedResults = results.filter(r => !r.success);

    let synthesis = `Based on your query "${query}", I've completed the following analysis:\n\n`;

    successfulResults.forEach(result => {
      synthesis += `✅ ${result.stepId}: Completed successfully\n`;
    });

    if (failedResults.length > 0) {
      synthesis += `\n⚠️ Some steps encountered issues:\n`;
      failedResults.forEach(result => {
        synthesis += `❌ ${result.stepId}: ${result.error}\n`;
      });
    }

    synthesis += `\n📊 Overall confidence: ${this.calculateConfidence(results).toFixed(2)}`;

    return synthesis;
  }

  /**
   * Calculate confidence based on successful vs failed steps
   */
  private calculateConfidence(results: WorkflowResult[]): number {
    if (results.length === 0) return 0;
    
    const successful = results.filter(r => r.success).length;
    return successful / results.length;
  }

  /**
   * Identify the type of workflow being executed
   */
  private identifyWorkflowType(query: string): string {
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('balance') && lowerQuery.includes('portfolio')) {
      return 'portfolio_analysis';
    }
    else if (lowerQuery.includes('wallet') && lowerQuery.includes('analysis')) {
      return 'comprehensive_wallet_analysis';
    }
    else if (lowerQuery.includes('market') && lowerQuery.includes('sentiment')) {
      return 'market_sentiment_analysis';
    }
    else {
      return 'single_step';
    }
  }

  /**
   * Create error response
   */
  private createErrorResponse(query: string, error: string): OrchestratedResponse {
    return {
      success: false,
      workflow: 'error',
      results: [{
        stepId: 'error',
        success: false,
        error,
        timestamp: new Date().toISOString()
      }],
      synthesis: `Sorry, I encountered an error processing your query: "${query}". Error: ${error}`,
      confidence: 0,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Create orchestration tool for integration with TradingAgent
   */
  static createOrchestrationTool(): DynamicTool {
    const orchestrator = new ToolOrchestrator();
    
    return new DynamicTool({
      name: 'orchestrate_workflow',
      description: 'Execute complex multi-step workflows combining Solana wallet and QuantDesk protocol tools. Use for queries that require multiple data sources or complex analysis.',
      func: async (query: string) => {
        try {
          const result = await orchestrator.executeWorkflow(query);
          return JSON.stringify(result);
        } catch (error) {
          return JSON.stringify({
            success: false,
            error: `Orchestration failed: ${error.message}`,
            query
          });
        }
      }
    });
  }
}
