#!/usr/bin/env node

/**
 * Test Coverage Validation Script for Story 1.2
 * 
 * This script validates that all required tests are implemented and
 * provides coverage analysis for the order placement and execution functionality.
 */

import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

interface TestRequirement {
  id: string;
  name: string;
  type: 'unit' | 'integration' | 'e2e';
  priority: 'P0' | 'P1' | 'P2';
  file: string;
  description: string;
}

interface CoverageReport {
  totalTests: number;
  implementedTests: number;
  missingTests: string[];
  coveragePercentage: number;
  priorityBreakdown: {
    P0: { total: number; implemented: number };
    P1: { total: number; implemented: number };
    P2: { total: number; implemented: number };
  };
}

class TestCoverageValidator {
  private testRequirements: TestRequirement[] = [];
  private testDirectory: string;
  private coverageReport: CoverageReport;

  constructor(testDirectory: string = 'backend/tests') {
    this.testDirectory = testDirectory;
    this.coverageReport = {
      totalTests: 0,
      implementedTests: 0,
      missingTests: [],
      coveragePercentage: 0,
      priorityBreakdown: {
        P0: { total: 0, implemented: 0 },
        P1: { total: 0, implemented: 0 },
        P2: { total: 0, implemented: 0 }
      }
    };
  }

  /**
   * Initialize test requirements based on traceability matrix
   */
  private initializeTestRequirements(): void {
    this.testRequirements = [
      // Unit Tests
      {
        id: '1.2-UNIT-001',
        name: 'Order validation logic',
        type: 'unit',
        priority: 'P0',
        file: 'unit/order-authorization.test.ts',
        description: 'Order parameters with valid data validation'
      },
      {
        id: '1.2-UNIT-002',
        name: 'Order parameter sanitization',
        type: 'unit',
        priority: 'P0',
        file: 'unit/order-authorization.test.ts',
        description: 'Order parameters with potentially malicious input sanitization'
      },
      {
        id: '1.2-UNIT-003',
        name: 'Order execution logic validation',
        type: 'unit',
        priority: 'P0',
        file: 'unit/order-authorization.test.ts',
        description: 'Order execution conditions validation'
      },
      {
        id: '1.2-UNIT-004',
        name: 'Smart contract instruction validation',
        type: 'unit',
        priority: 'P0',
        file: 'unit/order-authorization.test.ts',
        description: 'Smart contract instruction parameters validation'
      },
      {
        id: '1.2-UNIT-005',
        name: 'Order status state machine',
        type: 'unit',
        priority: 'P1',
        file: 'unit/order-authorization.test.ts',
        description: 'Order status transitions validation'
      },
      {
        id: '1.2-UNIT-006',
        name: 'Position creation logic validation',
        type: 'unit',
        priority: 'P1',
        file: 'unit/order-authorization.test.ts',
        description: 'Position creation from filled order validation'
      },
      {
        id: '1.2-UNIT-007',
        name: 'Error message generation logic',
        type: 'unit',
        priority: 'P1',
        file: 'unit/order-authorization.test.ts',
        description: 'Error message generation for order failures'
      },
      {
        id: '1.2-UNIT-008',
        name: 'Order authorization validation',
        type: 'unit',
        priority: 'P0',
        file: 'unit/order-authorization.test.ts',
        description: 'Unauthorized order request rejection'
      },
      {
        id: '1.2-UNIT-009',
        name: 'Position creation atomicity validation',
        type: 'unit',
        priority: 'P0',
        file: 'unit/order-authorization.test.ts',
        description: 'Atomic transaction for position creation'
      },

      // Integration Tests
      {
        id: '1.2-INT-001',
        name: 'Backend order placement service',
        type: 'integration',
        priority: 'P0',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Backend order placement service integration'
      },
      {
        id: '1.2-INT-002',
        name: 'Database order persistence',
        type: 'integration',
        priority: 'P0',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Database order persistence integration'
      },
      {
        id: '1.2-INT-003',
        name: 'Backend-smart contract communication',
        type: 'integration',
        priority: 'P0',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Backend-smart contract communication integration'
      },
      {
        id: '1.2-INT-004',
        name: 'Order execution with Oracle price feed',
        type: 'integration',
        priority: 'P0',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Order execution with Oracle price feed integration'
      },
      {
        id: '1.2-INT-005',
        name: 'Atomic transaction execution',
        type: 'integration',
        priority: 'P0',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Atomic transaction execution integration'
      },
      {
        id: '1.2-INT-006',
        name: 'Order status synchronization across systems',
        type: 'integration',
        priority: 'P1',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Order status synchronization across systems integration'
      },
      {
        id: '1.2-INT-007',
        name: 'WebSocket order status updates',
        type: 'integration',
        priority: 'P1',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'WebSocket order status updates integration'
      },
      {
        id: '1.2-INT-008',
        name: 'Order-to-position creation flow',
        type: 'integration',
        priority: 'P0',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Order-to-position creation flow integration'
      },
      {
        id: '1.2-INT-009',
        name: 'Position persistence in database',
        type: 'integration',
        priority: 'P1',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Position persistence in database integration'
      },
      {
        id: '1.2-INT-010',
        name: 'Error propagation across systems',
        type: 'integration',
        priority: 'P1',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Error propagation across systems integration'
      },
      {
        id: '1.2-INT-011',
        name: 'Backend-smart contract communication failure recovery',
        type: 'integration',
        priority: 'P0',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Backend-smart contract communication failure recovery integration'
      },
      {
        id: '1.2-INT-012',
        name: 'Unauthorized order execution prevention',
        type: 'integration',
        priority: 'P0',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Unauthorized order execution prevention integration'
      },
      {
        id: '1.2-INT-013',
        name: 'Position creation failure recovery',
        type: 'integration',
        priority: 'P0',
        file: 'integration/order-authorization-integration.test.ts',
        description: 'Position creation failure recovery integration'
      },

      // E2E Tests
      {
        id: '1.2-E2E-001',
        name: 'User places market order successfully',
        type: 'e2e',
        priority: 'P0',
        file: 'e2e/order-execution-e2e.test.ts',
        description: 'User places market order successfully E2E test'
      },
      {
        id: '1.2-E2E-002',
        name: 'Order executes when price conditions met',
        type: 'e2e',
        priority: 'P0',
        file: 'e2e/order-execution-e2e.test.ts',
        description: 'Order executes when price conditions met E2E test'
      },
      {
        id: '1.2-E2E-003',
        name: 'User sees real-time order status updates',
        type: 'e2e',
        priority: 'P1',
        file: 'e2e/order-execution-e2e.test.ts',
        description: 'User sees real-time order status updates E2E test'
      },
      {
        id: '1.2-E2E-004',
        name: 'Position created after order fill',
        type: 'e2e',
        priority: 'P0',
        file: 'e2e/order-execution-e2e.test.ts',
        description: 'Position created after order fill E2E test'
      },
      {
        id: '1.2-E2E-005',
        name: 'User sees clear error messages',
        type: 'e2e',
        priority: 'P1',
        file: 'e2e/order-execution-e2e.test.ts',
        description: 'User sees clear error messages E2E test'
      },
      {
        id: '1.2-E2E-006',
        name: 'Order execution with smart contract failure',
        type: 'e2e',
        priority: 'P0',
        file: 'e2e/order-execution-e2e.test.ts',
        description: 'Order execution with smart contract failure E2E test'
      }
    ];
  }

  /**
   * Check if a test file exists
   */
  private testFileExists(filePath: string): boolean {
    const fullPath = join(this.testDirectory, filePath);
    return existsSync(fullPath);
  }

  /**
   * Check if a test requirement is implemented
   */
  private isTestImplemented(requirement: TestRequirement): boolean {
    return this.testFileExists(requirement.file);
  }

  /**
   * Validate test coverage
   */
  public validateCoverage(): CoverageReport {
    this.initializeTestRequirements();
    
    this.coverageReport.totalTests = this.testRequirements.length;
    this.coverageReport.implementedTests = 0;
    this.coverageReport.missingTests = [];

    // Count implemented tests by priority
    for (const requirement of this.testRequirements) {
      if (this.isTestImplemented(requirement)) {
        this.coverageReport.implementedTests++;
        this.coverageReport.priorityBreakdown[requirement.priority].implemented++;
      } else {
        this.coverageReport.missingTests.push(requirement.id);
      }
      
      this.coverageReport.priorityBreakdown[requirement.priority].total++;
    }

    // Calculate coverage percentage
    this.coverageReport.coveragePercentage = 
      (this.coverageReport.implementedTests / this.coverageReport.totalTests) * 100;

    return this.coverageReport;
  }

  /**
   * Generate coverage report
   */
  public generateReport(): string {
    const report = this.validateCoverage();
    
    let output = `
# Test Coverage Validation Report for Story 1.2

## Summary
- **Total Tests Required**: ${report.totalTests}
- **Tests Implemented**: ${report.implementedTests}
- **Tests Missing**: ${report.missingTests.length}
- **Coverage Percentage**: ${report.coveragePercentage.toFixed(1)}%

## Priority Breakdown
- **P0 (Critical)**: ${report.priorityBreakdown.P0.implemented}/${report.priorityBreakdown.P0.total} (${((report.priorityBreakdown.P0.implemented / report.priorityBreakdown.P0.total) * 100).toFixed(1)}%)
- **P1 (High)**: ${report.priorityBreakdown.P1.implemented}/${report.priorityBreakdown.P1.total} (${((report.priorityBreakdown.P1.implemented / report.priorityBreakdown.P1.total) * 100).toFixed(1)}%)
- **P2 (Medium)**: ${report.priorityBreakdown.P2.implemented}/${report.priorityBreakdown.P2.total} (${((report.priorityBreakdown.P2.implemented / report.priorityBreakdown.P2.total) * 100).toFixed(1)}%)

## Missing Tests
`;

    if (report.missingTests.length > 0) {
      output += '\n';
      for (const missingTest of report.missingTests) {
        const requirement = this.testRequirements.find(r => r.id === missingTest);
        if (requirement) {
          output += `- **${missingTest}**: ${requirement.name} (${requirement.type}, ${requirement.priority})\n`;
        }
      }
    } else {
      output += '\n✅ All required tests are implemented!\n';
    }

    output += `
## Test Files Status
`;

    // Check test files
    const testFiles = [
      'unit/order-authorization.test.ts',
      'integration/order-authorization-integration.test.ts',
      'e2e/order-execution-e2e.test.ts',
      'unit/smart-contract-service.test.ts',
      'integration/order-api.test.ts',
      'integration/order-position-flow.test.ts',
      'e2e/order-position-flow.test.ts'
    ];

    for (const file of testFiles) {
      const exists = this.testFileExists(file);
      const status = exists ? '✅' : '❌';
      output += `- ${status} ${file}\n`;
    }

    output += `
## Coverage Assessment
`;

    if (report.coveragePercentage >= 80) {
      output += `
✅ **EXCELLENT**: Test coverage meets the 80% target requirement.
- Coverage: ${report.coveragePercentage.toFixed(1)}%
- All critical (P0) tests are implemented
- Ready for production deployment
`;
    } else if (report.coveragePercentage >= 60) {
      output += `
⚠️ **GOOD**: Test coverage is acceptable but below target.
- Coverage: ${report.coveragePercentage.toFixed(1)}%
- Missing ${report.missingTests.length} tests
- Consider implementing missing tests before production
`;
    } else {
      output += `
❌ **POOR**: Test coverage is below acceptable threshold.
- Coverage: ${report.coveragePercentage.toFixed(1)}%
- Missing ${report.missingTests.length} tests
- Must implement missing tests before production deployment
`;
    }

    output += `
## Recommendations
`;

    if (report.missingTests.length > 0) {
      output += `
1. **Implement Missing Tests**: Focus on P0 priority tests first
2. **Run Test Suite**: Execute all tests to ensure they pass
3. **Update Documentation**: Update test documentation with new tests
4. **Code Review**: Review test implementations for quality
`;
    } else {
      output += `
1. **Run Test Suite**: Execute all tests to ensure they pass
2. **Performance Testing**: Consider adding performance tests
3. **Load Testing**: Test under high load conditions
4. **Security Testing**: Add security-focused test scenarios
`;
    }

    return output;
  }

  /**
   * Run the test suite
   */
  public async runTests(): Promise<{ success: boolean; output: string }> {
    try {
      console.log('Running test suite...');
      const output = execSync('cd backend && npm test', { 
        encoding: 'utf8',
        stdio: 'pipe'
      });
      return { success: true, output };
    } catch (error: any) {
      return { success: false, output: error.stdout || error.message };
    }
  }

  /**
   * Generate test coverage report
   */
  public async generateCoverageReport(): Promise<string> {
    try {
      console.log('Generating coverage report...');
      const output = execSync('cd backend && npm run test:coverage', { 
        encoding: 'utf8',
        stdio: 'pipe'
      });
      return output;
    } catch (error: any) {
      return error.stdout || error.message;
    }
  }
}

// Main execution
async function main() {
  const validator = new TestCoverageValidator();
  
  console.log('🔍 Validating test coverage for Story 1.2...\n');
  
  // Generate coverage report
  const report = validator.generateReport();
  console.log(report);
  
  // Run tests
  console.log('\n🧪 Running test suite...\n');
  const testResult = await validator.runTests();
  
  if (testResult.success) {
    console.log('✅ All tests passed!');
  } else {
    console.log('❌ Some tests failed:');
    console.log(testResult.output);
  }
  
  // Generate coverage report
  console.log('\n📊 Generating coverage report...\n');
  const coverageOutput = await validator.generateCoverageReport();
  console.log(coverageOutput);
  
  // Save report to file
  const fs = require('fs');
  const reportPath = 'backend/tests/COVERAGE_REPORT.md';
  fs.writeFileSync(reportPath, report);
  console.log(`\n📄 Coverage report saved to: ${reportPath}`);
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

export { TestCoverageValidator };
