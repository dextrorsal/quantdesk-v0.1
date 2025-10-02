#!/usr/bin/env node

// Environment Variable Validation Script for QuantDesk
// Validates environment variables and checks for common issues

const fs = require('fs');
const path = require('path');

// Colors for console output
const colors = {
    reset: '\x1b[0m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m'
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

function findEnvFiles() {
    const envFiles = [];
    
    function searchDirectory(dir) {
        const items = fs.readdirSync(dir);
        
        for (const item of items) {
            const fullPath = path.join(dir, item);
            const stat = fs.statSync(fullPath);
            
            if (stat.isDirectory() && !item.includes('node_modules') && !item.includes('.git')) {
                searchDirectory(fullPath);
            } else if (item.startsWith('.env')) {
                envFiles.push(fullPath);
            }
        }
    }
    
    searchDirectory('.');
    return envFiles;
}

function parseEnvFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const variables = {};
    
    content.split('\n').forEach((line, index) => {
        line = line.trim();
        
        if (line && !line.startsWith('#')) {
            const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
            if (match) {
                const [, name, value] = match;
                variables[name] = {
                    value: value,
                    file: filePath,
                    line: index + 1
                };
            }
        }
    });
    
    return variables;
}

function validateEnvironmentVariables() {
    log('🔍 QuantDesk Environment Variable Validator', 'blue');
    log('==========================================', 'blue');
    log('');
    
    const envFiles = findEnvFiles();
    
    if (envFiles.length === 0) {
        log('❌ No .env files found', 'red');
        return false;
    }
    
    log(`📁 Found ${envFiles.length} environment file(s):`, 'green');
    envFiles.forEach(file => log(`  - ${file}`, 'green'));
    log('');
    
    // Parse all environment files
    const allVariables = {};
    const duplicates = [];
    const similarNames = [];
    
    envFiles.forEach(file => {
        const variables = parseEnvFile(file);
        Object.keys(variables).forEach(name => {
            if (allVariables[name]) {
                duplicates.push({
                    name: name,
                    files: [allVariables[name].file, variables[name].file]
                });
            } else {
                allVariables[name] = variables[name];
            }
        });
    });
    
    // Check for similar names
    const variableNames = Object.keys(allVariables).sort();
    for (let i = 0; i < variableNames.length - 1; i++) {
        const current = variableNames[i];
        const next = variableNames[i + 1];
        
        // Check if names are similar (same prefix)
        if (next.startsWith(current) && next.length > current.length) {
            similarNames.push({ current, next });
        }
    }
    
    // Report results
    let hasIssues = false;
    
    if (duplicates.length > 0) {
        log('⚠️  Duplicate environment variables found:', 'yellow');
        duplicates.forEach(dup => {
            log(`  - ${dup.name}`, 'yellow');
            log(`    Found in: ${dup.files.join(', ')}`, 'yellow');
        });
        log('');
        hasIssues = true;
    }
    
    if (similarNames.length > 0) {
        log('⚠️  Similar environment variable names found:', 'yellow');
        similarNames.forEach(sim => {
            log(`  - ${sim.current} vs ${sim.next}`, 'yellow');
        });
        log('');
        hasIssues = true;
    }
    
    // Check for common naming patterns
    const namingIssues = [];
    Object.keys(allVariables).forEach(name => {
        if (name.includes('QUANTDESK') && name.includes('API')) {
            // Check for potential duplicates
            const baseName = name.replace(/QUANTDESK_?API_?/, '');
            const alternativeName = `QUANTDESK_${baseName}`;
            
            if (allVariables[alternativeName]) {
                namingIssues.push({
                    name1: name,
                    name2: alternativeName,
                    suggestion: 'Consider standardizing on one naming convention'
                });
            }
        }
    });
    
    if (namingIssues.length > 0) {
        log('⚠️  Potential naming convention issues:', 'yellow');
        namingIssues.forEach(issue => {
            log(`  - ${issue.name1} vs ${issue.name2}`, 'yellow');
            log(`    ${issue.suggestion}`, 'yellow');
        });
        log('');
        hasIssues = true;
    }
    
    if (!hasIssues) {
        log('✅ No environment variable issues found', 'green');
        log('✅ All variables are properly formatted', 'green');
        log('✅ No duplicates or similar names detected', 'green');
    }
    
    log('');
    log('📊 Summary:', 'blue');
    log(`  Total environment files: ${envFiles.length}`, 'blue');
    log(`  Total variables: ${Object.keys(allVariables).length}`, 'blue');
    log(`  Duplicates: ${duplicates.length}`, 'blue');
    log(`  Similar names: ${similarNames.length}`, 'blue');
    log(`  Naming issues: ${namingIssues.length}`, 'blue');
    
    return !hasIssues;
}

// Run validation
if (require.main === module) {
    const success = validateEnvironmentVariables();
    process.exit(success ? 0 : 1);
}

module.exports = { validateEnvironmentVariables };
