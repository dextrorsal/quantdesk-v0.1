#!/bin/bash

# Install Environment Variable Management Tools for QuantDesk
# Installs tools for detecting unused, duplicate, and problematic environment variables

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install dotenv-linter
install_dotenv_linter() {
    print_status "Installing dotenv-linter..."
    
    if command_exists dotenv-linter; then
        print_success "dotenv-linter is already installed"
        return 0
    fi
    
    # Try different installation methods
    if command_exists cargo; then
        print_status "Installing via cargo..."
        cargo install dotenv-linter
        print_success "dotenv-linter installed via cargo"
    elif command_exists brew; then
        print_status "Installing via brew..."
        brew install dotenv-linter
        print_success "dotenv-linter installed via brew"
    elif command_exists npm; then
        print_status "Installing via npm..."
        npm install -g dotenv-linter
        print_success "dotenv-linter installed via npm"
    else
        print_warning "dotenv-linter installation failed. Install manually:"
        print_warning "  cargo install dotenv-linter"
        print_warning "  or"
        print_warning "  brew install dotenv-linter"
        print_warning "  or"
        print_warning "  npm install -g dotenv-linter"
        return 1
    fi
}

# Function to install envkey
install_envkey() {
    print_status "Installing envkey..."
    
    if command_exists envkey; then
        print_success "envkey is already installed"
        return 0
    fi
    
    # Try different installation methods
    if command_exists brew; then
        print_status "Installing via brew..."
        brew install envkey/envkey/envkey
        print_success "envkey installed via brew"
    elif command_exists npm; then
        print_status "Installing via npm..."
        npm install -g envkey
        print_success "envkey installed via npm"
    else
        print_warning "envkey installation failed. Install manually:"
        print_warning "  brew install envkey/envkey/envkey"
        print_warning "  or"
        print_warning "  npm install -g envkey"
        return 1
    fi
}

# Function to install ESLint environment variable plugin
install_eslint_env_plugin() {
    print_status "Installing ESLint environment variable plugin..."
    
    if [ -f "package.json" ]; then
        if command_exists npm; then
            print_status "Installing via npm..."
            npm install --save-dev eslint-plugin-env
            print_success "ESLint environment variable plugin installed"
        else
            print_warning "npm not found, skipping ESLint plugin installation"
            return 1
        fi
    else
        print_warning "package.json not found, skipping ESLint plugin installation"
        return 1
    fi
}

# Function to create ESLint configuration
create_eslint_config() {
    print_status "Creating ESLint configuration for environment variables..."
    
    if [ -f "package.json" ]; then
        cat > .eslintrc.env.json << 'EOF'
{
  "extends": ["plugin:env/recommended"],
  "plugins": ["env"],
  "rules": {
    "env/no-unused-env-vars": "error",
    "env/no-duplicate-env-vars": "error"
  }
}
EOF
        print_success "ESLint configuration created: .eslintrc.env.json"
    else
        print_warning "package.json not found, skipping ESLint configuration"
    fi
}

# Function to install Python environment variable tools
install_python_env_tools() {
    print_status "Installing Python environment variable tools..."
    
    if command_exists pip; then
        print_status "Installing python-dotenv..."
        pip install python-dotenv
        print_success "python-dotenv installed"
        
        print_status "Installing environs..."
        pip install environs
        print_success "environs installed"
    else
        print_warning "pip not found, skipping Python tools installation"
        return 1
    fi
}

# Function to create environment variable validation script
create_validation_script() {
    print_status "Creating environment variable validation script..."
    
    cat > scripts/validate-env.js << 'EOF'
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
EOF
    
    chmod +x scripts/validate-env.js
    print_success "Environment variable validation script created: scripts/validate-env.js"
}

# Function to create package.json scripts
create_package_scripts() {
    print_status "Adding environment variable scripts to package.json..."
    
    if [ -f "package.json" ]; then
        # Create a backup
        cp package.json package.json.backup
        
        # Add scripts using node
        node -e "
            const fs = require('fs');
            const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
            
            if (!pkg.scripts) pkg.scripts = {};
            
            pkg.scripts['env:scan'] = './scripts/env-scanner.sh';
            pkg.scripts['env:validate'] = 'node scripts/validate-env.js';
            pkg.scripts['env:lint'] = 'dotenv-linter';
            pkg.scripts['env:check'] = 'npm run env:scan && npm run env:validate';
            
            fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2));
            console.log('✅ Environment variable scripts added to package.json');
        "
    else
        print_warning "package.json not found, skipping script addition"
    fi
}

# Function to create environment variable documentation
create_documentation() {
    print_status "Creating environment variable documentation..."
    
    mkdir -p docs
    
    cat > docs/ENVIRONMENT_VARIABLES.md << 'EOF'
# Environment Variables Guide for QuantDesk

## Overview

This document describes the environment variables used in the QuantDesk project and provides guidelines for managing them effectively.

## Environment Files

The project uses multiple environment files for different purposes:

- `.env` - Main environment file (not committed to git)
- `env.example` - Template file with example values
- `data-ingestion/.env` - Data ingestion service specific variables
- `data-ingestion/env.example` - Data ingestion template

## Naming Conventions

### Standard Format
- Use UPPERCASE letters
- Use underscores to separate words
- Prefix with service name when applicable

### Examples
```bash
# ✅ Good
QUANTDESK_API_URL=https://api.quantdesk.com
DATABASE_URL=postgresql://user:pass@localhost:5432/quantdesk
REDIS_URL=redis://localhost:6379

# ❌ Bad
quantdesk_api_url=https://api.quantdesk.com
DATABASE-URL=postgresql://user:pass@localhost:5432/quantdesk
```

## Required Variables

### Core Application
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `JWT_SECRET` - Secret key for JWT tokens
- `NODE_ENV` - Environment (development, production)

### Solana Integration
- `SOLANA_NETWORK` - Network (devnet, mainnet-beta)
- `RPC_URL` - Solana RPC endpoint
- `PYTH_NETWORK_URL` - Pyth network endpoint

### External Services
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key

## Security Best Practices

1. **Never commit sensitive values** to version control
2. **Use strong, unique secrets** for JWT_SECRET
3. **Rotate secrets regularly** in production
4. **Use environment-specific values** for different deployments
5. **Validate all environment variables** at startup

## Validation Tools

The project includes several tools for managing environment variables:

### 1. Environment Scanner
```bash
npm run env:scan
```
Scans for duplicates, similar names, and format issues.

### 2. Environment Validator
```bash
npm run env:validate
```
Validates environment variables and checks for common issues.

### 3. dotenv-linter
```bash
npm run env:lint
```
Lints .env files for formatting and best practices.

### 4. Combined Check
```bash
npm run env:check
```
Runs all environment variable checks.

## Common Issues

### Duplicate Variables
Having the same variable in multiple files can cause confusion:
```bash
# ❌ Problem
# .env
QUANTDESK_URL=https://app.quantdesk.com

# data-ingestion/.env
QUANTDESK_URL=https://api.quantdesk.com
```

### Similar Names
Variables with similar names can cause confusion:
```bash
# ❌ Problem
QUANTDESK_URL=https://app.quantdesk.com
QUANTDESK_API_URL=https://api.quantdesk.com
```

### Inconsistent Naming
Different naming conventions can cause issues:
```bash
# ❌ Problem
DATABASE_URL=postgresql://...
DB_CONNECTION_STRING=postgresql://...
```

## Troubleshooting

### Variable Not Found
If an environment variable is not found:
1. Check if it's defined in the correct .env file
2. Verify the variable name matches exactly
3. Ensure the .env file is in the correct location
4. Restart the application after making changes

### Duplicate Variables
If you have duplicate variables:
1. Use the environment scanner to identify them
2. Choose one canonical location for each variable
3. Remove duplicates from other files
4. Update code to reference the canonical variable

### Similar Names
If you have similar variable names:
1. Use the environment scanner to identify them
2. Choose a consistent naming convention
3. Rename variables to follow the convention
4. Update code to use the new names

## Best Practices

1. **Use consistent naming** across all environment files
2. **Document all variables** in this file
3. **Validate variables** at application startup
4. **Use type validation** for numeric and boolean values
5. **Provide default values** where appropriate
6. **Use environment-specific files** for different deployments

## Tools and Scripts

- `scripts/env-scanner.sh` - Comprehensive environment variable scanner
- `scripts/validate-env.js` - Node.js environment variable validator
- `scripts/install-env-tools.sh` - Install environment variable tools

## References

- [dotenv-linter Documentation](https://github.com/dotenv-linter/dotenv-linter)
- [Node.js Environment Variables](https://nodejs.org/en/learn/command-line/how-to-read-environment-variables-from-nodejs)
- [Environment Variables Best Practices](https://12factor.net/config)
EOF
    
    print_success "Environment variable documentation created: docs/ENVIRONMENT_VARIABLES.md"
}

# Main execution function
main() {
    echo "🛠️  Installing Environment Variable Management Tools"
    echo "=================================================="
    echo ""
    
    # Install tools
    install_dotenv_linter
    echo ""
    
    install_envkey
    echo ""
    
    install_eslint_env_plugin
    echo ""
    
    install_python_env_tools
    echo ""
    
    # Create configurations and scripts
    create_eslint_config
    echo ""
    
    create_validation_script
    echo ""
    
    create_package_scripts
    echo ""
    
    create_documentation
    echo ""
    
    # Summary
    echo "🎉 Installation Complete!"
    echo "========================"
    echo ""
    echo "✅ Installed tools:"
    echo "  - dotenv-linter (environment file linter)"
    echo "  - envkey (environment variable management)"
    echo "  - ESLint environment variable plugin"
    echo "  - Python environment variable tools"
    echo ""
    echo "✅ Created scripts:"
    echo "  - scripts/env-scanner.sh (comprehensive scanner)"
    echo "  - scripts/validate-env.js (Node.js validator)"
    echo "  - scripts/install-env-tools.sh (this script)"
    echo ""
    echo "✅ Created documentation:"
    echo "  - docs/ENVIRONMENT_VARIABLES.md (comprehensive guide)"
    echo "  - .eslintrc.env.json (ESLint configuration)"
    echo ""
    echo "✅ Added npm scripts:"
    echo "  - npm run env:scan (run environment scanner)"
    echo "  - npm run env:validate (run validator)"
    echo "  - npm run env:lint (run dotenv-linter)"
    echo "  - npm run env:check (run all checks)"
    echo ""
    echo "🚀 Next Steps:"
    echo "1. Run: npm run env:check"
    echo "2. Review: docs/ENVIRONMENT_VARIABLES.md"
    echo "3. Fix any issues found"
    echo "4. Set up CI/CD to run environment checks"
    echo ""
    print_success "Environment variable tools installation complete! 🎉"
}

# Run main function
main "$@"

