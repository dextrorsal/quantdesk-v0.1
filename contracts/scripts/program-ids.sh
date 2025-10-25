# QuantDesk Program IDs Configuration
# This file contains all Program IDs for the QuantDesk specialized programs

# Program IDs for QuantDesk Specialized Programs
export QUANTDESK_CORE_PROGRAM_ID="CNfhSBoMkRbDEQ2EC3RkfJ2S39Up6WJLr4U31ZL49LrU"
export QUANTDESK_TRADING_PROGRAM_ID="AvxWXu25yWhDXJBy1V5GYcn2eVws4F2QWK5G3zV4t8sZ"
export QUANTDESK_COLLATERAL_PROGRAM_ID="GPrakftrbBUUiir2MpQZv6G7UB5Jq8yNGHV5YTVYPQ5i"
export QUANTDESK_SECURITY_PROGRAM_ID="84b7Khx4uj7mHDvn2V63kNSwkcgpagrBgZSdTJ7kTxWW"
export QUANTDESK_ORACLE_PROGRAM_ID="8gjwta4tMQshM7HbnEMsdFUMqjRe7XgVnxJVbcmf3cAC"

# Original Monolithic Program ID (for comparison)
export QUANTDESK_ORIGINAL_PROGRAM_ID="C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw"

# Program Names
export QUANTDESK_CORE_PROGRAM_NAME="quantdesk_core"
export QUANTDESK_TRADING_PROGRAM_NAME="quantdesk_trading"
export QUANTDESK_COLLATERAL_PROGRAM_NAME="quantdesk_collateral"
export QUANTDESK_SECURITY_PROGRAM_NAME="quantdesk_security"
export QUANTDESK_ORACLE_PROGRAM_NAME="quantdesk_oracle"
export QUANTDESK_ORIGINAL_PROGRAM_NAME="quantdesk_perp_dex"

# Keypair Files
export QUANTDESK_CORE_KEYPAIR="quantdesk-core-keypair.json"
export QUANTDESK_TRADING_KEYPAIR="quantdesk-trading-keypair.json"
export QUANTDESK_COLLATERAL_KEYPAIR="quantdesk-collateral-keypair.json"
export QUANTDESK_SECURITY_KEYPAIR="quantdesk-security-keypair.json"
export QUANTDESK_ORACLE_KEYPAIR="quantdesk-oracle-keypair.json"

# Program Descriptions
export QUANTDESK_CORE_DESCRIPTION="Core Program - User accounts, basic market operations, essential collateral"
export QUANTDESK_TRADING_DESCRIPTION="Trading Program - Position management, order management, advanced orders"
export QUANTDESK_COLLATERAL_DESCRIPTION="Collateral Program - Deposits, withdrawals, token operations, cross-collateral"
export QUANTDESK_SECURITY_DESCRIPTION="Security Program - Circuit breakers, keeper management, emergency controls"
export QUANTDESK_ORACLE_DESCRIPTION="Oracle Program - Oracle feeds, insurance fund, emergency controls"

# Program Instructions Count
export QUANTDESK_CORE_INSTRUCTIONS=15
export QUANTDESK_TRADING_INSTRUCTIONS=15
export QUANTDESK_COLLATERAL_INSTRUCTIONS=8
export QUANTDESK_SECURITY_INSTRUCTIONS=12
export QUANTDESK_ORACLE_INSTRUCTIONS=9
export QUANTDESK_ORIGINAL_INSTRUCTIONS=59

# Program Stack Usage (estimated)
export QUANTDESK_CORE_STACK_USAGE="<200KB"
export QUANTDESK_TRADING_STACK_USAGE="<300KB"
export QUANTDESK_COLLATERAL_STACK_USAGE="<150KB"
export QUANTDESK_SECURITY_STACK_USAGE="<250KB"
export QUANTDESK_ORACLE_STACK_USAGE="<200KB"
export QUANTDESK_ORIGINAL_STACK_USAGE=">4KB (Stack Overflow)"

# Deployment Status
export QUANTDESK_CORE_DEPLOYED=false
export QUANTDESK_TRADING_DEPLOYED=false
export QUANTDESK_COLLATERAL_DEPLOYED=false
export QUANTDESK_SECURITY_DEPLOYED=false
export QUANTDESK_ORACLE_DEPLOYED=false
export QUANTDESK_ORIGINAL_DEPLOYED=true

# Usage Instructions
echo "QuantDesk Program IDs Configuration Loaded"
echo "=========================================="
echo "Core Program: $QUANTDESK_CORE_PROGRAM_ID"
echo "Trading Program: $QUANTDESK_TRADING_PROGRAM_ID"
echo "Collateral Program: $QUANTDESK_COLLATERAL_PROGRAM_ID"
echo "Security Program: $QUANTDESK_SECURITY_PROGRAM_ID"
echo "Oracle Program: $QUANTDESK_ORACLE_PROGRAM_ID"
echo "Original Program: $QUANTDESK_ORIGINAL_PROGRAM_ID"
echo ""
echo "To use these IDs in your code:"
echo "source scripts/program-ids.sh"
echo "echo \$QUANTDESK_CORE_PROGRAM_ID"
