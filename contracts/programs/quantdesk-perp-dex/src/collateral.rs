use anchor_lang::prelude::*;

/// Collateral Management Module
/// Handles collateral accounts and cross-collateralization

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum CollateralType {
    SOL,
    USDC,
    BTC,
    ETH,
    USDT,
    AVAX,
    MATIC,
    ARB,
    OP,
    DOGE,
    ADA,
    DOT,
    LINK,
}

impl std::fmt::Display for CollateralType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollateralType::SOL => write!(f, "SOL"),
            CollateralType::USDC => write!(f, "USDC"),
            CollateralType::BTC => write!(f, "BTC"),
            CollateralType::ETH => write!(f, "ETH"),
            CollateralType::USDT => write!(f, "USDT"),
            CollateralType::AVAX => write!(f, "AVAX"),
            CollateralType::MATIC => write!(f, "MATIC"),
            CollateralType::ARB => write!(f, "ARB"),
            CollateralType::OP => write!(f, "OP"),
            CollateralType::DOGE => write!(f, "DOGE"),
            CollateralType::ADA => write!(f, "ADA"),
            CollateralType::DOT => write!(f, "DOT"),
            CollateralType::LINK => write!(f, "LINK"),
        }
    }
}

// CollateralAccount struct moved to state/collateral.rs per best practices
// Helper functions remain here as utility functions
