/**
 * Program IDL in camelCase format in order to be used in JS/TS.
 *
 * Note that this is only a type helper and is not the actual IDL. The original
 * IDL can be found at `target/idl/quantdesk_perp_dex.json`.
 */
export type QuantdeskPerpDex = {
  "address": "C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw",
  "metadata": {
    "name": "quantdeskPerpDex",
    "version": "0.1.0",
    "spec": "0.1.0",
    "description": "QuantDesk Perpetual DEX Smart Contracts"
  },
  "instructions": [
    {
      "name": "addCollateral",
      "discriminator": [
        127,
        82,
        121,
        42,
        161,
        176,
        249,
        206
      ],
      "accounts": [
        {
          "name": "collateralAccount",
          "writable": true
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "userTokenAccount",
          "writable": true
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "addCrossCollateral",
      "docs": [
        "Add collateral to cross-collateral account"
      ],
      "discriminator": [
        250,
        134,
        188,
        216,
        236,
        244,
        143,
        195
      ],
      "accounts": [
        {
          "name": "crossCollateralAccount",
          "writable": true
        },
        {
          "name": "collateralConfig",
          "writable": true
        },
        {
          "name": "oraclePriceFeed"
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "assetType",
          "type": {
            "defined": {
              "name": "collateralType"
            }
          }
        },
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "addOracleFeed",
      "docs": [
        "Add new oracle feed"
      ],
      "discriminator": [
        14,
        65,
        33,
        73,
        114,
        220,
        190,
        77
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "oracleManager",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "feedType",
          "type": {
            "defined": {
              "name": "oracleFeedType"
            }
          }
        },
        {
          "name": "weight",
          "type": "u8"
        }
      ]
    },
    {
      "name": "cancelOrder",
      "discriminator": [
        95,
        129,
        237,
        240,
        8,
        49,
        223,
        132
      ],
      "accounts": [
        {
          "name": "order",
          "writable": true
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        }
      ],
      "args": []
    },
    {
      "name": "checkUserPermissions",
      "discriminator": [
        121,
        136,
        44,
        44,
        82,
        232,
        26,
        99
      ],
      "accounts": [
        {
          "name": "userAccount",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "action",
          "type": {
            "defined": {
              "name": "userAction"
            }
          }
        }
      ]
    },
    {
      "name": "closePosition",
      "discriminator": [
        123,
        134,
        81,
        0,
        49,
        68,
        98,
        98
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "position",
          "writable": true
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "userCollateral",
          "writable": true
        }
      ],
      "args": []
    },
    {
      "name": "closeUserAccount",
      "discriminator": [
        236,
        181,
        3,
        71,
        194,
        18,
        151,
        191
      ],
      "accounts": [
        {
          "name": "userAccount",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": []
    },
    {
      "name": "collectFees",
      "docs": [
        "Collect accumulated fees"
      ],
      "discriminator": [
        164,
        152,
        207,
        99,
        30,
        186,
        19,
        182
      ],
      "accounts": [
        {
          "name": "feeCollector",
          "writable": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "feeVault",
          "writable": true
        },
        {
          "name": "collector",
          "writable": true,
          "signer": true
        }
      ],
      "args": []
    },
    {
      "name": "createUserAccount",
      "discriminator": [
        146,
        68,
        100,
        69,
        63,
        46,
        182,
        199
      ],
      "accounts": [
        {
          "name": "userAccount",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  117,
                  115,
                  101,
                  114,
                  95,
                  97,
                  99,
                  99,
                  111,
                  117,
                  110,
                  116
                ]
              },
              {
                "kind": "account",
                "path": "authority"
              },
              {
                "kind": "arg",
                "path": "accountIndex"
              }
            ]
          }
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "accountIndex",
          "type": "u16"
        }
      ]
    },
    {
      "name": "createUserTokenAccount",
      "docs": [
        "Create user token account if needed"
      ],
      "discriminator": [
        29,
        72,
        16,
        95,
        33,
        134,
        94,
        248
      ],
      "accounts": [
        {
          "name": "userTokenAccount",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "const",
                "value": [
                  6,
                  221,
                  246,
                  225,
                  215,
                  101,
                  161,
                  147,
                  217,
                  203,
                  225,
                  70,
                  206,
                  235,
                  121,
                  172,
                  28,
                  180,
                  133,
                  237,
                  95,
                  91,
                  55,
                  145,
                  58,
                  140,
                  245,
                  133,
                  126,
                  255,
                  0,
                  169
                ]
              },
              {
                "kind": "account",
                "path": "mint"
              }
            ],
            "program": {
              "kind": "const",
              "value": [
                140,
                151,
                37,
                143,
                78,
                36,
                137,
                241,
                187,
                61,
                16,
                41,
                20,
                142,
                13,
                131,
                11,
                90,
                19,
                153,
                218,
                255,
                16,
                132,
                4,
                142,
                123,
                216,
                219,
                233,
                248,
                89
              ]
            }
          }
        },
        {
          "name": "mint"
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "tokenProgram",
          "address": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        },
        {
          "name": "associatedTokenProgram",
          "address": "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": []
    },
    {
      "name": "depositInsuranceFund",
      "docs": [
        "Deposit funds to insurance pool"
      ],
      "discriminator": [
        237,
        0,
        91,
        28,
        87,
        48,
        239,
        248
      ],
      "accounts": [
        {
          "name": "insuranceFund",
          "writable": true
        },
        {
          "name": "depositor",
          "writable": true,
          "signer": true
        },
        {
          "name": "depositorTokenAccount",
          "writable": true
        },
        {
          "name": "fundVault",
          "writable": true
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "depositNativeSol",
      "docs": [
        "Deposit native SOL to user account"
      ],
      "discriminator": [
        16,
        147,
        179,
        138,
        225,
        77,
        137,
        35
      ],
      "accounts": [
        {
          "name": "userAccount",
          "writable": true
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "protocolVault",
          "docs": [
            "Protocol SOL vault PDA - this will hold the deposited SOL"
          ],
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  112,
                  114,
                  111,
                  116,
                  111,
                  99,
                  111,
                  108,
                  95,
                  115,
                  111,
                  108,
                  95,
                  118,
                  97,
                  117,
                  108,
                  116
                ]
              }
            ]
          }
        },
        {
          "name": "collateralAccount",
          "docs": [
            "SOL collateral account for the user - will be initialized if needed"
          ],
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  99,
                  111,
                  108,
                  108,
                  97,
                  116,
                  101,
                  114,
                  97,
                  108
                ]
              },
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "const",
                "value": [
                  83,
                  79,
                  76
                ]
              }
            ]
          }
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        },
        {
          "name": "rent",
          "address": "SysvarRent111111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "depositTokens",
      "docs": [
        "Deposit tokens into protocol vault"
      ],
      "discriminator": [
        176,
        83,
        229,
        18,
        191,
        143,
        176,
        150
      ],
      "accounts": [
        {
          "name": "vault",
          "writable": true
        },
        {
          "name": "userTokenAccount",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "const",
                "value": [
                  6,
                  221,
                  246,
                  225,
                  215,
                  101,
                  161,
                  147,
                  217,
                  203,
                  225,
                  70,
                  206,
                  235,
                  121,
                  172,
                  28,
                  180,
                  133,
                  237,
                  95,
                  91,
                  55,
                  145,
                  58,
                  140,
                  245,
                  133,
                  126,
                  255,
                  0,
                  169
                ]
              },
              {
                "kind": "account",
                "path": "vault.mint",
                "account": "tokenVault"
              }
            ],
            "program": {
              "kind": "const",
              "value": [
                140,
                151,
                37,
                143,
                78,
                36,
                137,
                241,
                187,
                61,
                16,
                41,
                20,
                142,
                13,
                131,
                11,
                90,
                19,
                153,
                218,
                255,
                16,
                132,
                4,
                142,
                123,
                216,
                219,
                233,
                248,
                89
              ]
            }
          }
        },
        {
          "name": "vaultTokenAccount",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "account",
                "path": "vault"
              },
              {
                "kind": "const",
                "value": [
                  6,
                  221,
                  246,
                  225,
                  215,
                  101,
                  161,
                  147,
                  217,
                  203,
                  225,
                  70,
                  206,
                  235,
                  121,
                  172,
                  28,
                  180,
                  133,
                  237,
                  95,
                  91,
                  55,
                  145,
                  58,
                  140,
                  245,
                  133,
                  126,
                  255,
                  0,
                  169
                ]
              },
              {
                "kind": "account",
                "path": "vault.mint",
                "account": "tokenVault"
              }
            ],
            "program": {
              "kind": "const",
              "value": [
                140,
                151,
                37,
                143,
                78,
                36,
                137,
                241,
                187,
                61,
                16,
                41,
                20,
                142,
                13,
                131,
                11,
                90,
                19,
                153,
                218,
                255,
                16,
                132,
                4,
                142,
                123,
                216,
                219,
                233,
                248,
                89
              ]
            }
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "tokenProgram",
          "address": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        },
        {
          "name": "associatedTokenProgram",
          "address": "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "distributeFees",
      "docs": [
        "Distribute fees to stakeholders"
      ],
      "discriminator": [
        120,
        56,
        27,
        7,
        53,
        176,
        113,
        186
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "feeCollector",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        },
        {
          "name": "feeVault",
          "writable": true
        },
        {
          "name": "recipient",
          "writable": true
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "emergencyOracleOverride",
      "docs": [
        "Emergency oracle price override"
      ],
      "discriminator": [
        51,
        3,
        217,
        248,
        248,
        41,
        5,
        192
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "price",
          "type": "u64"
        }
      ]
    },
    {
      "name": "emergencyWithdraw",
      "docs": [
        "Emergency withdrawal (only when paused)"
      ],
      "discriminator": [
        239,
        45,
        203,
        64,
        150,
        73,
        218,
        92
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        },
        {
          "name": "vault",
          "writable": true
        },
        {
          "name": "recipient",
          "writable": true
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "executeConditionalOrder",
      "discriminator": [
        41,
        108,
        144,
        244,
        28,
        207,
        141,
        254
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "order",
          "writable": true
        },
        {
          "name": "executor",
          "writable": true,
          "signer": true
        }
      ],
      "args": []
    },
    {
      "name": "executeIcebergChunk",
      "docs": [
        "Execute Iceberg order chunk"
      ],
      "discriminator": [
        17,
        189,
        160,
        255,
        75,
        174,
        97,
        51
      ],
      "accounts": [
        {
          "name": "order",
          "writable": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "executor",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "chunkSize",
          "type": "u64"
        }
      ]
    },
    {
      "name": "executeJitOrder",
      "docs": [
        "Execute order with JIT liquidity"
      ],
      "discriminator": [
        10,
        61,
        86,
        175,
        26,
        155,
        100,
        222
      ],
      "accounts": [
        {
          "name": "jitProvider",
          "writable": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "orderSize",
          "type": "u64"
        },
        {
          "name": "isBuy",
          "type": "bool"
        }
      ]
    },
    {
      "name": "executeTwapChunk",
      "docs": [
        "Execute TWAP order chunk"
      ],
      "discriminator": [
        150,
        92,
        103,
        84,
        62,
        15,
        64,
        143
      ],
      "accounts": [
        {
          "name": "order",
          "writable": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "executor",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "chunkSize",
          "type": "u64"
        }
      ]
    },
    {
      "name": "initializeCollateralAccount",
      "discriminator": [
        39,
        199,
        72,
        137,
        66,
        204,
        33,
        19
      ],
      "accounts": [
        {
          "name": "collateralAccount",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  99,
                  111,
                  108,
                  108,
                  97,
                  116,
                  101,
                  114,
                  97,
                  108
                ]
              },
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "arg",
                "path": "assetType"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "assetType",
          "type": {
            "defined": {
              "name": "collateralType"
            }
          }
        },
        {
          "name": "initialAmount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "initializeCrossCollateralAccount",
      "docs": [
        "Initialize cross-collateral account for a user"
      ],
      "discriminator": [
        215,
        32,
        248,
        216,
        198,
        40,
        118,
        3
      ],
      "accounts": [
        {
          "name": "crossCollateralAccount",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  99,
                  114,
                  111,
                  115,
                  115,
                  95,
                  99,
                  111,
                  108,
                  108,
                  97,
                  116,
                  101,
                  114,
                  97,
                  108
                ]
              },
              {
                "kind": "account",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": []
    },
    {
      "name": "initializeInsuranceFund",
      "docs": [
        "Initialize insurance fund with initial deposit"
      ],
      "discriminator": [
        2,
        239,
        39,
        87,
        50,
        28,
        108,
        12
      ],
      "accounts": [
        {
          "name": "insuranceFund",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  105,
                  110,
                  115,
                  117,
                  114,
                  97,
                  110,
                  99,
                  101,
                  95,
                  102,
                  117,
                  110,
                  100
                ]
              }
            ]
          }
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "initialDeposit",
          "type": "u64"
        }
      ]
    },
    {
      "name": "initializeMarket",
      "discriminator": [
        35,
        35,
        189,
        193,
        155,
        48,
        170,
        203
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  109,
                  97,
                  114,
                  107,
                  101,
                  116
                ]
              },
              {
                "kind": "arg",
                "path": "baseAsset"
              },
              {
                "kind": "arg",
                "path": "quoteAsset"
              }
            ]
          }
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "baseAsset",
          "type": "string"
        },
        {
          "name": "quoteAsset",
          "type": "string"
        },
        {
          "name": "initialPrice",
          "type": "u64"
        },
        {
          "name": "maxLeverage",
          "type": "u8"
        },
        {
          "name": "initialMarginRatio",
          "type": "u16"
        },
        {
          "name": "maintenanceMarginRatio",
          "type": "u16"
        }
      ]
    },
    {
      "name": "initializeProtocolSolVault",
      "docs": [
        "Initialize protocol SOL vault"
      ],
      "discriminator": [
        208,
        172,
        196,
        133,
        181,
        25,
        221,
        170
      ],
      "accounts": [
        {
          "name": "protocolVault",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  112,
                  114,
                  111,
                  116,
                  111,
                  99,
                  111,
                  108,
                  95,
                  115,
                  111,
                  108,
                  95,
                  118,
                  97,
                  117,
                  108,
                  116
                ]
              }
            ]
          }
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": []
    },
    {
      "name": "initializeTokenVault",
      "docs": [
        "Initialize a token vault for protocol deposits"
      ],
      "discriminator": [
        64,
        202,
        113,
        205,
        22,
        210,
        178,
        225
      ],
      "accounts": [
        {
          "name": "vault",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  118,
                  97,
                  117,
                  108,
                  116
                ]
              },
              {
                "kind": "arg",
                "path": "mintAddress"
              }
            ]
          }
        },
        {
          "name": "vaultTokenAccount",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "account",
                "path": "vault"
              },
              {
                "kind": "const",
                "value": [
                  6,
                  221,
                  246,
                  225,
                  215,
                  101,
                  161,
                  147,
                  217,
                  203,
                  225,
                  70,
                  206,
                  235,
                  121,
                  172,
                  28,
                  180,
                  133,
                  237,
                  95,
                  91,
                  55,
                  145,
                  58,
                  140,
                  245,
                  133,
                  126,
                  255,
                  0,
                  169
                ]
              },
              {
                "kind": "account",
                "path": "mint"
              }
            ],
            "program": {
              "kind": "const",
              "value": [
                140,
                151,
                37,
                143,
                78,
                36,
                137,
                241,
                187,
                61,
                16,
                41,
                20,
                142,
                13,
                131,
                11,
                90,
                19,
                153,
                218,
                255,
                16,
                132,
                4,
                142,
                123,
                216,
                219,
                233,
                248,
                89
              ]
            }
          }
        },
        {
          "name": "mint"
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        },
        {
          "name": "tokenProgram",
          "address": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        },
        {
          "name": "associatedTokenProgram",
          "address": "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "mintAddress",
          "type": "pubkey"
        }
      ]
    },
    {
      "name": "jupiterSwap",
      "docs": [
        "Jupiter DEX integration for token swaps"
      ],
      "discriminator": [
        116,
        207,
        0,
        196,
        252,
        120,
        243,
        18
      ],
      "accounts": [
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "inputTokenAccount",
          "writable": true
        },
        {
          "name": "outputTokenAccount",
          "writable": true
        },
        {
          "name": "jupiterProgram"
        }
      ],
      "args": [
        {
          "name": "inputAmount",
          "type": "u64"
        },
        {
          "name": "minimumOutputAmount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "liquidatePosition",
      "discriminator": [
        187,
        74,
        229,
        149,
        102,
        81,
        221,
        68
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "position",
          "writable": true
        },
        {
          "name": "liquidator",
          "writable": true,
          "signer": true
        },
        {
          "name": "vault",
          "writable": true
        }
      ],
      "args": []
    },
    {
      "name": "liquidatePositionCrossCollateral",
      "discriminator": [
        248,
        207,
        109,
        167,
        173,
        234,
        227,
        143
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "position",
          "writable": true
        },
        {
          "name": "liquidator",
          "writable": true,
          "signer": true
        },
        {
          "name": "vault",
          "writable": true
        }
      ],
      "args": []
    },
    {
      "name": "liquidatePositionKeeper",
      "docs": [
        "Execute liquidation through keeper network"
      ],
      "discriminator": [
        36,
        164,
        177,
        239,
        225,
        189,
        22,
        48
      ],
      "accounts": [
        {
          "name": "keeperNetwork",
          "writable": true
        },
        {
          "name": "position",
          "writable": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "keeper",
          "writable": true,
          "signer": true
        },
        {
          "name": "insuranceFund",
          "writable": true
        }
      ],
      "args": [
        {
          "name": "positionId",
          "type": "u64"
        }
      ]
    },
    {
      "name": "openPosition",
      "discriminator": [
        135,
        128,
        47,
        77,
        15,
        152,
        240,
        49
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "position",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  112,
                  111,
                  115,
                  105,
                  116,
                  105,
                  111,
                  110
                ]
              },
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "account",
                "path": "market"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "userCollateral",
          "writable": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "size",
          "type": "u64"
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "leverage",
          "type": "u8"
        }
      ]
    },
    {
      "name": "openPositionCrossCollateral",
      "discriminator": [
        54,
        122,
        163,
        223,
        223,
        97,
        223,
        165
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "position",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  112,
                  111,
                  115,
                  105,
                  116,
                  105,
                  111,
                  110
                ]
              },
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "account",
                "path": "market"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "collateralAccount1",
          "writable": true
        },
        {
          "name": "collateralAccount2",
          "writable": true,
          "optional": true
        },
        {
          "name": "collateralAccount3",
          "writable": true,
          "optional": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "size",
          "type": "u64"
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "leverage",
          "type": "u8"
        },
        {
          "name": "collateralAccounts",
          "type": {
            "vec": "pubkey"
          }
        }
      ]
    },
    {
      "name": "pauseProgram",
      "docs": [
        "Pause all program operations"
      ],
      "discriminator": [
        91,
        86,
        253,
        175,
        66,
        236,
        172,
        124
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": []
    },
    {
      "name": "placeBracketOrder",
      "docs": [
        "Place bracket order (entry + stop loss + take profit)"
      ],
      "discriminator": [
        107,
        18,
        96,
        108,
        210,
        18,
        108,
        50
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "entryOrder",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  98,
                  114,
                  97,
                  99,
                  107,
                  101,
                  116,
                  95,
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "account",
                "path": "market"
              }
            ]
          }
        },
        {
          "name": "stopOrder",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  115,
                  116,
                  111,
                  112,
                  95,
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "account",
                "path": "market"
              }
            ]
          }
        },
        {
          "name": "profitOrder",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  112,
                  114,
                  111,
                  102,
                  105,
                  116,
                  95,
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "account",
                "path": "market"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "userCollateral",
          "writable": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "size",
          "type": "u64"
        },
        {
          "name": "entryPrice",
          "type": "u64"
        },
        {
          "name": "stopLossPrice",
          "type": "u64"
        },
        {
          "name": "takeProfitPrice",
          "type": "u64"
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "leverage",
          "type": "u8"
        }
      ]
    },
    {
      "name": "placeFokOrder",
      "docs": [
        "Place an FOK order (Fill or Kill)"
      ],
      "discriminator": [
        195,
        173,
        15,
        13,
        237,
        137,
        112,
        91
      ],
      "accounts": [
        {
          "name": "order",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "size",
          "type": "u64"
        },
        {
          "name": "price",
          "type": "u64"
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "leverage",
          "type": "u8"
        }
      ]
    },
    {
      "name": "placeIcebergOrder",
      "docs": [
        "Place an Iceberg order (large order split into smaller chunks)"
      ],
      "discriminator": [
        146,
        111,
        167,
        176,
        225,
        53,
        237,
        171
      ],
      "accounts": [
        {
          "name": "order",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "totalSize",
          "type": "u64"
        },
        {
          "name": "displaySize",
          "type": "u64"
        },
        {
          "name": "price",
          "type": "u64"
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "leverage",
          "type": "u8"
        }
      ]
    },
    {
      "name": "placeIocOrder",
      "docs": [
        "Place an IOC order (Immediate or Cancel)"
      ],
      "discriminator": [
        99,
        220,
        219,
        190,
        132,
        253,
        111,
        233
      ],
      "accounts": [
        {
          "name": "order",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "size",
          "type": "u64"
        },
        {
          "name": "price",
          "type": "u64"
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "leverage",
          "type": "u8"
        }
      ]
    },
    {
      "name": "placeOcoOrder",
      "docs": [
        "Place One-Cancels-Other order"
      ],
      "discriminator": [
        64,
        198,
        35,
        187,
        255,
        86,
        181,
        251
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "order",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  111,
                  99,
                  111,
                  95,
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "account",
                "path": "market"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "userCollateral",
          "writable": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "size",
          "type": "u64"
        },
        {
          "name": "entryPrice",
          "type": "u64"
        },
        {
          "name": "stopPrice",
          "type": "u64"
        },
        {
          "name": "limitPrice",
          "type": "u64"
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "leverage",
          "type": "u8"
        }
      ]
    },
    {
      "name": "placeOrder",
      "discriminator": [
        51,
        194,
        155,
        175,
        109,
        130,
        96,
        106
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "order",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "account",
                "path": "market"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "userCollateral",
          "writable": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "orderType",
          "type": {
            "defined": {
              "name": "orderType"
            }
          }
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "size",
          "type": "u64"
        },
        {
          "name": "price",
          "type": "u64"
        },
        {
          "name": "stopPrice",
          "type": "u64"
        },
        {
          "name": "trailingDistance",
          "type": "u64"
        },
        {
          "name": "leverage",
          "type": "u8"
        },
        {
          "name": "expiresAt",
          "type": "i64"
        },
        {
          "name": "hiddenSize",
          "type": "u64"
        },
        {
          "name": "displaySize",
          "type": "u64"
        },
        {
          "name": "timeInForce",
          "type": {
            "defined": {
              "name": "timeInForce"
            }
          }
        },
        {
          "name": "targetPrice",
          "type": "u64"
        },
        {
          "name": "twapDuration",
          "type": "u64"
        },
        {
          "name": "twapInterval",
          "type": "u64"
        }
      ]
    },
    {
      "name": "placePostOnlyOrder",
      "docs": [
        "Place a Post-Only order (Maker only)"
      ],
      "discriminator": [
        253,
        171,
        187,
        207,
        158,
        181,
        93,
        176
      ],
      "accounts": [
        {
          "name": "order",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "size",
          "type": "u64"
        },
        {
          "name": "price",
          "type": "u64"
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "leverage",
          "type": "u8"
        }
      ]
    },
    {
      "name": "placeStopLimitOrder",
      "docs": [
        "Place a Stop-Limit order"
      ],
      "discriminator": [
        140,
        32,
        217,
        153,
        54,
        186,
        237,
        203
      ],
      "accounts": [
        {
          "name": "order",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "size",
          "type": "u64"
        },
        {
          "name": "stopPrice",
          "type": "u64"
        },
        {
          "name": "limitPrice",
          "type": "u64"
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "leverage",
          "type": "u8"
        }
      ]
    },
    {
      "name": "placeTwapOrder",
      "docs": [
        "Place a TWAP order (Time Weighted Average Price)"
      ],
      "discriminator": [
        0,
        140,
        86,
        197,
        70,
        38,
        229,
        173
      ],
      "accounts": [
        {
          "name": "order",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  111,
                  114,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "totalSize",
          "type": "u64"
        },
        {
          "name": "durationSeconds",
          "type": "u64"
        },
        {
          "name": "intervalSeconds",
          "type": "u64"
        },
        {
          "name": "priceLimit",
          "type": "u64"
        },
        {
          "name": "side",
          "type": {
            "defined": {
              "name": "positionSide"
            }
          }
        },
        {
          "name": "leverage",
          "type": "u8"
        }
      ]
    },
    {
      "name": "provideJitLiquidity",
      "docs": [
        "Provide Just-In-Time liquidity"
      ],
      "discriminator": [
        100,
        228,
        185,
        226,
        74,
        41,
        88,
        63
      ],
      "accounts": [
        {
          "name": "jitProvider",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  106,
                  105,
                  116,
                  95,
                  112,
                  114,
                  111,
                  118,
                  105,
                  100,
                  101,
                  114
                ]
              },
              {
                "kind": "account",
                "path": "provider"
              }
            ]
          }
        },
        {
          "name": "provider",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        },
        {
          "name": "feeRate",
          "type": "u16"
        }
      ]
    },
    {
      "name": "registerKeeper",
      "docs": [
        "Register a new keeper in the network"
      ],
      "discriminator": [
        175,
        126,
        140,
        213,
        21,
        174,
        234,
        239
      ],
      "accounts": [
        {
          "name": "keeperNetwork",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  107,
                  101,
                  101,
                  112,
                  101,
                  114,
                  95,
                  110,
                  101,
                  116,
                  119,
                  111,
                  114,
                  107
                ]
              }
            ]
          }
        },
        {
          "name": "keeper",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "stakeAmount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "removeCollateral",
      "discriminator": [
        86,
        222,
        130,
        86,
        92,
        20,
        72,
        65
      ],
      "accounts": [
        {
          "name": "collateralAccount",
          "writable": true
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "userTokenAccount",
          "writable": true
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "removeCrossCollateral",
      "docs": [
        "Remove collateral from cross-collateral account"
      ],
      "discriminator": [
        23,
        94,
        11,
        143,
        35,
        203,
        180,
        150
      ],
      "accounts": [
        {
          "name": "crossCollateralAccount",
          "writable": true
        },
        {
          "name": "oraclePriceFeed"
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "assetType",
          "type": {
            "defined": {
              "name": "collateralType"
            }
          }
        },
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "removeOracleFeed",
      "docs": [
        "Remove oracle feed"
      ],
      "discriminator": [
        251,
        233,
        30,
        162,
        136,
        146,
        120,
        56
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "oracleManager",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "feedIndex",
          "type": "u8"
        }
      ]
    },
    {
      "name": "resetCircuitBreaker",
      "docs": [
        "Reset circuit breaker after emergency is resolved"
      ],
      "discriminator": [
        225,
        48,
        84,
        136,
        90,
        146,
        26,
        149
      ],
      "accounts": [
        {
          "name": "circuitBreaker",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": []
    },
    {
      "name": "resumeProgram",
      "docs": [
        "Resume program operations"
      ],
      "discriminator": [
        253,
        125,
        38,
        109,
        196,
        141,
        189,
        30
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": []
    },
    {
      "name": "settleFunding",
      "discriminator": [
        11,
        251,
        12,
        161,
        199,
        228,
        133,
        87
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "keeper",
          "writable": true,
          "signer": true
        }
      ],
      "args": []
    },
    {
      "name": "triggerCircuitBreaker",
      "docs": [
        "Trigger circuit breaker for emergency situations"
      ],
      "discriminator": [
        18,
        87,
        214,
        42,
        239,
        136,
        160,
        81
      ],
      "accounts": [
        {
          "name": "circuitBreaker",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  99,
                  105,
                  114,
                  99,
                  117,
                  105,
                  116,
                  95,
                  98,
                  114,
                  101,
                  97,
                  107,
                  101,
                  114
                ]
              }
            ]
          }
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "breakerType",
          "type": {
            "defined": {
              "name": "circuitBreakerType"
            }
          }
        }
      ]
    },
    {
      "name": "updateCollateralConfig",
      "docs": [
        "Update collateral configuration"
      ],
      "discriminator": [
        87,
        81,
        178,
        108,
        188,
        71,
        197,
        125
      ],
      "accounts": [
        {
          "name": "collateralConfig",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "initialAssetWeight",
          "type": "u16"
        },
        {
          "name": "maintenanceAssetWeight",
          "type": "u16"
        },
        {
          "name": "initialLiabilityWeight",
          "type": "u16"
        },
        {
          "name": "maintenanceLiabilityWeight",
          "type": "u16"
        },
        {
          "name": "imfFactor",
          "type": "u16"
        },
        {
          "name": "maxCollateralAmount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "updateCollateralValue",
      "discriminator": [
        71,
        159,
        194,
        46,
        78,
        127,
        208,
        113
      ],
      "accounts": [
        {
          "name": "collateralAccount",
          "writable": true
        },
        {
          "name": "priceFeed"
        },
        {
          "name": "keeper",
          "signer": true
        }
      ],
      "args": [
        {
          "name": "newPrice",
          "type": "u64"
        }
      ]
    },
    {
      "name": "updateFundingFees",
      "docs": [
        "Update funding fee parameters"
      ],
      "discriminator": [
        33,
        65,
        8,
        5,
        13,
        128,
        175,
        104
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "feeCollector",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "fundingRateCap",
          "type": "i64"
        },
        {
          "name": "fundingRateFloor",
          "type": "i64"
        }
      ]
    },
    {
      "name": "updateKeeperPerformance",
      "docs": [
        "Update keeper performance score"
      ],
      "discriminator": [
        158,
        126,
        214,
        187,
        95,
        208,
        188,
        119
      ],
      "accounts": [
        {
          "name": "keeperNetwork",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "keeperPubkey",
          "type": "pubkey"
        },
        {
          "name": "performanceScore",
          "type": "u16"
        }
      ]
    },
    {
      "name": "updateMarketParameters",
      "docs": [
        "Update market parameters"
      ],
      "discriminator": [
        183,
        69,
        5,
        76,
        114,
        136,
        129,
        65
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "maxLeverage",
          "type": {
            "option": "u8"
          }
        },
        {
          "name": "initialMarginRatio",
          "type": {
            "option": "u16"
          }
        },
        {
          "name": "maintenanceMarginRatio",
          "type": {
            "option": "u16"
          }
        }
      ]
    },
    {
      "name": "updateOraclePrice",
      "discriminator": [
        14,
        35,
        163,
        150,
        65,
        116,
        149,
        154
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "priceFeed"
        },
        {
          "name": "authority",
          "signer": true
        }
      ],
      "args": [
        {
          "name": "newPrice",
          "type": "u64"
        }
      ]
    },
    {
      "name": "updateOracleWeights",
      "docs": [
        "Update oracle feed weights"
      ],
      "discriminator": [
        28,
        98,
        5,
        57,
        32,
        147,
        142,
        171
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "oracleManager",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "weights",
          "type": "bytes"
        }
      ]
    },
    {
      "name": "updateProgramAuthority",
      "docs": [
        "Update program authority"
      ],
      "discriminator": [
        15,
        214,
        181,
        183,
        136,
        194,
        245,
        18
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "newAuthority",
          "type": "pubkey"
        }
      ]
    },
    {
      "name": "updatePythPrice",
      "docs": [
        "Update Pyth price feed"
      ],
      "discriminator": [
        221,
        14,
        48,
        182,
        7,
        77,
        193,
        167
      ],
      "accounts": [
        {
          "name": "market",
          "writable": true
        },
        {
          "name": "pythPriceFeed"
        },
        {
          "name": "keeper",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "priceFeed",
          "type": "pubkey"
        }
      ]
    },
    {
      "name": "updateRiskParameters",
      "docs": [
        "Update risk management parameters"
      ],
      "discriminator": [
        123,
        122,
        245,
        194,
        157,
        28,
        86,
        77
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "insuranceFund",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "maxPositionSize",
          "type": "u64"
        },
        {
          "name": "maxLeverage",
          "type": "u8"
        },
        {
          "name": "liquidationThreshold",
          "type": "u16"
        }
      ]
    },
    {
      "name": "updateTradingFees",
      "docs": [
        "Update trading fee rates"
      ],
      "discriminator": [
        179,
        12,
        134,
        91,
        245,
        237,
        59,
        128
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "feeCollector",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "makerFeeRate",
          "type": "u16"
        },
        {
          "name": "takerFeeRate",
          "type": "u16"
        }
      ]
    },
    {
      "name": "updateUserAccount",
      "discriminator": [
        147,
        83,
        243,
        122,
        110,
        128,
        92,
        33
      ],
      "accounts": [
        {
          "name": "userAccount",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "totalCollateral",
          "type": {
            "option": "u64"
          }
        },
        {
          "name": "totalPositions",
          "type": {
            "option": "u16"
          }
        },
        {
          "name": "totalOrders",
          "type": {
            "option": "u16"
          }
        },
        {
          "name": "accountHealth",
          "type": {
            "option": "u16"
          }
        },
        {
          "name": "liquidationPrice",
          "type": {
            "option": "u64"
          }
        }
      ]
    },
    {
      "name": "updateWhitelist",
      "docs": [
        "Update user whitelist"
      ],
      "discriminator": [
        94,
        198,
        33,
        20,
        192,
        97,
        44,
        59
      ],
      "accounts": [
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        }
      ],
      "args": [
        {
          "name": "user",
          "type": "pubkey"
        },
        {
          "name": "isWhitelisted",
          "type": "bool"
        }
      ]
    },
    {
      "name": "withdrawInsuranceFund",
      "docs": [
        "Withdraw funds from insurance pool (admin only)"
      ],
      "discriminator": [
        228,
        196,
        230,
        109,
        1,
        95,
        171,
        196
      ],
      "accounts": [
        {
          "name": "insuranceFund",
          "writable": true
        },
        {
          "name": "programState",
          "writable": true
        },
        {
          "name": "authority",
          "writable": true,
          "signer": true
        },
        {
          "name": "fundVault",
          "writable": true
        },
        {
          "name": "recipientTokenAccount",
          "writable": true
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "withdrawNativeSol",
      "docs": [
        "Withdraw native SOL from user account"
      ],
      "discriminator": [
        201,
        104,
        187,
        105,
        80,
        204,
        84,
        138
      ],
      "accounts": [
        {
          "name": "userAccount",
          "writable": true
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "protocolVault",
          "docs": [
            "Protocol SOL vault PDA - this will hold the deposited SOL"
          ],
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  112,
                  114,
                  111,
                  116,
                  111,
                  99,
                  111,
                  108,
                  95,
                  115,
                  111,
                  108,
                  95,
                  118,
                  97,
                  117,
                  108,
                  116
                ]
              }
            ]
          }
        },
        {
          "name": "collateralAccount",
          "docs": [
            "SOL collateral account for the user - must exist for withdrawal"
          ],
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  99,
                  111,
                  108,
                  108,
                  97,
                  116,
                  101,
                  114,
                  97,
                  108
                ]
              },
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "const",
                "value": [
                  83,
                  79,
                  76
                ]
              }
            ]
          }
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    },
    {
      "name": "withdrawTokens",
      "docs": [
        "Withdraw tokens from protocol vault"
      ],
      "discriminator": [
        2,
        4,
        225,
        61,
        19,
        182,
        106,
        170
      ],
      "accounts": [
        {
          "name": "vault",
          "writable": true
        },
        {
          "name": "vaultTokenAccount",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "account",
                "path": "vault"
              },
              {
                "kind": "const",
                "value": [
                  6,
                  221,
                  246,
                  225,
                  215,
                  101,
                  161,
                  147,
                  217,
                  203,
                  225,
                  70,
                  206,
                  235,
                  121,
                  172,
                  28,
                  180,
                  133,
                  237,
                  95,
                  91,
                  55,
                  145,
                  58,
                  140,
                  245,
                  133,
                  126,
                  255,
                  0,
                  169
                ]
              },
              {
                "kind": "account",
                "path": "vault.mint",
                "account": "tokenVault"
              }
            ],
            "program": {
              "kind": "const",
              "value": [
                140,
                151,
                37,
                143,
                78,
                36,
                137,
                241,
                187,
                61,
                16,
                41,
                20,
                142,
                13,
                131,
                11,
                90,
                19,
                153,
                218,
                255,
                16,
                132,
                4,
                142,
                123,
                216,
                219,
                233,
                248,
                89
              ]
            }
          }
        },
        {
          "name": "userTokenAccount",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "account",
                "path": "user"
              },
              {
                "kind": "const",
                "value": [
                  6,
                  221,
                  246,
                  225,
                  215,
                  101,
                  161,
                  147,
                  217,
                  203,
                  225,
                  70,
                  206,
                  235,
                  121,
                  172,
                  28,
                  180,
                  133,
                  237,
                  95,
                  91,
                  55,
                  145,
                  58,
                  140,
                  245,
                  133,
                  126,
                  255,
                  0,
                  169
                ]
              },
              {
                "kind": "account",
                "path": "vault.mint",
                "account": "tokenVault"
              }
            ],
            "program": {
              "kind": "const",
              "value": [
                140,
                151,
                37,
                143,
                78,
                36,
                137,
                241,
                187,
                61,
                16,
                41,
                20,
                142,
                13,
                131,
                11,
                90,
                19,
                153,
                218,
                255,
                16,
                132,
                4,
                142,
                123,
                216,
                219,
                233,
                248,
                89
              ]
            }
          }
        },
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "tokenProgram",
          "address": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        },
        {
          "name": "associatedTokenProgram",
          "address": "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
        }
      ],
      "args": [
        {
          "name": "amount",
          "type": "u64"
        }
      ]
    }
  ],
  "accounts": [
    {
      "name": "circuitBreaker",
      "discriminator": [
        123,
        141,
        226,
        182,
        3,
        205,
        19,
        253
      ]
    },
    {
      "name": "collateralAccount",
      "discriminator": [
        134,
        2,
        192,
        39,
        194,
        239,
        19,
        17
      ]
    },
    {
      "name": "collateralConfig",
      "discriminator": [
        150,
        147,
        210,
        201,
        79,
        202,
        93,
        49
      ]
    },
    {
      "name": "crossCollateralAccount",
      "discriminator": [
        112,
        77,
        228,
        123,
        169,
        206,
        150,
        99
      ]
    },
    {
      "name": "feeCollector",
      "discriminator": [
        250,
        213,
        73,
        200,
        175,
        76,
        225,
        213
      ]
    },
    {
      "name": "insuranceFund",
      "discriminator": [
        43,
        134,
        170,
        87,
        102,
        16,
        142,
        147
      ]
    },
    {
      "name": "jitProvider",
      "discriminator": [
        20,
        83,
        71,
        135,
        80,
        14,
        183,
        159
      ]
    },
    {
      "name": "keeperNetwork",
      "discriminator": [
        221,
        221,
        106,
        72,
        85,
        224,
        136,
        246
      ]
    },
    {
      "name": "market",
      "discriminator": [
        219,
        190,
        213,
        55,
        0,
        227,
        198,
        154
      ]
    },
    {
      "name": "oracleManager",
      "discriminator": [
        153,
        29,
        174,
        0,
        184,
        99,
        104,
        2
      ]
    },
    {
      "name": "order",
      "discriminator": [
        134,
        173,
        223,
        185,
        77,
        86,
        28,
        51
      ]
    },
    {
      "name": "position",
      "discriminator": [
        170,
        188,
        143,
        228,
        122,
        64,
        247,
        208
      ]
    },
    {
      "name": "programState",
      "discriminator": [
        77,
        209,
        137,
        229,
        149,
        67,
        167,
        230
      ]
    },
    {
      "name": "protocolSolVault",
      "discriminator": [
        225,
        223,
        222,
        1,
        74,
        62,
        59,
        131
      ]
    },
    {
      "name": "tokenVault",
      "discriminator": [
        121,
        7,
        84,
        254,
        151,
        228,
        43,
        144
      ]
    },
    {
      "name": "userAccount",
      "discriminator": [
        211,
        33,
        136,
        16,
        186,
        110,
        242,
        127
      ]
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "invalidLeverage",
      "msg": "Invalid leverage amount"
    },
    {
      "code": 6001,
      "name": "invalidSize",
      "msg": "Invalid position size"
    },
    {
      "code": 6002,
      "name": "insufficientCollateral",
      "msg": "Insufficient collateral"
    },
    {
      "code": 6003,
      "name": "positionHealthy",
      "msg": "Position is healthy, no liquidation needed"
    },
    {
      "code": 6004,
      "name": "positionNotLiquidatable",
      "msg": "Position is not liquidatable"
    },
    {
      "code": 6005,
      "name": "fundingNotDue",
      "msg": "Funding settlement not due yet"
    },
    {
      "code": 6006,
      "name": "invalidPrice",
      "msg": "Invalid order price"
    },
    {
      "code": 6007,
      "name": "invalidStopPrice",
      "msg": "Invalid stop price"
    },
    {
      "code": 6008,
      "name": "invalidTrailingDistance",
      "msg": "Invalid trailing distance"
    },
    {
      "code": 6009,
      "name": "orderNotPending",
      "msg": "Order is not pending"
    },
    {
      "code": 6010,
      "name": "unauthorizedUser",
      "msg": "Unauthorized user"
    },
    {
      "code": 6011,
      "name": "conditionNotMet",
      "msg": "Condition not met for execution"
    },
    {
      "code": 6012,
      "name": "invalidMaxLeverage",
      "msg": "Invalid max leverage"
    },
    {
      "code": 6013,
      "name": "invalidMarginRatio",
      "msg": "Invalid margin ratio"
    },
    {
      "code": 6014,
      "name": "marketInactive",
      "msg": "Market is inactive"
    },
    {
      "code": 6015,
      "name": "positionTooLarge",
      "msg": "Position too large"
    },
    {
      "code": 6016,
      "name": "priceStale",
      "msg": "Price is stale"
    },
    {
      "code": 6017,
      "name": "priceTooHigh",
      "msg": "Price too high"
    },
    {
      "code": 6018,
      "name": "priceTooLow",
      "msg": "Price too low"
    },
    {
      "code": 6019,
      "name": "trailingDistanceTooLarge",
      "msg": "Trailing distance too large"
    },
    {
      "code": 6020,
      "name": "orderExpired",
      "msg": "Order expired"
    },
    {
      "code": 6021,
      "name": "orderExpirationTooLong",
      "msg": "Order expiration too long"
    },
    {
      "code": 6022,
      "name": "positionAlreadyClosed",
      "msg": "Position already closed"
    },
    {
      "code": 6023,
      "name": "invalidDuration",
      "msg": "Invalid duration"
    },
    {
      "code": 6024,
      "name": "invalidInterval",
      "msg": "Invalid interval"
    },
    {
      "code": 6025,
      "name": "invalidTargetPrice",
      "msg": "Invalid target price"
    },
    {
      "code": 6026,
      "name": "invalidAmount",
      "msg": "Invalid amount"
    },
    {
      "code": 6027,
      "name": "collateralAccountInactive",
      "msg": "Collateral account inactive"
    },
    {
      "code": 6028,
      "name": "noCollateralProvided",
      "msg": "No collateral provided"
    },
    {
      "code": 6029,
      "name": "noPositionsToRemove",
      "msg": "No positions to remove"
    },
    {
      "code": 6030,
      "name": "noOrdersToRemove",
      "msg": "No orders to remove"
    },
    {
      "code": 6031,
      "name": "invalidHealthValue",
      "msg": "Invalid health value"
    },
    {
      "code": 6032,
      "name": "accountHasPositions",
      "msg": "Account has open positions"
    },
    {
      "code": 6033,
      "name": "accountHasOrders",
      "msg": "Account has active orders"
    },
    {
      "code": 6034,
      "name": "accountInactive",
      "msg": "Account is not active"
    },
    {
      "code": 6035,
      "name": "accountAlreadyExists",
      "msg": "Account already exists"
    },
    {
      "code": 6036,
      "name": "accountNotFound",
      "msg": "Account not found"
    },
    {
      "code": 6037,
      "name": "invalidTokenAmount",
      "msg": "Invalid token amount"
    },
    {
      "code": 6038,
      "name": "vaultInactive",
      "msg": "Vault is inactive"
    },
    {
      "code": 6039,
      "name": "insufficientVaultBalance",
      "msg": "Insufficient vault balance"
    },
    {
      "code": 6040,
      "name": "unauthorizedTokenAuthority",
      "msg": "Unauthorized token authority"
    },
    {
      "code": 6041,
      "name": "unauthorizedTokenUser",
      "msg": "Unauthorized token user"
    },
    {
      "code": 6042,
      "name": "tokenAccountNotFound",
      "msg": "Token account not found"
    },
    {
      "code": 6043,
      "name": "invalidMintAddress",
      "msg": "Invalid mint address"
    },
    {
      "code": 6044,
      "name": "pdaDerivationFailed",
      "msg": "PDA derivation failed"
    },
    {
      "code": 6045,
      "name": "invalidPdaOwner",
      "msg": "Invalid PDA owner"
    },
    {
      "code": 6046,
      "name": "invalidInsuranceFundOperation",
      "msg": "Invalid insurance fund operation"
    },
    {
      "code": 6047,
      "name": "insufficientInsuranceFundBalance",
      "msg": "Insufficient insurance fund balance"
    },
    {
      "code": 6048,
      "name": "programPaused",
      "msg": "Program is paused"
    },
    {
      "code": 6049,
      "name": "invalidFeeParameters",
      "msg": "Invalid fee parameters"
    },
    {
      "code": 6050,
      "name": "invalidOracleWeight",
      "msg": "Invalid oracle weight"
    },
    {
      "code": 6051,
      "name": "oracleFeedNotFound",
      "msg": "Oracle feed not found"
    },
    {
      "code": 6052,
      "name": "unauthorizedAdminOperation",
      "msg": "Unauthorized admin operation"
    },
    {
      "code": 6053,
      "name": "insufficientKeeperStake",
      "msg": "Insufficient keeper stake"
    },
    {
      "code": 6054,
      "name": "keeperNotRegistered",
      "msg": "Keeper not registered"
    },
    {
      "code": 6055,
      "name": "keeperInactive",
      "msg": "Keeper is inactive"
    },
    {
      "code": 6056,
      "name": "invalidPerformanceScore",
      "msg": "Invalid performance score"
    },
    {
      "code": 6057,
      "name": "circuitBreakerAlreadyTriggered",
      "msg": "Circuit breaker is already triggered"
    },
    {
      "code": 6058,
      "name": "circuitBreakerNotTriggered",
      "msg": "Circuit breaker is not triggered"
    },
    {
      "code": 6059,
      "name": "invalidFeeRate",
      "msg": "Invalid fee rate"
    },
    {
      "code": 6060,
      "name": "insufficientJitLiquidity",
      "msg": "Insufficient JIT liquidity"
    },
    {
      "code": 6061,
      "name": "jitProviderInactive",
      "msg": "JIT provider not active"
    },
    {
      "code": 6062,
      "name": "invalidVaultStrategy",
      "msg": "Invalid vault strategy"
    },
    {
      "code": 6063,
      "name": "insufficientVaultCapital",
      "msg": "Insufficient vault capital"
    },
    {
      "code": 6064,
      "name": "invalidPointsMultiplier",
      "msg": "Invalid points multiplier"
    },
    {
      "code": 6065,
      "name": "userNotFoundInPointsSystem",
      "msg": "User not found in points system"
    },
    {
      "code": 6066,
      "name": "invalidTwapParameters",
      "msg": "Invalid TWAP parameters"
    },
    {
      "code": 6067,
      "name": "invalidOrderType",
      "msg": "Invalid order type"
    },
    {
      "code": 6068,
      "name": "collateralTypeNotActive",
      "msg": "Collateral type is not active"
    },
    {
      "code": 6069,
      "name": "exceedsMaxCollateral",
      "msg": "Exceeds maximum collateral amount"
    },
    {
      "code": 6070,
      "name": "collateralAssetNotFound",
      "msg": "Collateral asset not found"
    },
    {
      "code": 6071,
      "name": "insufficientHealthFactor",
      "msg": "Insufficient health factor"
    },
    {
      "code": 6072,
      "name": "invalidWeight",
      "msg": "Invalid weight"
    }
  ],
  "types": [
    {
      "name": "circuitBreaker",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "isTriggered",
            "type": "bool"
          },
          {
            "name": "triggerTime",
            "type": "i64"
          },
          {
            "name": "resetTime",
            "type": "i64"
          },
          {
            "name": "breakerType",
            "type": {
              "defined": {
                "name": "circuitBreakerType"
              }
            }
          },
          {
            "name": "triggeredBy",
            "type": "pubkey"
          },
          {
            "name": "resetBy",
            "type": "pubkey"
          },
          {
            "name": "priceChangeThreshold",
            "type": "u16"
          },
          {
            "name": "volumeThreshold",
            "type": "u64"
          },
          {
            "name": "timeWindow",
            "type": "u64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "circuitBreakerType",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "priceVolatility"
          },
          {
            "name": "volumeSpike"
          },
          {
            "name": "systemOverload"
          },
          {
            "name": "emergencyStop"
          }
        ]
      }
    },
    {
      "name": "collateralAccount",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "pubkey"
          },
          {
            "name": "assetType",
            "type": {
              "defined": {
                "name": "collateralType"
              }
            }
          },
          {
            "name": "amount",
            "type": "u64"
          },
          {
            "name": "valueUsd",
            "type": "u64"
          },
          {
            "name": "lastUpdated",
            "type": "i64"
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "collateralAsset",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "assetType",
            "type": {
              "defined": {
                "name": "collateralType"
              }
            }
          },
          {
            "name": "amount",
            "type": "u64"
          },
          {
            "name": "valueUsd",
            "type": "u64"
          },
          {
            "name": "assetWeight",
            "type": "u16"
          },
          {
            "name": "liabilityWeight",
            "type": "u16"
          },
          {
            "name": "lastPriceUpdate",
            "type": "i64"
          }
        ]
      }
    },
    {
      "name": "collateralConfig",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "assetType",
            "type": {
              "defined": {
                "name": "collateralType"
              }
            }
          },
          {
            "name": "initialAssetWeight",
            "type": "u16"
          },
          {
            "name": "maintenanceAssetWeight",
            "type": "u16"
          },
          {
            "name": "initialLiabilityWeight",
            "type": "u16"
          },
          {
            "name": "maintenanceLiabilityWeight",
            "type": "u16"
          },
          {
            "name": "imfFactor",
            "type": "u16"
          },
          {
            "name": "maxCollateralAmount",
            "type": "u64"
          },
          {
            "name": "oraclePriceFeed",
            "type": "pubkey"
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "collateralType",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "sol"
          },
          {
            "name": "usdc"
          },
          {
            "name": "btc"
          },
          {
            "name": "eth"
          },
          {
            "name": "usdt"
          },
          {
            "name": "avax"
          },
          {
            "name": "matic"
          },
          {
            "name": "arb"
          },
          {
            "name": "op"
          },
          {
            "name": "doge"
          },
          {
            "name": "ada"
          },
          {
            "name": "dot"
          },
          {
            "name": "link"
          }
        ]
      }
    },
    {
      "name": "crossCollateralAccount",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "pubkey"
          },
          {
            "name": "totalCollateralValue",
            "type": "u64"
          },
          {
            "name": "totalBorrowedValue",
            "type": "u64"
          },
          {
            "name": "collateralAssets",
            "type": {
              "vec": {
                "defined": {
                  "name": "collateralAsset"
                }
              }
            }
          },
          {
            "name": "initialAssetWeight",
            "type": "u16"
          },
          {
            "name": "maintenanceAssetWeight",
            "type": "u16"
          },
          {
            "name": "initialLiabilityWeight",
            "type": "u16"
          },
          {
            "name": "maintenanceLiabilityWeight",
            "type": "u16"
          },
          {
            "name": "imfFactor",
            "type": "u16"
          },
          {
            "name": "lastHealthCheck",
            "type": "i64"
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "feeCollector",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "tradingFeesCollected",
            "type": "u64"
          },
          {
            "name": "fundingFeesCollected",
            "type": "u64"
          },
          {
            "name": "makerFeeRate",
            "type": "u16"
          },
          {
            "name": "takerFeeRate",
            "type": "u16"
          },
          {
            "name": "fundingRateCap",
            "type": "i64"
          },
          {
            "name": "fundingRateFloor",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "insuranceFund",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "totalDeposits",
            "type": "u64"
          },
          {
            "name": "totalWithdrawals",
            "type": "u64"
          },
          {
            "name": "utilizationRate",
            "type": "u16"
          },
          {
            "name": "maxUtilization",
            "type": "u16"
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "jitProvider",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "providerPubkey",
            "type": "pubkey"
          },
          {
            "name": "availableLiquidity",
            "type": "u64"
          },
          {
            "name": "feeRate",
            "type": "u16"
          },
          {
            "name": "totalVolume",
            "type": "u64"
          },
          {
            "name": "totalFeesEarned",
            "type": "u64"
          },
          {
            "name": "minOrderSize",
            "type": "u64"
          },
          {
            "name": "maxOrderSize",
            "type": "u64"
          },
          {
            "name": "lastUpdate",
            "type": "i64"
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "keeperInfo",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "keeperPubkey",
            "type": "pubkey"
          },
          {
            "name": "stakeAmount",
            "type": "u64"
          },
          {
            "name": "performanceScore",
            "type": "u16"
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "totalLiquidations",
            "type": "u32"
          },
          {
            "name": "totalRewardsEarned",
            "type": "u64"
          },
          {
            "name": "lastActivity",
            "type": "i64"
          }
        ]
      }
    },
    {
      "name": "keeperNetwork",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "totalStake",
            "type": "u64"
          },
          {
            "name": "keepers",
            "type": {
              "vec": {
                "defined": {
                  "name": "keeperInfo"
                }
              }
            }
          },
          {
            "name": "liquidationRewardsPool",
            "type": "u64"
          },
          {
            "name": "minStakeRequirement",
            "type": "u64"
          },
          {
            "name": "performanceThreshold",
            "type": "u16"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "market",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "baseAsset",
            "type": "string"
          },
          {
            "name": "quoteAsset",
            "type": "string"
          },
          {
            "name": "baseReserve",
            "type": "u64"
          },
          {
            "name": "quoteReserve",
            "type": "u64"
          },
          {
            "name": "fundingRate",
            "type": "i64"
          },
          {
            "name": "lastFundingTime",
            "type": "i64"
          },
          {
            "name": "fundingInterval",
            "type": "i64"
          },
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "maxLeverage",
            "type": "u8"
          },
          {
            "name": "initialMarginRatio",
            "type": "u16"
          },
          {
            "name": "maintenanceMarginRatio",
            "type": "u16"
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "lastOraclePrice",
            "type": "u64"
          },
          {
            "name": "lastOracleUpdate",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "oracleFeed",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "feedType",
            "type": {
              "defined": {
                "name": "oracleFeedType"
              }
            }
          },
          {
            "name": "feedAccount",
            "type": "pubkey"
          },
          {
            "name": "weight",
            "type": "u8"
          },
          {
            "name": "isActive",
            "type": "bool"
          }
        ]
      }
    },
    {
      "name": "oracleFeedType",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "pyth"
          },
          {
            "name": "switchboard"
          },
          {
            "name": "chainlink"
          }
        ]
      }
    },
    {
      "name": "oracleManager",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "feeds",
            "type": {
              "vec": {
                "defined": {
                  "name": "oracleFeed"
                }
              }
            }
          },
          {
            "name": "weights",
            "type": "bytes"
          },
          {
            "name": "maxDeviation",
            "type": "u16"
          },
          {
            "name": "stalenessThreshold",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "order",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "pubkey"
          },
          {
            "name": "market",
            "type": "pubkey"
          },
          {
            "name": "orderType",
            "type": {
              "defined": {
                "name": "orderType"
              }
            }
          },
          {
            "name": "side",
            "type": {
              "defined": {
                "name": "positionSide"
              }
            }
          },
          {
            "name": "size",
            "type": "u64"
          },
          {
            "name": "price",
            "type": "u64"
          },
          {
            "name": "stopPrice",
            "type": "u64"
          },
          {
            "name": "trailingDistance",
            "type": "u64"
          },
          {
            "name": "leverage",
            "type": "u8"
          },
          {
            "name": "status",
            "type": {
              "defined": {
                "name": "orderStatus"
              }
            }
          },
          {
            "name": "createdAt",
            "type": "i64"
          },
          {
            "name": "expiresAt",
            "type": "i64"
          },
          {
            "name": "filledSize",
            "type": "u64"
          },
          {
            "name": "bump",
            "type": "u8"
          },
          {
            "name": "hiddenSize",
            "type": "u64"
          },
          {
            "name": "displaySize",
            "type": "u64"
          },
          {
            "name": "timeInForce",
            "type": {
              "defined": {
                "name": "timeInForce"
              }
            }
          },
          {
            "name": "targetPrice",
            "type": "u64"
          },
          {
            "name": "parentOrder",
            "type": {
              "option": "pubkey"
            }
          },
          {
            "name": "twapDuration",
            "type": "u64"
          },
          {
            "name": "twapInterval",
            "type": "u64"
          }
        ]
      }
    },
    {
      "name": "orderStatus",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "pending"
          },
          {
            "name": "filled"
          },
          {
            "name": "cancelled"
          },
          {
            "name": "expired"
          },
          {
            "name": "partiallyFilled"
          },
          {
            "name": "rejected"
          }
        ]
      }
    },
    {
      "name": "orderType",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "market"
          },
          {
            "name": "limit"
          },
          {
            "name": "stopLoss"
          },
          {
            "name": "takeProfit"
          },
          {
            "name": "trailingStop"
          },
          {
            "name": "postOnly"
          },
          {
            "name": "ioc"
          },
          {
            "name": "fok"
          },
          {
            "name": "iceberg"
          },
          {
            "name": "twap"
          },
          {
            "name": "stopLimit"
          },
          {
            "name": "bracket"
          }
        ]
      }
    },
    {
      "name": "position",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "pubkey"
          },
          {
            "name": "market",
            "type": "pubkey"
          },
          {
            "name": "size",
            "type": "u64"
          },
          {
            "name": "side",
            "type": {
              "defined": {
                "name": "positionSide"
              }
            }
          },
          {
            "name": "leverage",
            "type": "u8"
          },
          {
            "name": "entryPrice",
            "type": "u64"
          },
          {
            "name": "margin",
            "type": "u64"
          },
          {
            "name": "unrealizedPnl",
            "type": "i64"
          },
          {
            "name": "createdAt",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          },
          {
            "name": "collateralAccounts",
            "type": {
              "vec": "pubkey"
            }
          },
          {
            "name": "totalCollateralValue",
            "type": "u64"
          }
        ]
      }
    },
    {
      "name": "positionSide",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "long"
          },
          {
            "name": "short"
          }
        ]
      }
    },
    {
      "name": "programState",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "isPaused",
            "type": "bool"
          },
          {
            "name": "insuranceFund",
            "type": "pubkey"
          },
          {
            "name": "feeCollector",
            "type": "pubkey"
          },
          {
            "name": "oracleManager",
            "type": "pubkey"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "protocolSolVault",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "totalDeposits",
            "type": "u64"
          },
          {
            "name": "totalWithdrawals",
            "type": "u64"
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "timeInForce",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "gtc"
          },
          {
            "name": "ioc"
          },
          {
            "name": "fok"
          },
          {
            "name": "gtd"
          }
        ]
      }
    },
    {
      "name": "tokenVault",
      "docs": [
        "Token Operations Module",
        "Following Solana Cookbook patterns for professional token management",
        "https://solanacookbook.com/references/programs.html#token-program",
        "Token Vault Account Structure"
      ],
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "mint",
            "type": "pubkey"
          },
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "totalDeposits",
            "type": "u64"
          },
          {
            "name": "totalWithdrawals",
            "type": "u64"
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "userAccount",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "accountIndex",
            "type": "u16"
          },
          {
            "name": "totalCollateral",
            "type": "u64"
          },
          {
            "name": "totalPositions",
            "type": "u16"
          },
          {
            "name": "totalOrders",
            "type": "u16"
          },
          {
            "name": "accountHealth",
            "type": "u16"
          },
          {
            "name": "liquidationPrice",
            "type": "u64"
          },
          {
            "name": "createdAt",
            "type": "i64"
          },
          {
            "name": "lastActivity",
            "type": "i64"
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "userAction",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "deposit"
          },
          {
            "name": "withdraw"
          },
          {
            "name": "trade"
          },
          {
            "name": "createPosition"
          },
          {
            "name": "closePosition"
          }
        ]
      }
    }
  ]
};
