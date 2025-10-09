# Foundry Setup & Deployment Guide for SolarSync

## Quick Start (5 minutes)

### 1. Install Foundry
```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

### 2. Verify Installation
```bash
forge --version
anvil --version
cast --version
```

### 3. Create foundry.toml (see artifact above)
Save the `foundry.toml` file in your `contracts/` directory.

### 4. Create .env file
```bash
cd contracts
cat > .env << 'EOF'
# Network selection
NETWORK=anvil

# Private key for deployment (get from Hedera wallet)
PRIVATE_KEY=your_private_key_here

# Hedera details
HEDERA_ACCOUNT_ID=0.0.xxxxx
HEDERA_PRIVATE_KEY=your_hedera_pk

# API keys
ETHERSCAN_API_KEY=your_etherscan_key
INFURA_API_KEY=your_infura_key

# ML API
ML_API_URL=http://localhost:8000
ORACLE_UPDATE_INTERVAL=3600
EOF
```

### 5. Install Dependencies
```bash
cd contracts
forge install OpenZeppelin/openzeppelin-contracts
forge install smartcontractkit/chainlink
```

### 6. Build Contracts
```bash
forge build
```

**Expected output:**
```
Compiling 25 files with 0.8.20
Solc 0.8.20 finished in 3.42s
Compiler run successful!
```

---

## foundry.toml Explained (Section by Section)

### [profile.default]
This is your main configuration profile used for all operations.

**Key settings:**
- `src = "src"` → Location of your Solidity files
- `test = "test"` → Location of test files
- `script = "script"` → Location of deployment scripts
- `solc_version = "0.8.20"` → Solidity compiler version (must match your contracts)
- `optimizer_runs = 200` → Balance between code size and execution efficiency

**Why these values for SolarSync:**
- Solidity 0.8.20 has security improvements over 0.8.19
- 200 optimizer runs is standard for mainnet (lower = faster compilation, higher = cheaper execution)

---

### [profile.test]
Settings specifically for running tests.

```toml
[profile.test]
inherits = "default"      # Start with default settings
verbosity = 2             # Show more test output
evm_version = "london"    # Use London fork for consistency
```

**Why:**
- Tests need verbose output to debug failures
- London EVM version is widely supported across chains

---

### [profile.production]
Strict settings for production deployment.

```toml
[profile.production]
optimizer_runs = 10000    # Maximum optimization for gas efficiency
warnings_as_errors = true # Fail on any warning
no_console = true         # Don't allow console.log in production
```

**Usage:**
```bash
forge build --profile production
```

---

### [rpc_endpoints]
This tells Foundry how to connect to different blockchains.

**For SolarSync:**
```toml
[rpc_endpoints]
# Local development
anvil = "http://127.0.0.1:8545"

# Hedera testnet
hedera_testnet = "https://testnet.hashio.io/api"

# Base testnet (recommended for MVP)
base_sepolia = "https://sepolia.base.org"

# Production
base_mainnet = "https://mainnet.base.org"
```

**Usage:**
```bash
# Deploy to Hedera testnet
forge script script/Deploy.s.sol --rpc-url hedera_testnet --broadcast

# Deploy to Base Sepolia
forge script script/Deploy.s.sol --rpc-url base_sepolia --broadcast

# Deploy to Base mainnet
forge script script/Deploy.s.sol --rpc-url base_mainnet --broadcast
```

---

### [etherscan]
For verifying contracts on blockchain explorers.

```toml
[etherscan]
base_sepolia = { key = "", chain = 84532 }
base_mainnet = { key = "", chain = 8453 }
sepolia = { key = "${ETHERSCAN_API_KEY}", chain = 11155111 }
```

**Why you need this:**
When you deploy a contract, users can't see your source code unless verified on the explorer. Etherscan's API automates this verification.

**Get API keys:**
- Base: No API key needed (uses Etherscan API)
- Sepolia: https://etherscan.io/apis
- Polygon: https://polygonscan.com/apis

---

### [remappings]
Shortcuts for imports to keep code clean.

```toml
remappings = [
    "@openzeppelin/contracts/=lib/openzeppelin-contracts/contracts/",
    "@chainlink/contracts/=lib/chainlink/contracts/",
]
```

**Before remapping:**
```solidity
import "../lib/openzeppelin-contracts/contracts/token/ERC20/ERC20.sol";
```

**After remapping:**
```solidity
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
```

---

## Common Forge Commands

### Build
```bash
# Build all contracts
forge build

# Build specific contract
forge build --skip-tests

# Build in production mode
forge build --profile production
```

### Test
```bash
# Run all tests
forge test

# Run tests for specific file
forge test --match-contract SolarSyncCoreTest

# Run tests with gas report
forge test --gas-report

# Run tests in verbose mode (shows console.log output)
forge test -vvv

# Fuzz tests with custom runs
forge test --fuzz-runs 1000
```

### Deploy
```bash
# Deploy locally (requires anvil running in another terminal)
anvil
# In another terminal:
forge script script/Deploy.s.sol --rpc-url anvil --broadcast --private-key 0xac0...

# Deploy to Hedera testnet
forge script script/Deploy.s.sol \
  --rpc-url hedera_testnet \
  --broadcast \
  --verify \
  --etherscan-api-key $ETHERSCAN_API_KEY

# Deploy with specific environment
NETWORK=base_sepolia forge script script/Deploy.s.sol --broadcast

# Simulate deployment without sending (dry run)
forge script script/Deploy.s.sol --rpc-url base_sepolia --sender 0x1234...
```

### Verify Contract
```bash
# After deployment
forge verify-contract \
  --chain base_sepolia \
  --constructor-args $(cast abi-encode "constructor()" "") \
  0xYourContractAddress \
  SolarSyncCore \
  --etherscan-api-key $ETHERSCAN_API_KEY
```

### Cast (CLI interaction)
```bash
# Get balance
cast balance 0xYourAddress --rpc-url base_sepolia

# Call a contract function
cast call 0xContractAddress "balanceOf(address)" 0xYourAddress --rpc-url base_sepolia

# Send a transaction
cast send 0xContractAddress "transfer(address,uint256)" 0xRecipient 1000000000000000000 \
  --rpc-url base_sepolia \
  --private-key $PRIVATE_KEY
```

---

## Setup for Each Network

### 1. Local Development (Anvil)
```bash
# Terminal 1: Start local node
anvil --accounts 10 --balance 100

# Terminal 2: Deploy
forge script script/Deploy.s.sol --rpc-url anvil --broadcast --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
```

**Anvil provides 10 pre-funded accounts. Use the private key above for first account.**

---

### 2. Hedera Testnet
**Prerequisites:**
1. Create Hedera wallet: https://wallet.hedera.com
2. Get testnet tokens from faucet: https://testnet.hashio.io
3. Note your account ID (0.0.xxxxx)

```bash
# In .env
HEDERA_ACCOUNT_ID=0.0.YOUR_ACCOUNT_ID
HEDERA_PRIVATE_KEY=your_hedera_private_key

# Deploy
forge script script/Deploy.s.sol \
  --rpc-url hedera_testnet \
  --broadcast \
  --private-key $HEDERA_PRIVATE_KEY
```

---

### 3. Base Sepolia (Recommended for MVP)
**Prerequisites:**
1. Get testnet ETH from faucet: https://www.alchemy.com/faucets/base-sepolia
2. Set ETHERSCAN_API_KEY: https://etherscan.io/apis

```bash
# Deploy and verify
forge script script/Deploy.s.sol \
  --rpc-url base_sepolia \
  --broadcast \
  --verify \
  --etherscan-api-key $ETHERSCAN_API_KEY \
  --private-key $PRIVATE_KEY
```

---

### 4. Base Mainnet (Production)
**⚠️ WARNING: Real money - test thoroughly first**

```bash
# Always test on sepolia first
# Verify contract behavior
# Check gas costs
# Use hardware wallet for mainnet

forge script script/Deploy.s.sol \
  --rpc-url base_mainnet \
  --broadcast \
  --verify \
  --private-key $PRIVATE_KEY
```

---

## Troubleshooting

### Problem: "Solidity version mismatch"
```bash
# Error: Pragma in contract is 0.8.19 but foundry.toml specifies 0.8.20
# Solution: Update foundry.toml or contract pragma to match
solc_version = "0.8.20"  # in foundry.toml
pragma solidity ^0.8.20; // in contract
```

### Problem: "RPC endpoint not working"
```bash
# Test RPC endpoint
cast call 0x1234... "balanceOf(address)" 0xYourAddress --rpc-url $YOUR_RPC

# If timeout, RPC is down. Try alternatives:
# Hedera: https://testnet.hashio.io/api
# Base: https://sepolia.base.org
# Infura: https://sepolia.infura.io/v3/YOUR_KEY
```

### Problem: "Gas estimation failed"
```bash
# Error: "out of gas" during deployment
# Solution: Increase gas limit or reduce contract complexity

# In Deploy.s.sol:
forge script script/Deploy.s.sol \
  --rpc-url base_sepolia \
  --broadcast \
  --gas-limit 10000000  # Increase from default 5M
```

### Problem: "Private key rejected"
```bash
# Error: Invalid private key format
# Solution: Private key should be:
# - 64 hex characters (32 bytes)
# - Start with 0x
# - Not include quotes in .env

# Correct format in .env:
PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80

# Test:
cast wallet address --private-key $PRIVATE_KEY
```

---

## Best Practices for SolarSync

### 1. Version Control
```bash
# .gitignore for contracts/
echo "
.env
out/
cache/
build/
node_modules/
.DS_Store
.idea/
*.swp
*.iml
" > .gitignore
```

### 2. Environment Secrets
**Never commit private keys!**
```bash
# Use .env (add to .gitignore)
PRIVATE_KEY=0x...

# Or use environment variables
export PRIVATE_KEY=0x...
forge script script/Deploy.s.sol --broadcast
```

### 3. Gas Optimization
```bash
# Check contract size
forge build --profile production
# Output shows contract sizes

# Optimize large contracts
# - Remove unnecessary state variables
# - Use packed structs
# - Implement proxy pattern for upgradeable contracts
```

### 4. Testing Strategy
```bash
# Unit tests
forge test --match-contract SolarSyncCoreTest

# Integration tests
forge test --match-contract IntegrationTest

# Fuzz tests (find bugs automatically)
forge test --match-contract FuzzTest --fuzz-runs 10000

# Gas benchmarks
forge test --gas-report
```

### 5. Staging Before Production
```bash
# 1. Test locally (anvil)
# 2. Deploy to testnet (Base Sepolia)
# 3. Verify contract on explorer
# 4. Run transaction tests
# 5. Monitor for 24 hours
# 6. Deploy to mainnet
```

---

## Example Deploy Script Structure

```solidity
// script/Deploy.s.sol
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/SolarSyncCore.sol";
import "../src/TradingEngine.sol";
import "../src/CarbonCredits.sol";

contract Deploy is Script {
    function run() public {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        
        vm.startBroadcast(deployerPrivateKey);
        
        // Deploy core contract
        SolarSyncCore core = new SolarSyncCore();
        console.log("SolarSyncCore deployed at:", address(core));
        
        // Deploy trading engine
        TradingEngine engine = new TradingEngine(address(core));
        console.log("TradingEngine deployed at:", address(engine));
        
        // Deploy carbon credits
        CarbonCredits carbon = new CarbonCredits(address(core));
        console.log("CarbonCredits deployed at:", address(carbon));
        
        // Setup roles and permissions
        core.grantRole(core.TRADER_ROLE(), address(engine));
        core.grantRole(core.ORACLE_ROLE(), address(0xOracleAddress));
        
        vm.stopBroadcast();
    }
}
```

---

## Next Steps

1. **Copy foundry.toml** to your `contracts/` directory
2. **Run `forge build`** to verify setup
3. **Create Deploy.s.sol** using example above
4. **Deploy to testnet** with: `forge script script/Deploy.s.sol --rpc-url base_sepolia --broadcast`
5. **Verify on explorer** to enable contract interaction from UI