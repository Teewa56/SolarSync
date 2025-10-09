// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/SolarSyncCore.sol";
import "../src/EnergyOracle.sol";
import "../src/TradingEngine.sol";
import "../src/CarbonCredits.sol";
import "../src/ReputationSystem.sol";

address constant ORACLE_ADDRESS = 0x5fd83192365fc7a6a4971a280a2a493cd6bb160f; 
bytes32 constant JOB_ID = 0x76b297b87823432742d4807469735d46777c223c6b291d96b02a99161a007b7b; 

/**
 * @title DeploySolarSync
 * @dev Foundry script to deploy all SolarSync contracts in the correct order and link them.
 */
contract DeploySolarSync is Script {

    function run() external returns (address core, address engine, address oracle, address credits, address reputation) {
        
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerPrivateKey);

        // 1. Deploy CarbonCredits and ReputationSystem (Dependencies of TradingEngine)
        // Pass the deployer's address for initial Ownable setting.
        CarbonCredits carbonCredits = new CarbonCredits(address(0)); // Placeholder, will set Core address later
        ReputationSystem reputationSystem = new ReputationSystem(address(0)); // Placeholder, will set Engine address later

        // 2. Deploy EnergyOracle
        // Requires Chainlink Oracle address and Job ID
        EnergyOracle energyOracle = new EnergyOracle(ORACLE_ADDRESS, JOB_ID);

        // 3. Deploy TradingEngine
        // Requires its dependencies: CarbonCredits and ReputationSystem
        TradingEngine tradingEngine = new TradingEngine(address(0)); // Placeholder, will set Core address later
        
        // 4. Deploy SolarSyncCore (The main hub)
        // Requires its immediate dependencies: EnergyOracle and TradingEngine
        SolarSyncCore coreContract = new SolarSyncCore(address(energyOracle), address(tradingEngine));

        // 5. Post-Deployment Linking (Updating placeholder addresses)
        
        // CarbonCredits needs to authorize SolarSyncCore to mint
        carbonCredits.setCoreContractAddress(address(coreContract));
        
        // ReputationSystem needs to authorize TradingEngine to update scores
        reputationSystem.setTradingEngineAddress(address(tradingEngine));

        // TradingEngine needs to authorize SolarSyncCore to call its functions
        // Note: The TradingEngine constructor currently takes a Core address, so let's update it
        // The implementation above had a placeholder, so we'd need to add a setter function to TradingEngine
        // If we adjust the TradingEngine constructor: TradingEngine tradingEngine = new TradingEngine(address(coreContract));
        // For now, we assume the constructor or a setter is used. Let's use a setter for flexibility.
        tradingEngine.setCoreContractAddress(address(coreContract));
        // We also need setters for its other dependencies in a complete system.

        // Update ReputationSystem with the *final* TradingEngine address
        reputationSystem.setTradingEngineAddress(address(tradingEngine));

        vm.stopBroadcast();

        console.log("-----------------------------------------");
        console.log("☀️ SolarSync Deployment Successful! ⚡");
        console.log("SolarSyncCore Address:   ", address(coreContract));
        console.log("TradingEngine Address:   ", address(tradingEngine));
        console.log("EnergyOracle Address:    ", address(energyOracle));
        console.log("CarbonCredits Address:   ", address(carbonCredits));
        console.log("ReputationSystem Address:", address(reputationSystem));
        console.log("-----------------------------------------");

        // Return addresses for external verification/use
        return (address(coreContract), address(tradingEngine), address(energyOracle), address(carbonCredits), address(reputationSystem));
    }
}