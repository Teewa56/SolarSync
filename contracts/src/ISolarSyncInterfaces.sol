// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ISolarSyncInterfaces
 * @dev Complete interfaces for all SolarSync contracts
 */

interface IEnergyOracle {
    /**
     * @notice Get the latest ML prediction for a producer
     * @param producerAddress The producer's address
     * @return predictedAmountKWh The predicted energy generation in KWh
     */

    function getEnergyPrediction(address producerAddress) external view returns (uint256 predictedAmountKWh);
    
    /**
     * @notice Update the prediction for a producer (admin or oracle callback)
     * @param producerAddress The producer's address
     * @param predictedAmountKWh The new predicted amount
     */

    function updatePrediction(address producerAddress, uint256 predictedAmountKWh) external;
    
    /**
     * @notice Request a new prediction from Chainlink oracle
     * @param producer The producer's address
     * @param lat Latitude of the facility
     * @param lng Longitude of the facility
     * @return requestId The Chainlink request ID
     */

    function requestEnergyPrediction(address producer, int256 lat, int256 lng) external returns (bytes32 requestId);
}

interface ITradingEngine {
    /**
     * @notice Match all active orders in the order book
     */
    function matchOrders() external;

    /**
     * @notice Execute trade settlement after delivery confirmation
     * @param tradeId The ID of the trade to settle
     */
    function executeTradeSettlement(uint256 tradeId) external;

    /**
     * @notice Get the best match for a buy order
     * @param amount The requested energy amount
     * @param maxPrice The maximum price buyer is willing to pay
     * @param buyer The address of the buyer
     * @return matchedSellOrderId The ID of the matched sell order
     * @return matchedPrice The final matched price
     */
    function getBestMatch(uint256 amount, uint256 maxPrice, address buyer) external view returns (uint256 matchedSellOrderId, uint256 matchedPrice);

    /**
     * @notice Place a buy order (called by SolarSyncCore)
     * @param consumer The consumer's address
     * @param amountKWh Amount of energy to buy
     * @param maxPricePerKWhWei Maximum price per KWh
     * @return orderId The created order ID
     */
    function placeBuyOrder(address consumer, uint256 amountKWh, uint256 maxPricePerKWhWei
    ) external payable returns (uint256 orderId);
    
    /**
     * @notice Place a sell order (called by SolarSyncCore)
     * @param producer The producer's address
     * @param amountKWh Amount of energy to sell
     * @param pricePerKWhWei Price per KWh
     * @param deliveryTimestamp Delivery time
     * @return orderId The created order ID
     */
    function placeSellOrder(
        address producer, 
        uint256 amountKWh, 
        uint256 pricePerKWhWei,
        uint256 deliveryTimestamp
    ) external returns (uint256 orderId);

    /**
     * @notice Set the core contract address
     * @param _coreAddress The SolarSyncCore contract address
     */
    function setCoreContractAddress(address _coreAddress) external;

    /**
     * @notice Set supporting contract addresses
     * @param _reputationSystem ReputationSystem contract address
     * @param _carbonCredits CarbonCredits contract address
     */
    function setContractAddresses(
        address _reputationSystem,
        address _carbonCredits
    ) external;
}

// ============================================================================
// REPUTATION SYSTEM INTERFACE
// ============================================================================

interface IReputationSystem {
    /**
     * @notice Get the reputation score for a participant
     * @param participant The participant's address
     * @return The reputation score (0-1000)
     */
    function getReputationScore(address participant) 
        external 
        view 
        returns (uint256);
    
    /**
     * @notice Update producer's score based on delivery accuracy
     * @param producer The producer's address
     * @param promisedKWh The promised amount
     * @param actualKWh The actual delivered amount
     */
    function updateProducerScore(
        address producer, 
        uint256 promisedKWh, 
        uint256 actualKWh
    ) external;
    
    /**
     * @notice Update consumer's score based on payment behavior
     * @param consumer The consumer's address
     * @param wasSuccessful Whether the trade was successful
     */
    function updateConsumerScore(address consumer, bool wasSuccessful) 
        external;
    
    /**
     * @notice Initialize a participant's score
     * @param participant The participant's address
     */
    function initializeScore(address participant) external;

    /**
     * @notice Set the trading engine address
     * @param newEngineAddress The TradingEngine contract address
     */
    function setTradingEngineAddress(address newEngineAddress) external;

    /**
     * @notice Get detailed statistics for a participant
     * @param participant The participant's address
     */
    function getParticipantStats(address participant) 
        external 
        view 
        returns (
            uint256 totalTrades,
            uint256 successfulTrades,
            uint256 failedTrades,
            uint256 totalEnergyPromised,
            uint256 totalEnergyDelivered,
            uint256 lastUpdateTimestamp
        );

    /**
     * @notice Get reliability percentage
     * @param participant The participant's address
     * @return reliabilityPercentage The success rate (0-100)
     */
    function getReliabilityPercentage(address participant) 
        external 
        view 
        returns (uint256 reliabilityPercentage);
}

// ============================================================================
// CARBON CREDITS INTERFACE
// ============================================================================

interface ICarbonCredits {
    /**
     * @notice Calculate and mint carbon credits based on delivered energy
     * @param recipient The address to receive the credits
     * @param energyAmountKWh The verified amount of renewable energy delivered
     * @return creditsMinted The number of credits minted
     */
    function calculateAndMintCredits(
        address recipient, 
        uint256 energyAmountKWh
    ) external returns (uint256 creditsMinted);
    
    /**
     * @notice Get the balance of carbon credits for an address
     * @param account The account to check
     * @return The balance of carbon credits
     */
    function balanceOf(address account) external view returns (uint256);
    
    /**
     * @notice Transfer carbon credits to another address
     * @param to The recipient address
     * @param amount The amount to transfer
     * @return success Whether the transfer succeeded
     */
    function transfer(address to, uint256 amount) external returns (bool success);

    /**
     * @notice Set the core contract address
     * @param newCoreAddress The SolarSyncCore contract address
     */
    function setCoreContractAddress(address newCoreAddress) external;

    /**
     * @notice Get total verified energy for an address
     * @param account The address to check
     * @return Total KWh verified for this address
     */
    function getEnergyVerified(address account) external view returns (uint256);

    /**
     * @notice Get platform-wide statistics
     */
    function getPlatformStats() external view returns (
        uint256 totalEnergyVerified,
        uint256 totalCreditsIssued,
        uint256 totalSupply,
        uint256 maxSupply
    );
}

// ============================================================================
// SOLARSYNC CORE INTERFACE (for external contracts to call Core)
// ============================================================================

interface ISolarSyncCore {
    // Struct definitions
    struct Producer {
        uint256 id;
        uint256 capacityKWh;
        string location;
        uint256 listedEnergyCount;
        bool isRegistered;
        bool isActive;
        uint256 totalEnergyListed;
        uint256 totalEnergyDelivered;
    }

    struct Consumer {
        uint256 id;
        bool isRegistered;
        bool isActive;
        uint256 totalEnergyPurchased;
        uint256 totalOrdersPlaced;
    }

    struct EnergyListing {
        uint256 id;
        address producer;
        uint256 amountKWh;
        uint256 pricePerKWhWei;
        uint256 deliveryTimestamp;
        uint256 createdAt;
        bool isSold;
        bool isActive;
        uint256 actualDelivered;
        bool isConfirmed;
    }

    /**
     * @notice Register as a producer
     */
    function registerProducer(uint256 capacityKWh, string memory location) external;

    /**
     * @notice Register as a consumer
     */
    function registerConsumer() external;

    /**
     * @notice List energy for sale
     */
    function listEnergy(
        uint256 amountKWh, 
        uint256 pricePerKWhWei, 
        uint256 deliveryTimestamp
    ) external;

    /**
     * @notice Create a buy order
     */
    function createBuyOrder(
        uint256 amountKWh, 
        uint256 maxPricePerKWhWei
    ) external payable;

    /**
     * @notice Confirm energy generation
     */
    function confirmGeneration(
        uint256 listingId, 
        uint256 actualAmountKWh
    ) external;

    /**
     * @notice Get producer details
     */
    function getProducer(address producerAddress) 
        external 
        view 
        returns (Producer memory);

    /**
     * @notice Get consumer details
     */
    function getConsumer(address consumerAddress) 
        external 
        view 
        returns (Consumer memory);

    /**
     * @notice Get listing details
     */
    function getListing(uint256 listingId) 
        external 
        view 
        returns (EnergyListing memory);

    /**
     * @notice Get predicted generation for a producer
     */
    function getPredictedGeneration(address producerAddress) 
        external 
        view 
        returns (uint256);

    /**
     * @notice Set contract addresses
     */
    function setContractAddresses(
        address _carbonCreditsContract,
        address _reputationContract
    ) external;
}