// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "./ISolarSyncInterfaces.sol"; // Assuming interfaces are in a separate file;

/**
 * @title SolarSyncCore
 * @dev Main contract for managing producer/consumer registration and energy listings.
 * Handles logic for producers listing energy and consumers creating buy orders.
 */
contract SolarSyncCore {
    // --- STATE VARIABLES ---

    // Structs to hold participant data
    struct Producer {
        uint256 id;
        uint256 capacityKWh;
        string location; // Example: "lat,lng"
        uint256 listedEnergyCount;
        bool isRegistered;
    }

    struct Consumer {
        uint256 id;
        bool isRegistered;
    }

    // Struct to hold an energy listing
    struct EnergyListing {
        uint256 id;
        address payable producer; // Producer listing the energy
        uint256 amountKWh;
        uint256 pricePerKWhWei; // Price in Wei (smallest unit of Ether/Polygon currency)
        uint256 deliveryTimestamp;
        bool isSold;
    }

    // Mappings and counters
    mapping(address => Producer) public producers;
    mapping(address => Consumer) public consumers;
    mapping(uint256 => EnergyListing) public listings;

    uint256 private nextProducerId = 1;
    uint256 private nextConsumerId = 1;
    uint256 private nextListingId = 1;

    // Contract addresses for interaction
    IEnergyOracle public energyOracle;
    ITradingEngine public tradingEngine;

    // --- EVENTS ---

    event ProducerRegistered(address indexed producer, uint256 id, uint256 capacity);
    event ConsumerRegistered(address indexed consumer, uint256 id);
    event EnergyListed(uint256 indexed listingId, address indexed producer, uint256 amountKWh, uint256 pricePerKWhWei);
    event BuyOrderCreated(address indexed consumer, uint256 amountKWh, uint256 maxPricePerKWhWei);
    event GenerationConfirmed(uint256 indexed listingId, uint256 actualAmountKWh);

    // --- MODIFIERS ---

    modifier onlyRegisteredProducer() {
        require(producers[msg.sender].isRegistered, "Caller must be a registered producer.");
        _;
    }

    // --- CONSTRUCTOR & ORACLE/ENGINE SETUP ---

    constructor(address _oracleAddress, address _engineAddress) {
        // Set the addresses for the Oracle and Trading Engine contracts
        energyOracle = IEnergyOracle(_oracleAddress);
        tradingEngine = ITradingEngine(_engineAddress);
    }

    // --- PRODUCER FUNCTIONS ---

    /**
     * @notice Registers a new energy producer on the platform.
     * @param capacityKWh The maximum generating capacity of the producer (e.g., in kW).
     * @param location A string representing the location (e.g., "34.05,-118.24").
     */
    function registerProducer(uint256 capacityKWh, string memory location) external {
        require(!producers[msg.sender].isRegistered, "Producer already registered.");
        
        producers[msg.sender] = Producer({
            id: nextProducerId,
            capacityKWh: capacityKWh,
            location: location,
            listedEnergyCount: 0,
            isRegistered: true
        });
        
        emit ProducerRegistered(msg.sender, nextProducerId, capacityKWh);
        nextProducerId++;
    }

    /**
     * @notice Allows a producer to list energy for sale.
     * @dev This is typically done after receiving an ML prediction from the oracle.
     * @param amountKWh The amount of energy (in kWh) the producer wants to sell.
     * @param pricePerKWhWei The asking price per kWh in Wei.
     * @param deliveryTimestamp The Unix timestamp when the energy will be delivered.
     */
    function listEnergy(
        uint256 amountKWh, 
        uint256 pricePerKWhWei, 
        uint256 deliveryTimestamp
    ) external onlyRegisteredProducer {
        require(amountKWh > 0, "Amount must be greater than zero.");
        require(pricePerKWhWei > 0, "Price must be greater than zero.");
        require(deliveryTimestamp > block.timestamp, "Delivery time must be in the future.");

        uint256 listingId = nextListingId;
        
        listings[listingId] = EnergyListing({
            id: listingId,
            producer: payable(msg.sender),
            amountKWh: amountKWh,
            pricePerKWhWei: pricePerKWhWei,
            deliveryTimestamp: deliveryTimestamp,
            isSold: false
        });

        producers[msg.sender].listedEnergyCount++;
        nextListingId++;

        emit EnergyListed(listingId, msg.sender, amountKWh, pricePerKWhWei);
    }

    /**
     * @notice A producer confirms the actual amount of energy generated after the delivery time.
     * @dev This is crucial for reputation and final settlement.
     * @param listingId The ID of the energy listing.
     * @param actualAmountKWh The actual amount of energy generated (in kWh).
     */
    function confirmGeneration(uint256 listingId, uint256 actualAmountKWh) external onlyRegisteredProducer {
        EnergyListing storage listing = listings[listingId];
        
        require(listing.id == listingId, "Listing not found.");
        require(listing.producer == msg.sender, "Only the listing owner can confirm generation.");
        // A real system would also check that deliveryTimestamp has passed
        
        // Logic for final settlement and reputation update would go here, 
        // typically involving the TradingEngine and ReputationSystem contracts.

        emit GenerationConfirmed(listingId, actualAmountKWh);
    }
    
    // --- CONSUMER FUNCTIONS ---

    /**
     * @notice Registers a new energy consumer on the platform.
     */
    function registerConsumer() external {
        require(!consumers[msg.sender].isRegistered, "Consumer already registered.");

        consumers[msg.sender] = Consumer({
            id: nextConsumerId,
            isRegistered: true
        });

        emit ConsumerRegistered(msg.sender, nextConsumerId);
        nextConsumerId++;
    }

    /**
     * @notice Allows a consumer to create a buy order for energy.
     * @dev The actual order matching and execution happens in the TradingEngine.
     * @param amountKWh The amount of energy (in kWh) the consumer wants to buy.
     * @param maxPricePerKWhWei The maximum price per kWh the consumer is willing to pay.
     */
    function createBuyOrder(uint256 amountKWh, uint256 maxPricePerKWhWei) external payable {
        require(consumers[msg.sender].isRegistered, "Caller must be a registered consumer.");
        require(amountKWh > 0, "Amount must be greater than zero.");
        require(maxPricePerKWhWei > 0, "Max price must be greater than zero.");
        
        // Consumer sends the maximum possible funds needed to cover the order
        uint256 requiredDeposit = amountKWh * maxPricePerKWhWei;
        require(msg.value >= requiredDeposit, "Insufficient funds deposited for the order.");

        // The actual order matching logic is delegated to the TradingEngine
        (uint256 matchedListingId, uint256 matchedPrice) = tradingEngine.getBestMatch(
            amountKWh, 
            maxPricePerKWhWei, 
            msg.sender
        );

        if (matchedListingId != 0) {
            // In a full system, funds would be escrowed here and sent to the trading engine
            // for settlement once the generation is confirmed.
            // A refund of any excess deposit (msg.value - actual_cost) would also occur.
        }

        emit BuyOrderCreated(msg.sender, amountKWh, maxPricePerKWhWei);
    }
    
    // --- UTILITY FUNCTIONS ---
    
    /**
     * @notice Gets the AI-predicted energy generation for a producer.
     * @param producerAddress The address of the producer.
     * @return The predicted energy amount in KWh.
     */
    function getPredictedGeneration(address producerAddress) public view returns (uint256) {
        return energyOracle.getEnergyPrediction(producerAddress);
    }
    
    // Fallback function to prevent accidental Ether sends without a function call
    receive() external payable {
        revert("Ether transfer not allowed directly.");
    }
}