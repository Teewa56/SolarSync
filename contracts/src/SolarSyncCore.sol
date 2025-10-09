// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./ISolarSyncInterfaces.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title SolarSyncCore
 * @dev Main contract for managing producer/consumer registration and energy listings.
 * Handles logic for producers listing energy and consumers creating buy orders.
 */
contract SolarSyncCore is ReentrancyGuard, Pausable, Ownable {

    struct Producer {
        uint256 id;
        uint256 capacityKWh;
        string location; // Format: "lat,lng"
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
        address payable producer;
        uint256 amountKWh;
        uint256 pricePerKWhWei;
        uint256 deliveryTimestamp;
        uint256 createdAt;
        bool isSold;
        bool isActive;
        uint256 actualDelivered;
        bool isConfirmed;
    }

    // Mappings and counters
    mapping(address => Producer) public producers;
    mapping(address => Consumer) public consumers;
    mapping(uint256 => EnergyListing) public listings;
    mapping(address => uint256[]) public producerListings; // Track listings by producer
    mapping(address => uint256[]) public consumerOrders; // Track orders by consumer

    uint256 private nextProducerId = 1;
    uint256 private nextConsumerId = 1;
    uint256 private nextListingId = 1;

    // Platform configuration
    uint256 public constant MINIMUM_LISTING_AMOUNT = 1; // Minimum 1 KWh
    uint256 public constant MAXIMUM_LISTING_AMOUNT = 1000000; // Maximum 1 GWh
    uint256 public constant MINIMUM_PRICE = 1; // Minimum 1 Wei per KWh
    uint256 public constant MAXIMUM_DELIVERY_WINDOW = 7 days; // Maximum 7 days in future
    uint256 public platformFeePercentage = 50; // 0.5% (50 basis points)

    // Contract addresses for interaction
    IEnergyOracle public energyOracle;
    ITradingEngine public tradingEngine;
    address public carbonCreditsContract;
    address public reputationContract;

    // --- EVENTS ---

    event ProducerRegistered(
        address indexed producer, 
        uint256 indexed id, 
        uint256 capacity,
        string location
    );
    
    event ConsumerRegistered(
        address indexed consumer, 
        uint256 indexed id
    );
    
    event EnergyListed(
        uint256 indexed listingId, 
        address indexed producer, 
        uint256 amountKWh, 
        uint256 pricePerKWhWei,
        uint256 deliveryTimestamp
    );
    
    event BuyOrderCreated(
        address indexed consumer, 
        uint256 amountKWh, 
        uint256 maxPricePerKWhWei,
        uint256 deposit
    );
    
    event GenerationConfirmed(
        uint256 indexed listingId, 
        uint256 actualAmountKWh,
        address indexed producer
    );

    event ListingCancelled(
        uint256 indexed listingId,
        address indexed producer,
        string reason
    );

    event ProducerStatusChanged(
        address indexed producer,
        bool isActive,
        string reason
    );

    // --- MODIFIERS ---

    modifier onlyRegisteredProducer() {
        require(producers[msg.sender].isRegistered, "Caller must be a registered producer");
        require(producers[msg.sender].isActive, "Producer account is inactive");
        _;
    }

    modifier onlyRegisteredConsumer() {
        require(consumers[msg.sender].isRegistered, "Caller must be a registered consumer");
        require(consumers[msg.sender].isActive, "Consumer account is inactive");
        _;
    }

    modifier validListingId(uint256 listingId) {
        require(listingId > 0 && listingId < nextListingId, "Invalid listing ID");
        _;
    }

    // --- CONSTRUCTOR & SETUP ---

    constructor(address _oracleAddress, address _engineAddress) Ownable(msg.sender) {
        require(_oracleAddress != address(0), "Oracle address cannot be zero");
        require(_engineAddress != address(0), "Engine address cannot be zero");
        
        energyOracle = IEnergyOracle(_oracleAddress);
        tradingEngine = ITradingEngine(_engineAddress);
    }

    // --- ADMIN FUNCTIONS ---

    function setContractAddresses(
        address _carbonCreditsContract,
        address _reputationContract
    ) external onlyOwner {
        carbonCreditsContract = _carbonCreditsContract;
        reputationContract = _reputationContract;
    }

    function setPlatformFee(uint256 _feePercentage) external onlyOwner {
        require(_feePercentage <= 500, "Fee cannot exceed 5%"); // Max 5%
        platformFeePercentage = _feePercentage;
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    // --- PRODUCER FUNCTIONS ---

    /**
     * @notice Registers a new energy producer on the platform
     * @param capacityKWh The maximum generating capacity of the producer
     * @param location A string representing the location ("lat,lng")
     */
    function registerProducer(
        uint256 capacityKWh, 
        string memory location
    ) external whenNotPaused {
        require(!producers[msg.sender].isRegistered, "Producer already registered");
        require(capacityKWh > 0, "Capacity must be greater than zero");
        require(bytes(location).length > 0, "Location cannot be empty");
        
        producers[msg.sender] = Producer({
            id: nextProducerId,
            capacityKWh: capacityKWh,
            location: location,
            listedEnergyCount: 0,
            isRegistered: true,
            isActive: true,
            totalEnergyListed: 0,
            totalEnergyDelivered: 0
        });
        
        emit ProducerRegistered(msg.sender, nextProducerId, capacityKWh, location);
        nextProducerId++;
    }

    /**
     * @notice Allows a producer to list energy for sale
     * @param amountKWh The amount of energy (in kWh) to sell
     * @param pricePerKWhWei The asking price per kWh in Wei
     * @param deliveryTimestamp The Unix timestamp when energy will be delivered
     */
    function listEnergy(
        uint256 amountKWh, 
        uint256 pricePerKWhWei, 
        uint256 deliveryTimestamp
    ) external onlyRegisteredProducer whenNotPaused nonReentrant {
        require(amountKWh >= MINIMUM_LISTING_AMOUNT, "Amount too small");
        require(amountKWh <= MAXIMUM_LISTING_AMOUNT, "Amount too large");
        require(pricePerKWhWei >= MINIMUM_PRICE, "Price too low");
        require(deliveryTimestamp > block.timestamp, "Delivery time must be in the future");
        require(deliveryTimestamp <= block.timestamp + MAXIMUM_DELIVERY_WINDOW, 
                "Delivery time too far in future");

        uint256 listingId = nextListingId;
        
        listings[listingId] = EnergyListing({
            id: listingId,
            producer: payable(msg.sender),
            amountKWh: amountKWh,
            pricePerKWhWei: pricePerKWhWei,
            deliveryTimestamp: deliveryTimestamp,
            createdAt: block.timestamp,
            isSold: false,
            isActive: true,
            actualDelivered: 0,
            isConfirmed: false
        });

        producers[msg.sender].listedEnergyCount++;
        producers[msg.sender].totalEnergyListed += amountKWh;
        producerListings[msg.sender].push(listingId);
        nextListingId++;

        emit EnergyListed(listingId, msg.sender, amountKWh, pricePerKWhWei, deliveryTimestamp);
    }

    /**
     * @notice Producer confirms actual energy generation after delivery time
     * @param listingId The ID of the energy listing
     * @param actualAmountKWh The actual amount of energy generated
     */
    function confirmGeneration(
        uint256 listingId, 
        uint256 actualAmountKWh
    ) external validListingId(listingId) nonReentrant {
        EnergyListing storage listing = listings[listingId];
        
        require(listing.producer == msg.sender, "Only listing owner can confirm");
        require(block.timestamp >= listing.deliveryTimestamp, "Delivery time not reached");
        require(!listing.isConfirmed, "Generation already confirmed");
        require(actualAmountKWh <= listing.amountKWh * 12 / 10, "Actual cannot exceed 120% of promised");
        
        listing.actualDelivered = actualAmountKWh;
        listing.isConfirmed = true;
        producers[msg.sender].totalEnergyDelivered += actualAmountKWh;
        //it is called without properly checking if the listing was sold
        //because if it was not sold, the trading engine will simply do nothing
        
        // If there was a trade, trigger settlement through trading engine
        if (listing.isSold) {
            tradingEngine.executeTradeSettlement(listingId);
        }

        emit GenerationConfirmed(listingId, actualAmountKWh, msg.sender);
    }

    /**
     * @notice Cancel an active listing (only if not sold)
     * @param listingId The ID of the listing to cancel
     * @param reason Reason for cancellation
     */
    function cancelListing(
        uint256 listingId, 
        string memory reason
    ) external validListingId(listingId) {
        EnergyListing storage listing = listings[listingId];
        
        require(listing.producer == msg.sender, "Only listing owner can cancel");
        require(listing.isActive, "Listing already inactive");
        require(!listing.isSold, "Cannot cancel sold listing");
        
        listing.isActive = false;
        producers[msg.sender].listedEnergyCount--;
        
        emit ListingCancelled(listingId, msg.sender, reason);
    }
    
    // --- CONSUMER FUNCTIONS ---

    /**
     * @notice Registers a new energy consumer on the platform
     */
    function registerConsumer() external whenNotPaused {
        require(!consumers[msg.sender].isRegistered, "Consumer already registered");

        consumers[msg.sender] = Consumer({
            id: nextConsumerId,
            isRegistered: true,
            isActive: true,
            totalEnergyPurchased: 0,
            totalOrdersPlaced: 0
        });

        emit ConsumerRegistered(msg.sender, nextConsumerId);
        nextConsumerId++;
    }

    /**
     * @notice Allows a consumer to create a buy order for energy
     * @param amountKWh The amount of energy to buy
     * @param maxPricePerKWhWei The maximum price per kWh willing to pay
     */
    function createBuyOrder(
        uint256 amountKWh, 
        uint256 maxPricePerKWhWei
    ) external payable onlyRegisteredConsumer whenNotPaused nonReentrant {
        require(amountKWh >= MINIMUM_LISTING_AMOUNT, "Amount too small");
        require(amountKWh <= MAXIMUM_LISTING_AMOUNT, "Amount too large");
        require(maxPricePerKWhWei >= MINIMUM_PRICE, "Max price too low");
        
        uint256 requiredDeposit = amountKWh * maxPricePerKWhWei;
        require(msg.value >= requiredDeposit, "Insufficient deposit");

        consumers[msg.sender].totalOrdersPlaced++;

        // Delegate order matching to trading engine
        uint256 orderId = tradingEngine.placeBuyOrder{value: msg.value}(
            msg.sender, 
            amountKWh, 
            maxPricePerKWhWei
        );

        consumerOrders[msg.sender].push(orderId);

        emit BuyOrderCreated(msg.sender, amountKWh, maxPricePerKWhWei, msg.value);
    }
    
    // --- VIEW FUNCTIONS ---
    
    /**
     * @notice Gets AI-predicted energy generation for a producer
     * @param producerAddress The address of the producer
     * @return The predicted energy amount in KWh
     */
    function getPredictedGeneration(address producerAddress) external view returns (uint256) {
        return energyOracle.getEnergyPrediction(producerAddress);
    }

    /**
     * @notice Get producer details
     * @param producerAddress The producer's address
     * @return Producer struct data
     */
    function getProducer(address producerAddress) external view returns (Producer memory) {
        return producers[producerAddress];
    }

    /**
     * @notice Get consumer details
     * @param consumerAddress The consumer's address
     * @return Consumer struct data
     */
    function getConsumer(address consumerAddress) external view returns (Consumer memory) {
        return consumers[consumerAddress];
    }

    /**
     * @notice Get listing details
     * @param listingId The listing ID
     * @return EnergyListing struct data
     */
    function getListing(uint256 listingId) external view validListingId(listingId) returns (EnergyListing memory) {
        return listings[listingId];
    }

    /**
     * @notice Get all listings by a producer
     * @param producerAddress The producer's address
     * @return Array of listing IDs
     */
    function getProducerListings(address producerAddress) external view returns (uint256[] memory) {
        return producerListings[producerAddress];
    }

    /**
     * @notice Get all orders by a consumer
     * @param consumerAddress The consumer's address
     * @return Array of order IDs
     */
    function getConsumerOrders(address consumerAddress) external view returns (uint256[] memory) {
        return consumerOrders[consumerAddress];
    }

    /**
     * @notice Get active listings count
     * @return Number of active listings
     */
    function getActiveListingsCount() external view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 1; i < nextListingId; i++) {
            if (listings[i].isActive && !listings[i].isSold) {
                count++;
            }
        }
        return count;
    }

    /**
     * @notice Get platform statistics
     * @return totalProducers, totalConsumers, totalListings, totalVolume
     */
    function getPlatformStats() external view returns (
        uint256 totalProducers,
        uint256 totalConsumers, 
        uint256 totalListings,
        uint256 totalVolume
    ) {
        totalProducers = nextProducerId - 1;
        totalConsumers = nextConsumerId - 1;
        totalListings = nextListingId - 1;
        
        // Calculate total volume listed
        for (uint256 i = 1; i < nextListingId; i++) {
            totalVolume += listings[i].amountKWh;
        }
    }
    
    // --- EMERGENCY FUNCTIONS ---

    /**
     * @notice Emergency function to deactivate a producer (only owner)
     * @param producerAddress The producer to deactivate
     * @param reason Reason for deactivation
     */
    function deactivateProducer(
        address producerAddress, 
        string memory reason
    ) external onlyOwner {
        require(producers[producerAddress].isRegistered, "Producer not registered");
        producers[producerAddress].isActive = false;
        
        emit ProducerStatusChanged(producerAddress, false, reason);
    }

    /**
     * @notice Reactivate a producer (only owner)
     * @param producerAddress The producer to reactivate
     * @param reason Reason for reactivation
     */
    function reactivateProducer(
        address producerAddress, 
        string memory reason
    ) external onlyOwner {
        require(producers[producerAddress].isRegistered, "Producer not registered");
        producers[producerAddress].isActive = true;
        
        emit ProducerStatusChanged(producerAddress, true, reason);
    }

    // --- FALLBACK ---

    receive() external payable {
        revert("Direct Ether transfers not allowed");
    }

    fallback() external payable {
        revert("Function not found");
    }
}