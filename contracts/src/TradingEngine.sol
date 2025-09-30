// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title TradingEngine
 * @dev Handles order matching, escrow, and automated settlement between producers and consumers
 */
contract TradingEngine is ReentrancyGuard, Pausable, Ownable {
    
    struct BuyOrder {
        uint256 id;
        address consumer;
        uint256 amountKWh;
        uint256 maxPricePerKWhWei;
        uint256 escrowedFunds;
        uint256 createdAt;
        bool isActive;
        bool isPartiallyFilled;
        uint256 filledAmount;
    }
    
    struct SellOrder {
        uint256 id;
        address producer;
        uint256 amountKWh;
        uint256 pricePerKWhWei;
        uint256 deliveryTimestamp;
        uint256 createdAt;
        bool isActive;
        bool isPartiallyFilled;
        uint256 filledAmount;
    }
    
    struct Trade {
        uint256 id;
        uint256 buyOrderId;
        uint256 sellOrderId;
        uint256 matchedAmountKWh;
        uint256 finalPricePerKWhWei;
        uint256 escrowedAmount;
        uint256 executedAt;
        bool isSettled;
        bool isCompleted;
        uint256 actualDelivered;
        uint256 settledAt;
    }

    mapping(uint256 => BuyOrder) public buyOrders;
    mapping(uint256 => SellOrder) public sellOrders;
    mapping(uint256 => Trade) public trades;
    uint256[] public activeBuyOrders;
    uint256[] public activeSellOrders;
    mapping(address => uint256[]) public userBuyOrders;
    mapping(address => uint256[]) public userSellOrders;
    mapping(address => uint256[]) public userTrades;

    uint256 private nextBuyOrderId = 1;
    uint256 private nextSellOrderId = 1;
    uint256 private nextTradeId = 1;

    address public solarSyncCoreAddress;
    address public reputationSystemAddress;
    address public carbonCreditsAddress;
    
    uint256 public platformFeePercentage = 50; // 0.5% (50 basis points)
    uint256 public constant MAX_PLATFORM_FEE = 500; // Maximum 5%
    address public feeRecipient;
    
    uint256 public constant MIN_TRADE_AMOUNT = 1; // 1 KWh
    uint256 public constant MAX_TRADE_AMOUNT = 1000000; // 1 GWh
    uint256 public constant SETTLEMENT_WINDOW = 48 hours; // Time to confirm delivery

    event BuyOrderPlaced(
        uint256 indexed orderId,
        address indexed consumer,
        uint256 amountKWh,
        uint256 maxPrice,
        uint256 escrowAmount
    );
    
    event SellOrderPlaced(
        uint256 indexed orderId,
        address indexed producer,
        uint256 amountKWh,
        uint256 price,
        uint256 deliveryTime
    );
    
    event TradeExecuted(
        uint256 indexed tradeId,
        uint256 indexed buyOrderId,
        uint256 indexed sellOrderId,
        uint256 amountKWh,
        uint256 price,
        address buyer,
        address seller
    );
    
    event TradeSettled(
        uint256 indexed tradeId,
        uint256 actualDelivered,
        uint256 paymentAmount,
        uint256 refundAmount
    );
    
    event OrderCancelled(
        uint256 indexed orderId,
        address indexed user,
        bool isBuyOrder,
        string reason
    );

    event FundsWithdrawn(
        address indexed user,
        uint256 amount,
        string reason
    );
    
    modifier onlyCoreContract() {
        require(msg.sender == solarSyncCoreAddress, "Only SolarSync Core can call this");
        _;
    }

    modifier validBuyOrder(uint256 orderId) {
        require(orderId > 0 && orderId < nextBuyOrderId, "Invalid buy order ID");
        _;
    }

    modifier validSellOrder(uint256 orderId) {
        require(orderId > 0 && orderId < nextSellOrderId, "Invalid sell order ID");
        _;
    }

    modifier validTrade(uint256 tradeId) {
        require(tradeId > 0 && tradeId < nextTradeId, "Invalid trade ID");
        _;
    }

    constructor(address _coreAddress) Ownable(msg.sender) {
        require(_coreAddress != address(0), "Core address cannot be zero");
        solarSyncCoreAddress = _coreAddress;
        feeRecipient = msg.sender;
    }

    function setContractAddresses(
        address _reputationSystem,
        address _carbonCredits
    ) external onlyOwner {
        reputationSystemAddress = _reputationSystem;
        carbonCreditsAddress = _carbonCredits;
    }

    function setCoreContractAddress(address _coreAddress) external onlyOwner {
        require(_coreAddress != address(0), "Core address cannot be zero");
        solarSyncCoreAddress = _coreAddress;
    }

    function setPlatformFee(uint256 _feePercentage) external onlyOwner {
        require(_feePercentage <= MAX_PLATFORM_FEE, "Fee too high");
        platformFeePercentage = _feePercentage;
    }

    function setFeeRecipient(address _feeRecipient) external onlyOwner {
        require(_feeRecipient != address(0), "Fee recipient cannot be zero");
        feeRecipient = _feeRecipient;
    }

    /**
     * @notice Places a consumer buy order and escrows funds
     * @param consumer The address of the consumer
     * @param amountKWh Amount of energy to buy
     * @param maxPricePerKWhWei Maximum price willing to pay
     * @return buyId The ID of the created buy order
     */
    function placeBuyOrder(
        address consumer, 
        uint256 amountKWh, 
        uint256 maxPricePerKWhWei
    ) external payable onlyCoreContract whenNotPaused nonReentrant returns (uint256 buyId) {
        require(amountKWh >= MIN_TRADE_AMOUNT && amountKWh <= MAX_TRADE_AMOUNT, "Invalid amount");
        require(maxPricePerKWhWei > 0, "Price must be positive");
        
        uint256 requiredDeposit = amountKWh * maxPricePerKWhWei;
        require(msg.value >= requiredDeposit, "Insufficient escrow");
        
        buyId = nextBuyOrderId++;
        
        buyOrders[buyId] = BuyOrder({
            id: buyId,
            consumer: consumer,
            amountKWh: amountKWh,
            maxPricePerKWhWei: maxPricePerKWhWei,
            escrowedFunds: msg.value,
            createdAt: block.timestamp,
            isActive: true,
            isPartiallyFilled: false,
            filledAmount: 0
        });
        
        activeBuyOrders.push(buyId);
        userBuyOrders[consumer].push(buyId);
        
        emit BuyOrderPlaced(buyId, consumer, amountKWh, maxPricePerKWhWei, msg.value);
        _matchOrders();
        
        return buyId;
    }
    
    /**
     * @notice Places a producer sell order
     * @param producer The address of the producer
     * @param amountKWh Amount of energy to sell
     * @param pricePerKWhWei Price per KWh
     * @param deliveryTimestamp When energy will be delivered
     * @return sellId The ID of the created sell order
     */
    function placeSellOrder(
        address producer, 
        uint256 amountKWh, 
        uint256 pricePerKWhWei, 
        uint256 deliveryTimestamp
    ) external onlyCoreContract whenNotPaused returns (uint256 sellId) {
        require(amountKWh >= MIN_TRADE_AMOUNT && amountKWh <= MAX_TRADE_AMOUNT, "Invalid amount");
        require(pricePerKWhWei > 0, "Price must be positive");
        require(deliveryTimestamp > block.timestamp, "Invalid delivery time");
        
        sellId = nextSellOrderId++;
        
        sellOrders[sellId] = SellOrder({
            id: sellId,
            producer: producer,
            amountKWh: amountKWh,
            pricePerKWhWei: pricePerKWhWei,
            deliveryTimestamp: deliveryTimestamp,
            createdAt: block.timestamp,
            isActive: true,
            isPartiallyFilled: false,
            filledAmount: 0
        });
        
        activeSellOrders.push(sellId);
        userSellOrders[producer].push(sellId);
        
        emit SellOrderPlaced(sellId, producer, amountKWh, pricePerKWhWei, deliveryTimestamp);
        
        // Attempt immediate matching
        _matchOrders();
        
        return sellId;
    }

    /**
     * @notice Internal function to match buy and sell orders
     */
    function _matchOrders() internal {
        // Simple matching algorithm: match highest buy price with lowest sell price
        for (uint256 i = 0; i < activeBuyOrders.length; i++) {
            uint256 buyId = activeBuyOrders[i];
            BuyOrder storage buyOrder = buyOrders[buyId];
            
            if (!buyOrder.isActive) continue;
            
            for (uint256 j = 0; j < activeSellOrders.length; j++) {
                uint256 sellId = activeSellOrders[j];
                SellOrder storage sellOrder = sellOrders[sellId];
                
                if (!sellOrder.isActive) continue;
                
                // Check if orders can match
                if (buyOrder.maxPricePerKWhWei >= sellOrder.pricePerKWhWei) {
                    uint256 matchAmount = _min(
                        buyOrder.amountKWh - buyOrder.filledAmount,
                        sellOrder.amountKWh - sellOrder.filledAmount
                    );
                    
                    if (matchAmount > 0) {
                        _executeTrade(buyId, sellId, matchAmount, sellOrder.pricePerKWhWei);
                    }
                }
            }
        }
        _cleanupInactiveOrders();
    }

    /**
     * @notice Execute a matched trade
     */
    function _executeTrade(
        uint256 buyId, 
        uint256 sellId, 
        uint256 matchedAmount,
        uint256 finalPrice
    ) internal {
        BuyOrder storage buyOrder = buyOrders[buyId];
        SellOrder storage sellOrder = sellOrders[sellId];
        
        uint256 tradeId = nextTradeId++;
        uint256 totalCost = matchedAmount * finalPrice;
        
        buyOrder.filledAmount += matchedAmount;
        sellOrder.filledAmount += matchedAmount;
        
        if (buyOrder.filledAmount >= buyOrder.amountKWh) {
            buyOrder.isActive = false;
        } else {
            buyOrder.isPartiallyFilled = true;
        }
        
        if (sellOrder.filledAmount >= sellOrder.amountKWh) {
            sellOrder.isActive = false;
        } else {
            sellOrder.isPartiallyFilled = true;
        }
        
        trades[tradeId] = Trade({
            id: tradeId,
            buyOrderId: buyId,
            sellOrderId: sellId,
            matchedAmountKWh: matchedAmount,
            finalPricePerKWhWei: finalPrice,
            escrowedAmount: totalCost,
            executedAt: block.timestamp,
            isSettled: false,
            isCompleted: false,
            actualDelivered: 0,
            settledAt: 0
        });
        
        userTrades[buyOrder.consumer].push(tradeId);
        userTrades[sellOrder.producer].push(tradeId);
        
        uint256 excessFunds = buyOrder.escrowedFunds - totalCost;
        if (excessFunds > 0 && buyOrder.filledAmount >= buyOrder.amountKWh) {
            buyOrder.escrowedFunds = totalCost;
            (bool success, ) = buyOrder.consumer.call{value: excessFunds}("");
            require(success, "Refund failed");
        }
        
        emit TradeExecuted(
            tradeId,
            buyId,
            sellId,
            matchedAmount,
            finalPrice,
            buyOrder.consumer,
            sellOrder.producer
        );
    }

    /**
     * @notice Clean up inactive orders from active arrays
     */
    function _cleanupInactiveOrders() internal {
        for (uint256 i = activeBuyOrders.length; i > 0; i--) {
            if (!buyOrders[activeBuyOrders[i - 1]].isActive) {
                activeBuyOrders[i - 1] = activeBuyOrders[activeBuyOrders.length - 1];
                activeBuyOrders.pop();
            }
        }
        
        for (uint256 i = activeSellOrders.length; i > 0; i--) {
            if (!sellOrders[activeSellOrders[i - 1]].isActive) {
                activeSellOrders[i - 1] = activeSellOrders[activeSellOrders.length - 1];
                activeSellOrders.pop();
            }
        }
    }

    /**
     * @notice Settles a trade after delivery confirmation
     * @param tradeId The ID of the trade to settle
     */
    function executeTradeSettlement(uint256 tradeId) 
        external 
        validTrade(tradeId) 
        nonReentrant 
    {
        Trade storage trade = trades[tradeId];
        require(!trade.isSettled, "Trade already settled");
        require(msg.sender == solarSyncCoreAddress, "Only Core can settle");
        
        BuyOrder storage buyOrder = buyOrders[trade.buyOrderId];
        SellOrder storage sellOrder = sellOrders[trade.sellOrderId];
        
        uint256 actualDelivered = trade.matchedAmountKWh;
        uint256 paymentAmount = actualDelivered * trade.finalPricePerKWhWei;

        uint256 fee = (paymentAmount * platformFeePercentage) / 10000;
        uint256 producerPayment = paymentAmount - fee;
        
        uint256 refundAmount = 0;
        if (actualDelivered < trade.matchedAmountKWh) {
            refundAmount = trade.escrowedAmount - paymentAmount;
            (bool refundSuccess, ) = buyOrder.consumer.call{value: refundAmount}("");
            require(refundSuccess, "Consumer refund failed");
        }
        
        (bool paymentSuccess, ) = sellOrder.producer.call{value: producerPayment}("");
        require(paymentSuccess, "Producer payment failed");
        
        if (fee > 0) {
            (bool feeSuccess, ) = feeRecipient.call{value: fee}("");
            require(feeSuccess, "Fee transfer failed");
        }
        
        // Update trade status
        trade.isSettled = true;
        trade.isCompleted = true;
        trade.actualDelivered = actualDelivered;
        trade.settledAt = block.timestamp;
        
        emit TradeSettled(tradeId, actualDelivered, paymentAmount, refundAmount);
    }

    /**
     * @notice Cancel an active buy order
     */
    function cancelBuyOrder(uint256 orderId, string memory reason) 
        external 
        validBuyOrder(orderId) 
        nonReentrant 
    {
        BuyOrder storage order = buyOrders[orderId];
        require(order.consumer == msg.sender, "Not order owner");
        require(order.isActive, "Order not active");
        
        order.isActive = false;
        
        // Refund escrowed funds
        if (order.escrowedFunds > 0) {
            uint256 refundAmount = order.escrowedFunds;
            order.escrowedFunds = 0;
            
            (bool success, ) = order.consumer.call{value: refundAmount}("");
            require(success, "Refund failed");
        }
        
        emit OrderCancelled(orderId, msg.sender, true, reason);
    }

    /**
     * @notice Cancel an active sell order
     */
    function cancelSellOrder(uint256 orderId, string memory reason) 
        external 
        validSellOrder(orderId) 
    {
        SellOrder storage order = sellOrders[orderId];
        require(order.producer == msg.sender, "Not order owner");
        require(order.isActive, "Order not active");
        
        order.isActive = false;
        
        emit OrderCancelled(orderId, msg.sender, false, reason);
    }

    /**
     * @notice Get best match for a buy order
     */
    function getBestMatch(
        uint256 amount, 
        uint256 maxPrice, 
        address buyer
    ) external view returns (uint256 matchedSellOrderId, uint256 matchedPrice) {
        uint256 bestId = 0;
        uint256 lowestPrice = type(uint256).max;

        for (uint256 i = 0; i < activeSellOrders.length; i++) {
            uint256 sellId = activeSellOrders[i];
            SellOrder storage sellOrder = sellOrders[sellId];

            if (sellOrder.isActive && 
                (sellOrder.amountKWh - sellOrder.filledAmount) >= amount &&
                sellOrder.pricePerKWhWei <= maxPrice &&
                sellOrder.pricePerKWhWei < lowestPrice) 
            {
                lowestPrice = sellOrder.pricePerKWhWei;
                bestId = sellId;
            }
        }
        
        return (bestId, lowestPrice);
    }

    /**
     * @notice Get all active buy orders
     */
    function getActiveBuyOrders() external view returns (uint256[] memory) {
        return activeBuyOrders;
    }

    /**
     * @notice Get all active sell orders
     */
    function getActiveSellOrders() external view returns (uint256[] memory) {
        return activeSellOrders;
    }

    /**
     * @notice Get user's buy orders
     */
    function getUserBuyOrders(address user) external view returns (uint256[] memory) {
        return userBuyOrders[user];
    }

    /**
     * @notice Get user's sell orders
     */
    function getUserSellOrders(address user) external view returns (uint256[] memory) {
        return userSellOrders[user];
    }

    /**
     * @notice Get user's trades
     */
    function getUserTrades(address user) external view returns (uint256[] memory) {
        return userTrades[user];
    }

    /**
     * @notice Get trade details
     */
    function getTrade(uint256 tradeId) external view validTrade(tradeId) returns (Trade memory) {
        return trades[tradeId];
    }

    /**
     * @notice Get order book statistics
     */
    function getOrderBookStats() external view returns (
        uint256 totalBuyOrders,
        uint256 totalSellOrders,
        uint256 activeBuyOrdersCount,
        uint256 activeSellOrdersCount,
        uint256 totalTrades
    ) {
        totalBuyOrders = nextBuyOrderId - 1;
        totalSellOrders = nextSellOrderId - 1;
        activeBuyOrdersCount = activeBuyOrders.length;
        activeSellOrdersCount = activeSellOrders.length;
        totalTrades = nextTradeId - 1;
    }

    function _min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    function matchOrders() external onlyCoreContract {
        _matchOrders();
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @notice Emergency withdrawal for stuck funds (only owner)
     */
    function emergencyWithdraw(address payable recipient, uint256 amount) external onlyOwner {
        require(recipient != address(0), "Invalid recipient");
        (bool success) = recipient.call{value: amount}("");
        require(success, "Withdrawal failed");
    }

    receive() external payable {}
}