// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/**
 * @title TradingEngine
 * @dev Handles order matching, escrow, and automated settlement between producers and consumers.
 */
contract TradingEngine {
    
    // --- STRUCTS ---
    
    struct BuyOrder {
        uint256 id;
        address consumer;
        uint256 amountKWh;
        uint256 maxPricePerKWhWei;
        uint256 escrowedFunds; // Funds locked by the consumer
        bool isActive;
    }
    
    struct SellOrder {
        uint256 id;
        address producer;
        uint256 amountKWh;
        uint256 pricePerKWhWei;
        uint256 deliveryTimestamp;
        bool isActive;
    }
    
    struct Trade {
        uint256 id;
        uint256 buyOrderId;
        uint256 sellOrderId;
        uint256 matchedAmountKWh;
        uint256 finalPricePerKWhWei;
        uint256 escrowedAmount;
        bool isSettled;
    }

    // --- STATE VARIABLES ---

    // Note: In a production system, these might use more complex data structures (e.g., priority queue)
    // for efficient matching, but simple mappings suffice for demonstration.
    mapping(uint256 => BuyOrder) public buyOrders;
    mapping(uint256 => SellOrder) public sellOrders;
    mapping(uint256 => Trade) public trades;

    uint256 private nextBuyOrderId = 1;
    uint256 private nextSellOrderId = 1;
    uint256 private nextTradeId = 1;

    address public solarSyncCoreAddress; // Address of the main contract

    // --- CONSTRUCTOR ---

    constructor(address _coreAddress) {
        solarSyncCoreAddress = _coreAddress;
    }
    
    // --- MODIFIERS ---
    
    // Ensure only the SolarSyncCore contract can call critical functions
    modifier onlyCoreContract() {
        require(msg.sender == solarSyncCoreAddress, "Only SolarSync Core can call this.");
        _;
    }

    // --- ORDER PLACEMENT (Called by SolarSyncCore on behalf of participants) ---

    /**
     * @notice Places a consumer buy order and escrows funds.
     * @dev This is called by SolarSyncCore.
     */
    function placeBuyOrder(
        address consumer, 
        uint256 amountKWh, 
        uint256 maxPricePerKWhWei
    ) external payable onlyCoreContract returns (uint256 buyId) {
        uint256 orderId = nextBuyOrderId++;
        uint256 requiredDeposit = amountKWh * maxPricePerKWhWei;
        
        require(msg.value >= requiredDeposit, "Insufficient escrow deposit.");
        
        buyOrders[orderId] = BuyOrder({
            id: orderId,
            consumer: consumer,
            amountKWh: amountKWh,
            maxPricePerKWhWei: maxPricePerKWhWei,
            escrowedFunds: msg.value, // Escrow the full amount
            isActive: true
        });
        
        return orderId;
    }
    
    /**
     * @notice Places a producer sell order.
     * @dev This is called by SolarSyncCore.
     */
    function placeSellOrder(
        address producer, 
        uint256 amountKWh, 
        uint256 pricePerKWhWei, 
        uint256 deliveryTimestamp
    ) external onlyCoreContract returns (uint256 sellId) {
        uint256 orderId = nextSellOrderId++;
        
        sellOrders[orderId] = SellOrder({
            id: orderId,
            producer: producer,
            amountKWh: amountKWh,
            pricePerKWhWei: pricePerKWhWei,
            deliveryTimestamp: deliveryTimestamp,
            isActive: true
        });
        
        return orderId;
    }

    // --- ORDER MATCHING ---

    /**
     * @notice Finds the best match for a buy order from available sell orders.
     * @dev This is a simplified matching algorithm (First-In, First-Out, Best Price).
     * @param amount The requested energy amount.
     * @param maxPrice The maximum price buyer is willing to pay.
     * @param buyer The address of the buyer.
     * @return matchedSellOrderId The ID of the best matched sell order.
     * @return matchedPrice The final matched price per KWh.
     */
    function getBestMatch(
        uint256 amount, 
        uint256 maxPrice, 
        address buyer // Buyer is passed to ensure matching integrity
    ) external view returns (uint256 matchedSellOrderId, uint256 matchedPrice) {
        
        uint256 bestId = 0;
        uint256 lowestPrice = type(uint256).max; // Initialize with max possible value

        // Iterate through all active sell orders (inefficient but simple)
        for (uint256 i = 1; i < nextSellOrderId; i++) {
            SellOrder storage sellOrder = sellOrders[i];

            if (sellOrder.isActive && 
                sellOrder.amountKWh >= amount &&
                sellOrder.pricePerKWhWei <= maxPrice &&
                sellOrder.pricePerKWhWei < lowestPrice) 
            {
                // Found a better match
                lowestPrice = sellOrder.pricePerKWhWei;
                bestId = i;
            }
        }
        
        return (bestId, lowestPrice);
    }
    
    // --- TRADE EXECUTION AND SETTLEMENT ---

    /**
     * @notice Executes a matched trade and creates a Trade record.
     * @dev Called by the Core contract after a match is found. This is where escrow funds are linked to the trade.
     */
    function executeTrade(
        uint256 buyId, 
        uint256 sellId, 
        uint256 matchedAmountKWh,
        uint256 finalPricePerKWhWei
    ) external onlyCoreContract returns (uint256 tradeId) {
        
        BuyOrder storage buyOrder = buyOrders[buyId];
        SellOrder storage sellOrder = sellOrders[sellId];
        
        require(buyOrder.isActive && sellOrder.isActive, "Order not active.");
        require(matchedAmountKWh > 0, "Matched amount must be positive.");

        // Calculate actual cost
        uint256 actualCost = matchedAmountKWh * finalPricePerKWhWei;
        require(buyOrder.escrowedFunds >= actualCost, "Escrow funds too low for trade.");

        // Adjust remaining amounts for partial fills (if necessary)
        buyOrder.amountKWh -= matchedAmountKWh;
        sellOrder.amountKWh -= matchedAmountKWh;

        // Deactivate orders if fully filled
        if (buyOrder.amountKWh == 0) buyOrder.isActive = false;
        if (sellOrder.amountKWh == 0) sellOrder.isActive = false;

        uint256 currentTradeId = nextTradeId++;
        trades[currentTradeId] = Trade({
            id: currentTradeId,
            buyOrderId: buyId,
            sellOrderId: sellId,
            matchedAmountKWh: matchedAmountKWh,
            finalPricePerKWhWei: finalPricePerKWhWei,
            escrowedAmount: actualCost, // Lock only the required funds for the trade
            isSettled: false
        });

        // Refund any excess funds from the consumer's initial deposit
        uint256 excessFunds = buyOrder.escrowedFunds - actualCost;
        if (excessFunds > 0) {
            // Note: This should ideally be handled carefully with re-entrancy protection
            (bool success, ) = buyOrder.consumer.call{value: excessFunds}("");
            require(success, "Refund failed.");
            buyOrder.escrowedFunds = actualCost;
        }

        return currentTradeId;
    }

    /**
     * @notice Finalizes the trade by transferring funds to the producer after generation confirmation.
     * @dev Called by the Core contract upon successful generation confirmation.
     * @param tradeId The ID of the trade to settle.
     * @param actualAmountKWh The final confirmed amount (used for pro-rata settlement).
     */
    function executeTradeSettlement(uint256 tradeId, uint256 actualAmountKWh) external onlyCoreContract {
        Trade storage trade = trades[tradeId];
        require(!trade.isSettled, "Trade already settled.");

        // --- Settlement Logic ---
        uint256 amountToSettle = trade.matchedAmountKWh;
        uint256 paymentAmount = trade.escrowedAmount;

        // If actual generation is less than promised (pro-rata settlement)
        if (actualAmountKWh < trade.matchedAmountKWh) {
            // Recalculate payment based on actual delivered amount
            amountToSettle = actualAmountKWh;
            paymentAmount = amountToSettle * trade.finalPricePerKWhWei;
            
            // Refund the consumer for the undelivered portion
            uint256 refundAmount = trade.escrowedAmount - paymentAmount;
            BuyOrder storage buyOrder = buyOrders[trade.buyOrderId];
            (bool refundSuccess, ) = buyOrder.consumer.call{value: refundAmount}("");
            require(refundSuccess, "Final refund failed.");
        }
        
        // --- Platform Fee (e.g., 0.5%) ---
        uint256 fee = paymentAmount / 200; // 1/200 = 0.005 = 0.5%
        uint256 producerPayment = paymentAmount - fee;
        
        // Transfer the net amount to the producer
        SellOrder storage sellOrder = sellOrders[trade.sellOrderId];
        (bool paymentSuccess, ) = sellOrder.producer.call{value: producerPayment}("");
        require(paymentSuccess, "Producer payment failed.");

        trade.isSettled = true;
        // In a full implementation, the fee would be sent to a treasury contract.
    }
}