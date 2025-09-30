// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/**
 * @title ISolarSyncInterfaces
 * @dev Defines interfaces for the specialized contracts used by SolarSyncCore
 */
interface IEnergyOracle {
    function getEnergyPrediction(address producerAddress) external view returns (uint256 predictedAmountKWh);
    function updatePrediction(address producerAddress, uint256 predictedAmountKWh) external;
}

interface ITradingEngine {
    function matchOrders() external;
    function executeTradeSettlement(uint256 tradeId) external;
    function getBestMatch(uint256 amount, uint256 maxPrice, address buyer) external returns (uint256 listingId, uint256 matchedPrice);
}