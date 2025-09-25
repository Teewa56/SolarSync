// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title ReputationSystem
 * @dev Manages the reliability score for energy producers and consumers.
 */
contract ReputationSystem is Ownable {

    // Maps participant address to their reputation score (e.g., 0 to 1000)
    mapping(address => uint256) private s_reputationScores;

    // The address of the TradingEngine contract, authorized to update scores
    address public tradingEngineAddress;

    // Configuration constants for score adjustments
    uint256 private constant STARTING_SCORE = 500;
    uint256 private constant MAX_SCORE = 1000;
    uint256 private constant MIN_SCORE = 0;
    
    // Score adjustments
    uint256 private constant SCORE_SUCCESS = 5;
    uint256 private constant SCORE_FAILURE = 20;
    uint256 private constant SCORE_PREDICTION_BONUS = 10; // Extra bonus for highly accurate predictions

    // --- CONSTRUCTOR ---
    
    constructor(address _engineAddress) Ownable(msg.sender) {
        tradingEngineAddress = _engineAddress;
    }

    // --- MODIFIER ---

    // Restrict score update ability to only the TradingEngine contract
    modifier onlyTradingEngine() {
        require(msg.sender == tradingEngineAddress, "Caller must be the Trading Engine.");
        _;
    }

    // --- REPUTATION UPDATE LOGIC ---
    
    /**
     * @notice Initializes a participant's score upon first successful trade.
     * @param participant The address of the participant.
     */
    function initializeScore(address participant) public {
        if (s_reputationScores[participant] == 0) {
            s_reputationScores[participant] = STARTING_SCORE;
        }
    }

    /**
     * @notice Updates the producer's score based on their generation accuracy.
     * @dev Called by the TradingEngine after a producer confirms generation.
     * @param producer The address of the producer.
     * @param promisedKWh The amount of energy promised/predicted.
     * @param actualKWh The actual amount of energy delivered.
     */
    function updateProducerScore(
        address producer, 
        uint256 promisedKWh, 
        uint256 actualKWh
    ) external onlyTradingEngine {
        
        initializeScore(producer);
        uint256 currentScore = s_reputationScores[producer];

        // 1. Calculate accuracy deviation (Simplified to absolute difference)
        uint256 difference = (promisedKWh > actualKWh) ? (promisedKWh - actualKWh) : (actualKWh - promisedKWh);

        // 2. Adjust based on success/failure
        if (actualKWh >= promisedKWh * 95 / 100) { // Success: delivered at least 95%
            currentScore += SCORE_SUCCESS;
            
            // 3. Add bonus for high accuracy (e.g., within 5%)
            if (difference * 100 / promisedKWh <= 5) {
                currentScore += SCORE_PREDICTION_BONUS;
            }
        } else { // Failure: significant under-delivery
            currentScore = (currentScore > SCORE_FAILURE) ? currentScore - SCORE_FAILURE : MIN_SCORE;
        }

        // Clamp the score between MIN and MAX
        s_reputationScores[producer] = _clampScore(currentScore);
    }

    /**
     * @notice Updates the consumer's score based on their payment and consumption confirmation.
     * @dev Called by the TradingEngine after trade settlement.
     * @param consumer The address of the consumer.
     * @param wasSuccessful A boolean indicating if the trade settled successfully.
     */
    function updateConsumerScore(address consumer, bool wasSuccessful) external onlyTradingEngine {
        initializeScore(consumer);
        uint256 currentScore = s_reputationScores[consumer];

        if (wasSuccessful) {
            currentScore += SCORE_SUCCESS;
        } else {
            // Penalize for payment defaults or failure to confirm consumption
            currentScore = (currentScore > SCORE_FAILURE) ? currentScore - SCORE_FAILURE : MIN_SCORE;
        }

        s_reputationScores[consumer] = _clampScore(currentScore);
    }
    
    // --- HELPER FUNCTION ---
    
    /**
     * @dev Ensures the score remains within the defined bounds [MIN_SCORE, MAX_SCORE].
     */
    function _clampScore(uint256 score) private pure returns (uint256) {
        if (score > MAX_SCORE) return MAX_SCORE;
        if (score < MIN_SCORE) return MIN_SCORE;
        return score;
    }

    // --- VIEW/GETTER FUNCTION ---

    /**
     * @notice Retrieves the current reputation score for a participant.
     * @param participant The address of the producer or consumer.
     * @return The current reputation score.
     */
    function getReputationScore(address participant) external view returns (uint256) {
        return s_reputationScores[participant];
    }
    
    // --- ADMINISTRATION ---
    
    /**
     * @notice Allows the owner to update the address of the TradingEngine contract.
     */
    function setTradingEngineAddress(address newEngineAddress) external onlyOwner {
        require(newEngineAddress != address(0), "New address cannot be zero.");
        tradingEngineAddress = newEngineAddress;
    }
}