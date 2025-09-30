// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@chainlink/contracts/src/v0.8/ChainlinkClient.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@chainlink/contracts/src/v0.8/interfaces/LinkTokenInterface.sol";

/**
 * @title EnergyOracle
 * @dev Manages requests to the external ML-Engine API (FastAPI) via Chainlink
 * and stores the resulting energy predictions.
 */
contract EnergyOracle is ChainlinkClient, Ownable {
    mapping(address => uint256) private s_producerPredictions;
    mapping(bytes32 => address) private s_requestIdToProducer;

    address private constant LINK_TOKEN = 0x0; // Sepolia LINK- change this to hedera
    address private constant ORACLE_ADDRESS = "0x..."; // Replace with actual Chainlink Oracle address
    bytes32 private constant JOB_ID = "JOB_ID"; // Replace with actual job ID for /predict/solar

    event ChainlinkRequested(bytes32 indexed requestId);
    event PredictionFulfilled(address indexed producer, uint256 predictedAmountKWh);

    constructor(address _oracle, bytes32 _jobId) {
        setChainlinkToken(LINK_TOKEN);
        setChainlinkOracle(_oracle);
        setChainlinkJobId(_jobId);
    }

    /**
     * @notice Initiates a Chainlink request to the ML API to fetch a prediction.
     * @dev Only the owner (e.g., SolarSync governance or platform) can trigger this.
     * @param producer The address of the energy producer.
     * @param lat The latitude of the producer's facility.
     * @param lng The longitude of the producer's facility.
     */
    function requestEnergyPrediction(address producer, int256 lat,int256 lng) public onlyOwner returns (bytes32 requestId) {
        Chainlink.Request memory request = buildChainlinkRequest(JOB_ID, address(this), this.fulfillPrediction.selector);
        request.add("endpoint", "predict/solar");
        request.addInt("lat", lat);
        request.addInt("lng", lng);
        request.add("path", "predicted_kwh.0");
        request.addInt("times", 100);
        requestId = sendChainlinkRequest(request, 0.1 * 10**18);
        s_requestIdToProducer[requestId] = producer;
        emit ChainlinkRequested(requestId);
        return requestId;
    }

    /**
     * @notice Callback function to receive the prediction result from the Chainlink node.
     * @dev This function signature MUST match the one defined in the request.
     * @param requestId The ID of the original request.
     * @param prediction The data returned from the external adapter (ML-Engine API).
     */
    function fulfillPrediction(bytes32 requestId, uint256 prediction) public recordChainlinkFulfillment(requestId){
        address producerAddress = s_requestIdToProducer[requestId];
        require(producerAddress != address(0), "Request ID not mapped to producer.");
        s_producerPredictions[producerAddress] = prediction;
        delete s_requestIdToProducer[requestId];
        emit PredictionFulfilled(producerAddress, prediction);
    }

    /**
     * @notice Getter function for the latest ML prediction.
     * @param producerAddress The address of the producer.
     * @return predictedAmountKWh The predicted energy generation in KWh.
     */
    function getEnergyPrediction(address producerAddress) external view returns (uint256 predictedAmountKWh) {
        return s_producerPredictions[producerAddress];
    }

    /**
     * @notice Allows the owner to deposit LINK for the contract to pay for oracle requests.
     */
    function fundContract() public onlyOwner {
        // This is a placeholder: funds must be sent externally to the contract's address.
        // The owner would transfer LINK tokens to this contract's address.
    }

    /**
     * @notice Allows the owner to withdraw excess LINK tokens.
     */
    function withdrawLink() public onlyOwner {
        LinkTokenInterface link = LinkTokenInterface(chainlinkToken());
        require(link.transfer(msg.sender, link.balanceOf(address(this))), "Unable to transfer LINK.");
    }
}