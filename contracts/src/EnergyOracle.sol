// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@chainlink/contracts/src/v0.8/ChainlinkClient.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title EnergyOracle
 * @dev Manages requests to the external ML-Engine API (FastAPI) via Chainlink
 * and stores the resulting energy predictions.
 */
contract EnergyOracle is ChainlinkClient, Ownable {

    // --- STATE VARIABLES ---
    
    // Stores the latest predicted energy generation (in KWh) for a producer.
    // The key is the producer's address.
    mapping(address => uint256) private s_producerPredictions;

    // Chainlink request ID to track specific data requests
    mapping(bytes32 => address) private s_requestIdToProducer;

    // Chainlink network configuration (Ropsten/Sepolia Testnet defaults)
    address private constant LINK_TOKEN = 0x326C977E6efdC840EaF6d11055d6DFc0CcE6bB6c; // Sepolia LINK
    address private constant ORACLE_ADDRESS = 0x...; // Replace with actual Chainlink Oracle address
    bytes32 private constant JOB_ID = "YOUR_JOB_ID"; // Replace with actual job ID for /predict/solar

    // --- EVENTS ---
    
    event ChainlinkRequested(bytes32 indexed requestId);
    event PredictionFulfilled(address indexed producer, uint256 predictedAmountKWh);

    // --- CONSTRUCTOR ---

    constructor(address _oracle, bytes32 _jobId) {
        // Set the Chainlink token address (mandatory for payments)
        setChainlinkToken(LINK_TOKEN);
        
        // Set the address of the Chainlink node and the specific job ID
        setChainlinkOracle(_oracle);
        setChainlinkJobId(_jobId);
    }
    
    // --- EXTERNAL DATA REQUEST (Called by SolarSyncCore or a Frontend) ---

    /**
     * @notice Initiates a Chainlink request to the ML API to fetch a prediction.
     * @dev Only the owner (e.g., SolarSync governance or platform) can trigger this.
     * @param producer The address of the energy producer.
     * @param lat The latitude of the producer's facility.
     * @param lng The longitude of the producer's facility.
     */
    function requestEnergyPrediction(
        address producer, 
        int256 lat, 
        int256 lng
    ) public onlyOwner returns (bytes32 requestId) {
        
        // 1. Create a Chainlink request object
        Chainlink.Request memory request = buildChainlinkRequest(JOB_ID, address(this), this.fulfillPrediction.selector);

        // 2. Set the data source parameters for the Chainlink External Adapter
        // The external adapter will call the FastAPI endpoint: 
        // GET /api/v1/predict/solar?location_lat={lat}&location_lng={lng}
        request.add("endpoint", "predict/solar"); 
        request.addInt("lat", lat);
        request.addInt("lng", lng);

        // 3. Define the path to parse the JSON response
        // Assuming the ML API returns: {"predicted_kwh": [val1, val2, ...]}
        // We only take the first hour's prediction (index 0)
        request.add("path", "predicted_kwh.0"); 

        // 4. Multiply the resulting value (to handle fixed-point math if needed)
        // Since we're using uint256 for KWh, we might use no multiplier or a small one (e.g., 100)
        // request.addInt("times", 100); 

        // 5. Send the request and pay for it with LINK
        requestId = sendChainlinkRequest(request, 0.1 * 10**18); // 0.1 LINK (example fee)
        
        // 6. Map the request ID back to the producer waiting for the prediction
        s_requestIdToProducer[requestId] = producer;
        
        emit ChainlinkRequested(requestId);
        return requestId;
    }

    // --- CHAINLINK FULFILLMENT (Called by Chainlink Node) ---

    /**
     * @notice Callback function to receive the prediction result from the Chainlink node.
     * @dev This function signature MUST match the one defined in the request.
     * @param requestId The ID of the original request.
     * @param prediction The data returned from the external adapter (ML-Engine API).
     */
    function fulfillPrediction(bytes32 requestId, uint256 prediction) 
        public 
        recordChainlinkFulfillment(requestId)
    {
        // Get the address of the producer associated with the request
        address producerAddress = s_requestIdToProducer[requestId];
        require(producerAddress != address(0), "Request ID not mapped to producer.");

        // Store the official, oracle-verified prediction
        s_producerPredictions[producerAddress] = prediction;
        
        // Cleanup the mapping
        delete s_requestIdToProducer[requestId];
        
        emit PredictionFulfilled(producerAddress, prediction);
    }

    // --- VIEW/GETTER FUNCTION (Called by SolarSyncCore) ---

    /**
     * @notice Getter function for the latest ML prediction.
     * @param producerAddress The address of the producer.
     * @return predictedAmountKWh The predicted energy generation in KWh.
     */
    function getEnergyPrediction(address producerAddress) external view returns (uint256 predictedAmountKWh) {
        return s_producerPredictions[producerAddress];
    }
    
    // --- UTILITY/ADMIN FUNCTIONS ---

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