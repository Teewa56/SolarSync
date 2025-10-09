// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@chainlink/contracts/src/v0.8/operatorforwarder/Operator.sol";

contract MyOperator is Operator {    
    constructor(address _link) Operator(_link, msg.sender) {}
}