pub const ValidateError = error{
    EmptyState,
    NoDecisions,
    TooManyDecisions,
    DuplicateId,
    ReservedId,
    BadOptionCount,
    BadRange,
    BadItemCount,
    OutOfMemory,
};

pub const EngineError = ValidateError || error{
    ModelFailed,
    BadTemperature,
    BadState,
    BadModelIO,
};

pub const GraphError = error{
    EmptyGraph,
    TooManyNodes,
    DuplicateNodeId,
    EmptyNodeId,
    UnknownDecision,
    DuplicateDecisionRef,
    UnknownEdgeNode,
    SelfLoop,
    Cycle,
    OutOfMemory,
};
