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
};
