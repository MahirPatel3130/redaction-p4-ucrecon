class PipelineError(Exception):
    """Base exception whose message must never include protected content."""


class ConfigurationError(PipelineError):
    pass


class ExtractionError(PipelineError):
    pass


class InferenceError(PipelineError):
    pass


class ContractError(PipelineError):
    pass


class PrivacyError(PipelineError):
    pass


class ReleaseError(PipelineError):
    pass

