"""Explicit content rejections, distinct from classifier or device failures."""
from copy import deepcopy


class GuardrailBlockedError(RuntimeError):
    def __init__(self, stage, reason, details=None):
        if stage not in ('positive_prompt', 'negative_prompt', 'output'):
            raise ValueError(f'Unknown guardrail stage: {stage}')
        self.stage = stage
        self.reason = reason
        self.details = deepcopy(details or {})
        super().__init__(f'Guardrail blocked {stage}: {reason}')


def rejection(stage, runner, fallback):
    decision = getattr(runner, 'last_result', None) or {}
    return GuardrailBlockedError(stage, decision.get('message') or fallback, decision)
