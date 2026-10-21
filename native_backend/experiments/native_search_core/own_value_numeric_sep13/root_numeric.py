"""Root-only model boundary for the isolated native own-value port.

All inference arithmetic is inherited from the existing numeric bridge. This
component does not enable the still-unported native controller search mode.
"""
from numeric_bridge import NumericInferenceBridge, OBSERVATION_WIDTH, Operation, integer
from reply_evaluation import validate_root_context


class RootNumericInferenceBridge(NumericInferenceBridge):
    def __init__(self, ai, own_net, root_side):
        tag=validate_root_context(ai,root_side,own_net)
        self.ai=ai;self.root_side=root_side;self.root_index=0 if tag=='fp' else 1
        self.own_net=own_net
        self.nets=(own_net,None) if self.root_index==0 else (None,own_net)
        self.feature_dim=OBSERVATION_WIDTH
        self.action_dim=ai.ACTION_ENCODING_DIM;self.region_count=ai.NUM_REGIONS

    def run(self, request):
        # Validate the operation first: combat metadata is deliberately handled
        # elsewhere and its actor label must never be changed to the root side.
        operation=Operation(integer(request.operation,'operation',4))
        if operation==Operation.COMBAT_DIST:
            raise ValueError('COMBAT_DIST must use the model-free combat service')
        if integer(request.model_side,'model_side',1)!=self.root_index:
            raise ValueError('Own-value numeric inference accepts only the root model')
        validate_root_context(self.ai,self.root_side,self.own_net)
        expected=(self.own_net,None) if self.root_index==0 else (None,self.own_net)
        if any(a is not b for a,b in zip(self.nets,expected)) or len(self.nets)!=2:
            raise ValueError('Own-value numeric model slots changed during search')
        return super().run(request)


def dispatch_request(bridge, request, combat_service):
    """Keep the established model-free combat route and its original actor."""
    operation=Operation(integer(request.operation,'operation',4))
    if operation==Operation.COMBAT_DIST:
        return combat_service(request)
    return bridge.run(request)
