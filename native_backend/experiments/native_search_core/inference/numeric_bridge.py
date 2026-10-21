"""Numeric pause/reply boundary; all model math stays in the frozen ai module.

One request == one original inference call (SHARED_ABI.md section 3):

* PV   -> the remote branch of ai._get_priors_and_value (`client.pv(key, vec, ae, ri)`)
          when the net carries `_client` (or ai.INFER_CLIENT) and `remote_key`;
          otherwise that function's local torch branch on the single vector
          (forward_backbone / value_head / score_actions_spatial).
* PVM  -> ai._pv_forward_many(net, vecs, aes, ris) followed by the per-record
          post-processing of ai._flush_pending.
* EVAL -> ai._eval_forward_many(net, vecs); the singleton ai._mcts_eval_state makes
          the identical `client.eval(key, X[1,4647])` / `net(x)` call.

Reply arrays: priors = _validate_search_priors(p) then `p / (p.sum() + 1e-10)` in
float32; values = float(_validate_search_value(v)); conf = the _conf_shrink
confidence rule (1.0 when inactive, NaN means retain prior confidence); mixed = the
_remember_opponent_policy arithmetic over the normalized priors (float64).
Requests are never merged, split, deduplicated or reordered. No heuristic fallback.
"""
from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np

OBSERVATION_WIDTH = 4647   # wotr_snapshot_encode_state output width (frozen v52 observation)


class Operation(IntEnum):
    PV = 1
    PVM = 2
    EVAL = 3
    COMBAT_DIST = 4    # SHARED_ABI 9.5: metadata payload, never a numeric request (rejected below)


def integer(value, name, maximum=(1 << 64) - 1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not 0 <= int(value) <= maximum:
        raise ValueError(name + ' is out of range')
    return int(value)


def array(value, dtype, shape, name, writable=False):
    if (type(value) is not np.ndarray or value.dtype != np.dtype(dtype)
            or value.shape != shape or not value.flags.c_contiguous
            or not value.flags.aligned or (writable and not value.flags.writeable)):
        raise ValueError(name + ' has an invalid dtype, shape, layout or mutability')
    return value


def owned(value, dtype, writable=False):
    result = np.array(value, dtype=dtype, order='C', copy=True)
    result.flags.writeable = writable
    return result


@dataclass(frozen=True)
class Request:
    """One controller request (ABI WotrSearchRequest) as owned NumPy arrays."""
    request_id: int
    operation: int
    model_side: int
    purpose: int
    leaf_ids: np.ndarray       # uint64 [B]
    observations: np.ndarray   # float32 [B, feature_dim]
    actions: np.ndarray        # float32 [rows, action_dim]
    regions: np.ndarray        # int64 [rows]
    offsets: np.ndarray        # uint64 [B+1]

    @classmethod
    def from_view(cls, view):
        """Adopt a controller_binding.RequestView (already owned copies)."""
        return cls(view.request_id, view.operation, view.model_side, view.purpose, view.leaf_ids,
                   view.observations, view.actions, view.regions, view.offsets)

    def validate(self, feature_dim, action_dim, region_count):
        integer(self.request_id, 'request_id')
        integer(self.model_side, 'model_side', 1)
        integer(self.purpose, 'purpose', (1 << 31) - 1)
        try:
            op = Operation(integer(self.operation, 'operation', 4))
        except ValueError as error:
            raise ValueError('Unknown inference operation') from error
        if op == Operation.COMBAT_DIST:
            raise ValueError('COMBAT_DIST is a metadata request (orchestrator.service_combat_request), not a numeric one')
        if type(self.observations) is not np.ndarray or self.observations.ndim != 2:
            raise ValueError('Observations must be a matrix')
        batch = self.observations.shape[0]
        if batch < 1 or (op == Operation.PV and batch != 1):
            raise ValueError('Invalid inference batch size')
        array(self.observations, np.float32, (batch, feature_dim), 'observations')
        array(self.leaf_ids, np.uint64, (batch,), 'leaf_ids')
        array(self.offsets, np.uint64, (batch + 1,), 'offsets')
        if len(np.unique(self.leaf_ids)) != batch:
            raise ValueError('Leaf correlation IDs must be unique within a request')
        if self.offsets[0] != 0 or np.any(self.offsets[1:] < self.offsets[:-1]):
            raise ValueError('Offsets must begin at zero and be monotonic')
        rows = int(self.offsets[-1])
        array(self.actions, np.float32, (rows, action_dim), 'actions')
        array(self.regions, np.int64, (rows,), 'regions')
        if op == Operation.EVAL:
            if rows or np.any(self.offsets):
                raise ValueError('Value-only batches cannot carry policy rows')
        elif np.any(self.offsets[1:] == self.offsets[:-1]):
            raise ValueError('Policy leaves must have at least one legal row')
        if np.any(self.regions >= region_count) or np.any(self.regions < -1):
            raise ValueError('Region index outside the original region table (-1 = non-regional)')
        if not np.isfinite(self.observations).all() or not np.isfinite(self.actions).all():
            raise ValueError('Non-finite numeric inference input')
        return op, batch, rows

    def bounds(self):
        return [(int(a), int(b)) for a, b in zip(self.offsets[:-1], self.offsets[1:])]


@dataclass(frozen=True)
class Reply:
    """One controller reply (ABI WotrSearchReply); arrays are owned and read-only."""
    request_id: int
    operation: int
    batch: int
    rows: int
    priors: np.ndarray   # float32 [rows] normalized, None for EVAL
    values: np.ndarray   # float64 [B] raw validated values
    conf: np.ndarray     # float64 [B]
    mixed: np.ndarray    # float64 [rows], None for EVAL

    def validate(self, request):
        if integer(self.request_id, 'request_id') != integer(request.request_id, 'request_id'):
            raise ValueError('Inference response correlation differs: request_id')
        if int(self.operation) != int(request.operation):
            raise ValueError('Inference response correlation differs: operation')
        batch = len(request.leaf_ids)
        rows = int(request.offsets[-1])
        if self.batch != batch or self.rows != rows:
            raise ValueError('Response batch/rows differ from the request')
        array(self.values, np.float64, (batch,), 'response values')
        array(self.conf, np.float64, (batch,), 'response conf')
        if not np.isfinite(self.values).all():
            raise ValueError('Non-finite inference response value')
        # The controller's conf_shrink treats NaN as no measurement: do not
        # overwrite last_conf. Infinity is not that sentinel.
        if np.isinf(self.conf).any() or np.any(self.conf < 0.0) or np.any(self.conf > 1.0):
            raise ValueError('Confidence outside [0, 1]')
        if int(request.operation) == Operation.EVAL:
            if self.priors is not None or self.mixed is not None:
                raise ValueError('EVAL reply cannot carry priors or mixed policy')
            return self
        array(self.priors, np.float32, (rows,), 'response priors')
        array(self.mixed, np.float64, (rows,), 'response mixed')
        if not np.isfinite(self.priors).all() or np.any(self.priors < 0):
            raise ValueError('Invalid inference prior')
        if not np.isfinite(self.mixed).all() or np.any(self.mixed < 0):
            raise ValueError('Invalid mixed opponent policy')
        return self


def _side_enum(ai, model_side):
    return ai.Side.FREE_PEOPLES if int(model_side) == 0 else ai.Side.SHADOW


class NumericInferenceBridge:
    """Dispatch one request to the frozen inference functions; own the reply.

    Python receives no actions, state objects, menus, node objects or callbacks.
    The caller must not mutate a request in flight; reply arrays are read-only.
    """

    def __init__(self, ai, fp_net, sh_net):
        if fp_net is None or sh_net is None:
            raise ValueError('Numeric inference has no heuristic model fallback; both nets are required')
        self.ai = ai
        self.nets = (fp_net, sh_net)
        # Coordinator fix (2026-09-09): never call ai.get_feature_dim() inside a search.
        # The frozen helper builds an initial game state and shuffles the hunt pool,
        # consuming 18 words of the active (search) RNG stream; the source search never
        # does that. The observation width is the controller's fixed 4647 (encode_state).
        self.feature_dim = OBSERVATION_WIDTH
        self.action_dim = ai.ACTION_ENCODING_DIM
        self.region_count = ai.NUM_REGIONS
        for index, net in enumerate(self.nets):
            side = getattr(net, 'side', None)
            expected = ('fp', 'sh')[index]
            if isinstance(side, ai.Side):
                side = 'fp' if side == ai.Side.FREE_PEOPLES else 'sh'
            if side in ('fp', 'sh') and side != expected:
                raise ValueError('Model side %r differs from its slot binding %r' % (side, expected))

    # -- confidence: the exact _conf_shrink rule for c (not the shrink itself) --
    def confidence(self, vec, side, v_raw):
        ai = self.ai
        if (not ai._CONF_LEAF and not ai._CONF_TRACK) or ai._CONF_FN is None:
            return 1.0
        if ai._CONF_REF != 'abs':
            raise NotImplementedError('Numeric bridge supports _CONF_REF="abs" only; got %r' % (ai._CONF_REF,))
        try:
            c = float(ai._CONF_FN(vec, side, v_raw))
        except Exception:
            return float('nan')
        if not (c == c):
            return float('nan')
        return 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)

    # -- mixed: _remember_opponent_policy arithmetic over the normalized priors --
    def mixed(self, priors, count):
        ai = self.ai
        probabilities = np.asarray(ai._validate_search_priors(priors, count), dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        probabilities = ((1.0 - ai._OPPONENT_POLICY_EPSILON) * probabilities
                         + ai._OPPONENT_POLICY_EPSILON / count)
        return probabilities

    @staticmethod
    def _client(ai, net):
        client = getattr(net, '_client', None) or ai.INFER_CLIENT
        if client is not None and getattr(net, 'remote_key', None):
            return client
        return None

    def _pv(self, net, vec, ae, ri, rows):
        """_get_priors_and_value: remote branch or local torch branch; one vector."""
        ai = self.ai
        client = self._client(ai, net)
        if client is not None:
            priors, v = client.pv(net.remote_key, np.asarray(vec, dtype=np.float32), ae, ri)
            v = ai._validate_search_value(v)
            priors = ai._validate_search_priors(priors, rows)
            priors = priors / (priors.sum() + 1e-10)
            return priors, float(v)
        torch = ai.torch
        net.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.array([vec])).to(ai.DEVICE)
            backbone_out, node_emb = net.forward_backbone(x)
            v = ai._validate_search_value(net.value_head(backbone_out).squeeze(-1).item())
            ae_t = torch.from_numpy(ae).to(ai.DEVICE)
            ri_t = torch.from_numpy(ri).to(ai.DEVICE)
            priors_t = net.score_actions_spatial(backbone_out, node_emb, ae_t, ri_t)
            priors = priors_t.cpu().numpy().astype(np.float32)
        priors = ai._validate_search_priors(priors, rows)
        priors = priors / (priors.sum() + 1e-10)
        return priors, float(v)

    def run(self, request):
        ai = self.ai
        op, batch, rows = request.validate(self.feature_dim, self.action_dim, self.region_count)
        net = self.nets[int(request.model_side)]
        side = _side_enum(ai, request.model_side)
        # Writable working copies: torch.from_numpy refuses/warns on read-only input,
        # and the bridge must never hand controller memory to the nets.
        observations = owned(request.observations, np.float32, writable=True)
        vecs = [observations[i] for i in range(batch)]
        if op == Operation.EVAL:
            values = ai._eval_forward_many(net, vecs)
            ai._validate_search_batch_count(values, batch, 'Deferred root value replies')
            values = [float(v) for v in values]
            conf = [self.confidence(vecs[i], side, values[i]) for i in range(batch)]
            return Reply(int(request.request_id), int(op), batch, rows, None,
                         owned(values, np.float64), owned(conf, np.float64), None).validate(request)
        actions = owned(request.actions, np.float32, writable=True)
        regions = owned(request.regions, np.int64, writable=True)
        bounds = request.bounds()
        aes = [actions[a:b] for a, b in bounds]
        ris = [regions[a:b] for a, b in bounds]
        normalized = []
        values = []
        if op == Operation.PV:
            priors, v = self._pv(net, vecs[0], aes[0], ris[0], rows)
            normalized.append(priors)
            values.append(v)
        else:
            out = ai._pv_forward_many(net, vecs, aes, ris)
            ai._validate_search_batch_count(out, batch, 'Deferred policy/value replies')
            for (a, b), (p, v) in zip(bounds, out):
                pr = np.asarray(ai._validate_search_priors(p, b - a), dtype=np.float32)
                v = ai._validate_search_value(v)
                normalized.append(pr / (pr.sum() + 1e-10))
                values.append(float(v))
        conf = [self.confidence(vecs[i], side, values[i]) for i in range(batch)]
        for p, (a, b) in zip(normalized, bounds):
            array(p, np.float32, (b - a,), 'normalized priors')
        mixed = [self.mixed(p, b - a) for p, (a, b) in zip(normalized, bounds)]
        return Reply(int(request.request_id), int(op), batch, rows,
                     owned(np.concatenate(normalized), np.float32), owned(values, np.float64),
                     owned(conf, np.float64), owned(np.concatenate(mixed), np.float64)).validate(request)


def request_from_view(view):
    return Request.from_view(view)
