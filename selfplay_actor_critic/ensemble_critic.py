"""Mean of several value functions.

Why: the one-ply scoring compares sibling moves by the DIFFERENCE of two
values. On 2026-10-04 two value functions fitted on the same results were
compared on identical candidate states: for the Free Peoples their sibling
credit correlated +0.46, for the Shadow +0.16 (army moves +0.07, card
options -0.55). What one value function says about which Shadow sibling is
better is mostly its own approximation error, and errors of differently
built value functions are close to independent, so their mean keeps what
they agree on and averages the rest away (same 512 games, Shadow
alternative moves within a kind: validity +0.35 and +0.33 alone, +0.57 for
the mean).

The members keep their own architecture and inputs (race critic, full-state
critic, ...). The ensemble exposes the critic interface the actor uses
(forward_backbone, value_head) plus member_values, so a collection can store
what EVERY member said about every candidate and any subset can be compared
afterwards on the same decisions.

  python -m selfplay_actor_critic.ensemble_critic --members seeds/full_critic_r2 seeds/race_critic_r1 \
      --output seeds/ensemble_a
"""
import argparse
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "members."


class EnsembleCritic(torch.nn.Module):
    def __init__(self, members):
        super().__init__()
        members = list(members)
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = torch.nn.ModuleList(members)

    def forward_backbone(self, x):
        """The members read the raw observation themselves."""
        return x, None

    def member_values(self, x):
        """[members, N] values of raw observations."""
        return torch.stack([m.value_head(m.forward_backbone(x)[0]).reshape(-1) for m in self.members])

    def value_head(self, x):
        return self.member_values(x).mean(dim=0)

    def forward(self, x):
        return self.value_head(x)

    @property
    def layout(self):
        """Race read-out layout shared by the members that have one (payloads carry its scalar names)."""
        for member in self.members:
            layout = getattr(member, "layout", None)
            if layout is not None:
                return layout
        return None

    @staticmethod
    def is_ensemble_state(state: dict) -> bool:
        return any(k.startswith(PREFIX) for k in state)

    @staticmethod
    def split_state(state: dict) -> list:
        """The members' own state dicts, in order."""
        parts = {}
        for key, value in state.items():
            if not key.startswith(PREFIX):
                raise ValueError("not an ensemble state: " + key)
            index, _, rest = key[len(PREFIX):].partition(".")
            parts.setdefault(int(index), {})[rest] = value
        if sorted(parts) != list(range(len(parts))):
            raise ValueError("ensemble members are not numbered 0..n-1")
        return [parts[i] for i in range(len(parts))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", nargs="+", required=True, help="critic directories holding {fp,sh}_critic.pt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from .generation import require_generation
    require_generation("ensemble_critic.main")
    os.environ.setdefault("WOTR_NATIVE", "0")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    cwd = os.getcwd()
    import ai
    os.chdir(cwd)
    from .policy import critic_scalar_names, load_critic
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for side in ("fp", "sh"):
        members = [load_critic(ai, Path(d) / f"{side}_critic.pt", side) for d in args.members]
        ensemble = EnsembleCritic(members).eval()
        payload = dict(schema="critic_ensemble_v1", side=side, model_state=ensemble.state_dict(),
                       scalar_names=critic_scalar_names(ensemble), members=[str(Path(d).name) for d in args.members])
        torch.save(payload, output / f"{side}_critic.pt")
        reloaded = load_critic(ai, output / f"{side}_critic.pt", side)
        from .contracts import CONTRACT
        x = torch.rand(8, CONTRACT.observation_width)
        with torch.no_grad():
            if not torch.equal(reloaded.member_values(x), ensemble.member_values(x)):
                raise RuntimeError("ensemble did not round-trip")
        print(f"{side}: {len(members)} members {payload['members']} -> {output / (side + '_critic.pt')}", flush=True)


if __name__ == "__main__":
    main()
