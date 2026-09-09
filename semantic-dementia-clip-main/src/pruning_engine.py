import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

PRUNING_SEED = 42


class CLIPPruningEngine:
    """Engine for applying unstructured and structured decay to CLIP joint

    embedding space layers and transformer backbones across differential
    pruning scenarios (depth zone, sub-module, and structure dimensions per
    the Multidimensional Targeted Pruning Framework).
    """

    def __init__(self, model: nn.Module):
        self.base_model = model
        # Accumulator state for masking_mode="iterative", keyed by
        # (pruning_method, target_area, depth_zone, sub_module) so different
        # sweep configurations don't share/pollute each other's accumulated
        # damage. reset_accumulator() must be called at the start of any new
        # iterative sweep (ascending pruning-level order is required).
        self._accum_state: dict[tuple, dict] = {}

    def reset_accumulator(self, key: tuple | None = None) -> None:
        """Clears iterative-mode accumulated damage.

        Call this before starting a fresh ascending sweep of pruning levels
        with masking_mode="iterative". Pass no argument to clear every
        accumulated configuration, or a specific key (as returned by
        _accum_key) to clear just one.
        """
        if key is None:
            self._accum_state = {}
        else:
            self._accum_state.pop(key, None)

    @staticmethod
    def _accum_key(pruning_method: str, target_area: str, depth_zone: str, sub_module: str) -> tuple:
        return (pruning_method, target_area, depth_zone, sub_module)

    def get_pruned_model(
        self,
        amount: float | dict[str, float],
        encoder_type: str = "joint",
        pruning_method: str = "l1_unstructured",
        target_area: str = "projection",
        depth_zone: str = "global",
        sub_module: str = "all",
        masking_mode: str = "static",
    ) -> nn.Module:
        """Generates a pruned deep copy of the base CLIP model.

        Args:
            amount: Pruning ratio (float between 0.0 and 1.0) applied globally,
              or a dict specifying differential rates, e.g., {"text": 0.4,
              "vision": 0.2}.
            encoder_type: Standardized modality paradigm (defaults to "joint").
            pruning_method: "l1_unstructured" (deterministic global magnitude
              decay, default), "random_unstructured" (global random decay),
              "structured_channel" (L2-norm row/channel pruning on MLP
              linear layers), or "structured_head" (L2-norm ranked removal
              of complete multi-head attention heads).

              "l1_unstructured" and "random_unstructured" are GLOBAL within
              each modality (Directive 6.1's magnitude-driven masking, read
              literally): every targeted weight across every selected layer
              for a given modality (e.g. all 12 vision-encoder attention/MLP
              matrices when target_area="full") is pooled into one
              distribution, and a single percentile threshold is computed
              across that pool -- not a separate threshold recomputed
              per-tensor. This matters: forcing the identical sparsity ratio
              onto every layer independently (the old behavior) is far more
              destructive at a given nominal amount than a true global rank,
              because naturally larger-magnitude (typically more load-bearing)
              layers end up protected while naturally smaller-magnitude,
              more redundant layers absorb correspondingly more of the cut.
              At target_area="projection" there is only one tensor per
              modality, so global vs. per-tensor is equivalent there.
            target_area: "projection" (joint embedding space only) or "full"
              (encoders + joint space). Non-Iterative Masking (Directive 6.1):
              this always deep-copies the pristine base model, so pruning
              masks are computed fresh from the original weights at every
              call rather than compounding across calls.
            depth_zone: Restricts "full"/"backbone" pruning to a Transformer
              depth range: "early" (first third of layers), "middle"
              (middle third), "deep" (final third), or "global" (all
              layers, default -- matches the Global Baseline condition).
              Ignored when target_area == "projection".
            sub_module: Restricts "full"/"backbone" pruning to a specific
              operational weight group: "attention" (Wq,k,v,o), "mlp"
              (Win,out), or "all" (default). Ignored when
              target_area == "projection".
            masking_mode: "static" (default) always deep-copies the
              pristine base model and computes the mask fresh at every call
              (Directive 6.1's non-iterative simplification). "iterative"
              instead compounds damage on top of whatever was previously
              pruned under this exact (pruning_method, target_area,
              depth_zone, sub_module) configuration -- call
              reset_accumulator() before starting a fresh ascending sweep.
              NOTE: for l1_unstructured/structured_channel/structured_head
              (all rank-by-magnitude methods with no retraining in
              between), "iterative" and "static" are mathematically
              IDENTICAL at any given target amount, because weight
              magnitudes never change between calls -- the same global
              threshold recovers the same pruned set either way. The two
              modes only diverge for random_unstructured, where each
              independent random draw is not automatically a superset of
              the previous one; "iterative" mode handles this case
              explicitly so that previously-zeroed weights are guaranteed
              to stay zero as pruning progresses.

        Determinism (Directive 6.1): both random_unstructured mask sampling
        and structured-head/channel tie-breaking are seeded immediately
        before use, so repeated calls with the same arguments against the
        same base model always produce the same pruned weights.
        """
        if encoder_type != "joint":
            raise ValueError(
                "Standardized processing requires encoder_type='joint'."
            )

        # Standardize amount dictionary for differential pruning comparisons
        if isinstance(amount, (float, int)):
            if not (0.0 <= amount <= 1.0):
                raise ValueError("Pruning amount must be between 0.0 and 1.0.")
            rates = {"text": float(amount), "vision": float(amount)}
        elif isinstance(amount, dict):
            rates = {
                "text": float(amount.get("text", 0.0)),
                "vision": float(amount.get("vision", 0.0)),
            }
        else:
            raise TypeError("amount must be a float or a dict.")

        if masking_mode not in ("static", "iterative"):
            raise ValueError("masking_mode must be 'static' or 'iterative'.")

        torch.manual_seed(PRUNING_SEED)

        if masking_mode == "iterative":
            return self._get_iterative_pruned_model(
                rates,
                pruning_method=pruning_method,
                target_area=target_area,
                depth_zone=depth_zone,
                sub_module=sub_module,
            )

        # -------- masking_mode == "static" (Directive 6.1, unchanged) --------
        pruned_model = copy.deepcopy(self.base_model)

        if rates["text"] == 0.0 and rates["vision"] == 0.0:
            return pruned_model

        if pruning_method in ("structured_channel", "structured_head"):
            self._apply_structured_pruning(
                pruned_model,
                rates=rates,
                target_area=target_area,
                depth_zone=depth_zone,
                sub_module=sub_module,
                structure=pruning_method,
            )
            return pruned_model

        target_layers = self._gather_target_layers(
            pruned_model,
            target_area=target_area,
            depth_zone=depth_zone,
            sub_module=sub_module,
        )

        if not target_layers:
            print(
                f"[!] Warning: No layers identified for target_area='{target_area}', "
                f"depth_zone='{depth_zone}', sub_module='{sub_module}'."
            )
            return pruned_model

        pruning_cls = (
            prune.RandomUnstructured
            if pruning_method == "random_unstructured"
            else prune.L1Unstructured
        )

        for modality in ("text", "vision"):
            prune_rate = rates[modality]
            if prune_rate <= 0.0:
                continue

            modality_params = [
                (module, param_name)
                for m, module, param_name in target_layers
                if m == modality
            ]
            if not modality_params:
                continue

            # Global magnitude/random threshold computed once across every
            # targeted tensor for this modality (see the pruning_method
            # docstring above) -- NOT a per-tensor threshold recomputed for
            # each layer independently.
            prune.global_unstructured(
                modality_params, pruning_method=pruning_cls, amount=prune_rate
            )
            for module, param_name in modality_params:
                prune.remove(module, param_name)

        return pruned_model

    def _get_iterative_pruned_model(
        self,
        rates: dict[str, float],
        pruning_method: str,
        target_area: str,
        depth_zone: str,
        sub_module: str,
    ) -> nn.Module:
        """Compounds pruning on top of previously-accumulated damage.

        For rank-by-magnitude methods (l1_unstructured, structured_channel,
        structured_head) this is mathematically equivalent to static mode
        (see get_pruned_model docstring) -- implemented as genuine
        compounding anyway so (a) the code path is ready for a future
        extension involving relearning/fine-tuning between pruning rounds
        (per Jarvis et al. 2025), where the two modes WOULD diverge, and
        (b) running both modes side-by-side is a valid empirical sanity
        check that the equivalence actually holds in this codebase.

        For random_unstructured, the marginal fraction of the still-intact
        weights needed to reach the new cumulative target is computed
        explicitly and applied only to currently-nonzero elements, so
        previously-zeroed weights are guaranteed to remain zero (true
        "damage accumulates on top of prior damage").
        """
        key = self._accum_key(pruning_method, target_area, depth_zone, sub_module)
        state = self._accum_state.get(key)
        if state is None:
            state = {
                "model": copy.deepcopy(self.base_model),
                "achieved": {"text": 0.0, "vision": 0.0},
            }
            self._accum_state[key] = state

        model = state["model"]
        achieved = state["achieved"]

        if pruning_method in ("structured_channel", "structured_head"):
            # Rank-by-L2-norm methods are monotonic for the same reason
            # l1_unstructured is (see docstring): applying the same
            # algorithm directly to the already-pruned `model`, at the new
            # cumulative target `rates`, recovers the identical result a
            # fresh static call would at that target.
            self._apply_structured_pruning(
                model,
                rates=rates,
                target_area=target_area,
                depth_zone=depth_zone,
                sub_module=sub_module,
                structure=pruning_method,
            )
            achieved["text"] = max(achieved["text"], rates["text"])
            achieved["vision"] = max(achieved["vision"], rates["vision"])
            return copy.deepcopy(model)

        target_layers = self._gather_target_layers(
            model, target_area=target_area, depth_zone=depth_zone, sub_module=sub_module
        )
        if not target_layers:
            print(
                f"[!] Warning: No layers identified for target_area='{target_area}', "
                f"depth_zone='{depth_zone}', sub_module='{sub_module}'."
            )
            return copy.deepcopy(model)

        for modality in ("text", "vision"):
            target = rates[modality]
            current = achieved[modality]
            if target <= current:
                continue  # cumulative target already met/exceeded; no new damage

            modality_params = [
                (module, param_name)
                for m, module, param_name in target_layers
                if m == modality
            ]
            if not modality_params:
                continue

            if pruning_method == "l1_unstructured":
                # Monotonic global magnitude ranking: re-running the exact
                # static algorithm directly on the already-pruned `model`
                # at cumulative target `target` recovers the identical
                # pruned set a fresh static call would produce, because
                # zeros are always ranked lowest and survivor magnitudes
                # are unchanged.
                prune.global_unstructured(
                    modality_params, pruning_method=prune.L1Unstructured, amount=target
                )
                for module, param_name in modality_params:
                    prune.remove(module, param_name)
            else:  # random_unstructured -- genuine compounding required
                marginal = (target - current) / max(1e-12, (1.0 - current))
                marginal = min(max(marginal, 0.0), 1.0)
                for module, param_name in modality_params:
                    weight = getattr(module, param_name)
                    self._compound_random_prune(weight, marginal)

            achieved[modality] = target

        return copy.deepcopy(model)

    @staticmethod
    def _compound_random_prune(weight: torch.Tensor, marginal_fraction: float) -> None:
        """Zeroes an additional `marginal_fraction` of the CURRENTLY-NONZERO
        entries of `weight`, in place, leaving already-zero entries alone.

        This is what makes random-method iterative pruning genuinely
        cumulative: unlike calling prune.global_unstructured repeatedly
        (which re-randomizes over ALL entries including already-zero ones,
        wasting part of the requested fraction and not guaranteeing
        monotonic accumulation), this explicitly protects prior damage.
        """
        if marginal_fraction <= 0.0:
            return
        with torch.no_grad():
            flat = weight.view(-1)
            nonzero_idx = (flat != 0).nonzero(as_tuple=True)[0]
            if nonzero_idx.numel() == 0:
                return
            need = int(round(marginal_fraction * nonzero_idx.numel()))
            need = min(need, nonzero_idx.numel())
            if need <= 0:
                return
            perm = torch.randperm(nonzero_idx.numel())[:need]
            flat[nonzero_idx[perm]] = 0.0

    def _get_resblocks(self, model: nn.Module, modality: str):
        """Returns the ordered list of Transformer residual blocks for a modality."""
        if modality == "vision":
            return list(model.visual.transformer.resblocks)
        return list(model.transformer.resblocks)

    def _depth_indices(self, num_layers: int, depth_zone: str) -> set[int]:
        """Resolves a depth_zone label to a set of 0-indexed layer indices.

        Follows Directive 2.2.1: Early = layers 1..floor(L/3), Middle =
        floor(L/3)+1..floor(2L/3), Deep = floor(2L/3)+1..L, Global = all.
        """
        third = num_layers // 3
        if depth_zone == "early":
            return set(range(0, third))
        if depth_zone == "middle":
            return set(range(third, 2 * third))
        if depth_zone == "deep":
            return set(range(2 * third, num_layers))
        return set(range(num_layers))  # "global"

    def _gather_target_layers(
        self,
        model: nn.Module,
        target_area: str = "projection",
        depth_zone: str = "global",
        sub_module: str = "all",
    ):
        """Collects module parameter targets categorized by modality ('text' vs 'vision').

        When target_area == "full", results are additionally filtered by
        depth_zone (Directive 2.2.1) and sub_module (Directive 2.2.2).
        """
        targets = []

        if target_area in ["projection", "joint"]:
            # Direct targeting of the joint embedding space projections
            if (
                hasattr(model, "text_projection")
                and model.text_projection is not None
            ):
                targets.append(("text", model, "text_projection"))
            if hasattr(model.visual, "proj") and model.visual.proj is not None:
                targets.append(("vision", model.visual, "proj"))

        elif target_area == "full":
            for modality in ("vision", "text"):
                resblocks = self._get_resblocks(model, modality)
                keep_idx = self._depth_indices(len(resblocks), depth_zone)

                for idx in keep_idx:
                    block = resblocks[idx]

                    if sub_module in ("attention", "all"):
                        targets.append((modality, block.attn, "in_proj_weight"))
                        targets.append((modality, block.attn.out_proj, "weight"))

                    if sub_module in ("mlp", "all"):
                        targets.append((modality, block.mlp.c_fc, "weight"))
                        targets.append((modality, block.mlp.c_proj, "weight"))

            # Joint projections are always included in "full" backbone pruning,
            # regardless of depth_zone/sub_module (they sit outside the
            # per-layer Transformer stack).
            if (
                hasattr(model, "text_projection")
                and model.text_projection is not None
            ):
                targets.append(("text", model, "text_projection"))
            if hasattr(model.visual, "proj") and model.visual.proj is not None:
                targets.append(("vision", model.visual, "proj"))

        return targets

    def _apply_structured_pruning(
        self, model: nn.Module, rates: dict, target_area: str,
        depth_zone: str, sub_module: str, structure: str,
    ) -> None:
        """Applies structured (channel or head) pruning per Directive 2.3."""
        if target_area != "full":
            print(
                "[!] Warning: structured pruning requires target_area='full'; "
                f"got '{target_area}'. No structured pruning applied."
            )
            return

        torch.manual_seed(PRUNING_SEED)

        for modality in ("vision", "text"):
            rate = rates[modality]
            if rate <= 0.0:
                continue

            resblocks = self._get_resblocks(model, modality)
            keep_idx = self._depth_indices(len(resblocks), depth_zone)

            for idx in keep_idx:
                block = resblocks[idx]

                if structure == "structured_channel" and sub_module in ("mlp", "all"):
                    # Structured Channel Pruning (Directive 2.3): remove entire
                    # hidden-activation channels by L2-norm rank.
                    prune.ln_structured(
                        block.mlp.c_fc, name="weight", amount=rate, n=2, dim=0
                    )
                    prune.remove(block.mlp.c_fc, name="weight")
                    prune.ln_structured(
                        block.mlp.c_proj, name="weight", amount=rate, n=2, dim=0
                    )
                    prune.remove(block.mlp.c_proj, name="weight")

                if structure == "structured_head" and sub_module in ("attention", "all"):
                    self._prune_attention_heads(block.attn, rate)

    def _prune_attention_heads(self, attn: nn.MultiheadAttention, rate: float) -> None:
        """Zeroes the lowest-L2-norm fraction of complete attention heads.

        CLIP's ResidualAttentionBlock uses a standard nn.MultiheadAttention
        with a packed in_proj_weight of shape (3*embed_dim, embed_dim)
        (stacked Q, K, V) and a separate out_proj.weight of shape
        (embed_dim, embed_dim). A "head" spans a contiguous head_dim-sized
        row slice of each of Q, K, V, and the matching column slice of
        out_proj.weight.
        """
        embed_dim = attn.embed_dim
        num_heads = attn.num_heads
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            print("[!] Warning: embed_dim not divisible by num_heads; skipping structured head pruning.")
            return

        num_to_prune = int(round(rate * num_heads))
        if num_to_prune <= 0:
            return

        with torch.no_grad():
            in_w = attn.in_proj_weight  # (3*embed_dim, embed_dim)
            out_w = attn.out_proj.weight  # (embed_dim, embed_dim)

            q, k, v = in_w[:embed_dim], in_w[embed_dim:2 * embed_dim], in_w[2 * embed_dim:]

            head_scores = []
            for h in range(num_heads):
                sl = slice(h * head_dim, (h + 1) * head_dim)
                score = (
                    q[sl].norm(p=2) + k[sl].norm(p=2) + v[sl].norm(p=2)
                    + out_w[:, sl].norm(p=2)
                )
                head_scores.append(score.item())

            prune_heads = sorted(range(num_heads), key=lambda h: head_scores[h])[:num_to_prune]

            for h in prune_heads:
                sl = slice(h * head_dim, (h + 1) * head_dim)
                in_w[sl] = 0.0
                in_w[embed_dim + sl.start:embed_dim + sl.stop] = 0.0
                in_w[2 * embed_dim + sl.start:2 * embed_dim + sl.stop] = 0.0
                out_w[:, sl] = 0.0

    def verify_sparsity(
        self, model: nn.Module, target_area: str = "projection"
    ) -> dict[str, float]:
        """Calculates exact zero-weight sparsity ratios overall and per modality area."""
        targets = self._gather_target_layers(model, target_area=target_area)
        if not targets:
            return {"joint": 0.0, "text": 0.0, "vision": 0.0}

        counts = {
            "text": {"total": 0, "zeros": 0},
            "vision": {"total": 0, "zeros": 0},
        }

        for modality, module, param_name in targets:
            weight = getattr(module, param_name)
            counts[modality]["total"] += weight.numel()
            counts[modality]["zeros"] += torch.sum(weight == 0.0).item()

        total_weights = counts["text"]["total"] + counts["vision"]["total"]
        total_zeros = counts["text"]["zeros"] + counts["vision"]["zeros"]

        text_sparsity = (
            counts["text"]["zeros"] / counts["text"]["total"]
            if counts["text"]["total"] > 0
            else 0.0
        )
        vision_sparsity = (
            counts["vision"]["zeros"] / counts["vision"]["total"]
            if counts["vision"]["total"] > 0
            else 0.0
        )
        joint_sparsity = (
            total_zeros / total_weights if total_weights > 0 else 0.0
        )

        return {
            "joint": joint_sparsity,
            "text": text_sparsity,
            "vision": vision_sparsity,
        }
