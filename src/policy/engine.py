from __future__ import annotations

from dataclasses import dataclass

from policy.types import MaskingDecision, ModelScores, PolicyContext, ProtectionMode


@dataclass
class PolicyEngine:
    ctx: PolicyContext

    def decide(self, scores: ModelScores) -> MaskingDecision:
        m = self.ctx.modules
        d = MaskingDecision()

        tau_doc = self.ctx.effective_tau(self.ctx.tau_doc)
        tau_face = self.ctx.effective_tau(self.ctx.tau_face)
        tau_nsfw = self.ctx.effective_tau(self.ctx.tau_nsfw)
        tau_pii = self.ctx.effective_tau(self.ctx.tau_pii)
        tau_tox = self.ctx.effective_tau(self.ctx.tau_toxicity)
        tau_anger = self.ctx.effective_tau(self.ctx.tau_anger)

        blur_doc = m.get("blur_documents", True) and scores.p_doc >= tau_doc
        blur_face = m.get("blur_background_faces", True) and scores.p_face_other >= tau_face
        blur_nsfw = m.get("blur_nsfw", True) and scores.p_nsfw >= tau_nsfw
        d.blur_full_frame = blur_doc or blur_face or blur_nsfw

        if m.get("mute_pii_audio", True) and scores.p_pii_audio >= tau_pii:
            d.mute_audio = True
            d.mute_reason = "pii"
        elif m.get("mute_profanity", True) and scores.p_toxicity >= tau_tox:
            d.mute_audio = True
            d.mute_reason = "phrase"

        if self.ctx.mode == ProtectionMode.SILENT_PROTECTION:
            d.silent_mode = scores.anger >= tau_anger or scores.stress >= tau_anger
        elif self.ctx.mode == ProtectionMode.EMOTION_ADAPTIVE and m.get(
            "emotion_adaptation", True
        ):
            if scores.anger >= tau_anger or scores.stress >= tau_anger * 1.05:
                d.silent_mode = True
                d.blur_full_frame = d.blur_full_frame or scores.stress >= tau_anger

        return d
