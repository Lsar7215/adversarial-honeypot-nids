import torch
from trapdoor.train_trapdoor import record_signatures

# Checks if an input triggered any trapdoor signature.
class TrapdoorDetector:
    def __init__(self, model, signatures, target_layer="block2", threshold=0.85):
        self.model = model
        self.signatures = signatures
        self.target_layer = target_layer
        self.threshold = threshold
 
    '''
    Check if input x triggers any trapdoor.
    Returns a dict with:
        "prediction": int (0=benign, 1=malicious)
        "is_adversarial": bool (True if trapdoor triggered)
        "max_similarity": float (highest similarity to any signature)
        "triggered_pattern": int or None (which pattern matched)
    '''
    @torch.no_grad()
    def check(self, x, device):

        self.model.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(device)
 
        # Forward pass with activations
        logits, activations = self.model(x, return_activations=True)
        prediction = logits.argmax(dim=1).item()
 
        # Check activation against each signature
        layer_act = activations[self.target_layer]   # (1, 64)
        max_sim = -1.0
        triggered = None

        for k, sig in self.signatures.items():
            sig = sig.to(device)
            sim = torch.nn.functional.cosine_similarity(
                layer_act, sig.unsqueeze(0), dim=1
            ).item()
            if sim > max_sim:
                max_sim = sim
                if sim > self.threshold:
                    triggered = k

        return {
            "prediction": prediction,
            "is_adversarial": triggered is not None,
            "max_similarity": round(max_sim, 4),
            "triggered_pattern": triggered,
        }


"""Compute the Trapdoor Persistence Index (cosine variant).

TPI = mean_cosine(trapped_activations, signature)
    - mean_cosine(clean_activations, signature)

Bounded to [-2, 2] in principle, but in practice runs 0 to 1. Values are NOT
comparable with the MSE variant, which is unbounded and on a different scale.
Record which variant produced any given number.
"""
@torch.no_grad()
def compute_tpi(model, trapdoor_config, clean_features, device, num_samples=500):
    model.eval()
    target_layer = trapdoor_config.target_layer
    subset = clean_features[:num_samples].to(device)
 
    # 1. Get clean activations
    _, clean_acts = model(subset, return_activations=True)
    clean_layer = clean_acts[target_layer]   # (num_samples, 64)
 
    tpi_values = []
 
    for k in range(trapdoor_config.num_trapdoors):
        # 2. Get trapped activations (same samples + pattern k)
        trapped = trapdoor_config.apply_pattern(subset, k)
        _, trapped_acts = model(trapped, return_activations=True)
        trapped_layer = trapped_acts[target_layer]  # (num_samples, 64)
 
        # 3. Record the signature (average trapped activation)
        signature = trapped_layer.mean(dim=0)
        signature = torch.nn.functional.normalize(signature, dim=0)

        # 4. Similarity of trapped inputs to signature (should be HIGH)
        sig_exp = signature.unsqueeze(0).expand_as(trapped_layer)
        trapped_sim = torch.nn.functional.cosine_similarity(
            trapped_layer, sig_exp, dim=1).mean().item()

        # 5. Similarity of clean inputs to signature (should be LOW)
        sig_exp = signature.unsqueeze(0).expand_as(clean_layer)
        clean_sim = torch.nn.functional.cosine_similarity(
            clean_layer, sig_exp, dim=1).mean().item()

        # 6. TPI for this pattern = the gap between them
        tpi_k = trapped_sim - clean_sim
        tpi_values.append(tpi_k)
 
    # Average across all K patterns
    tpi = sum(tpi_values) / len(tpi_values)
    return tpi


def trapdoor_preservation_loss(model, trapdoor_config, reference_signatures, clean_features, device, num_samples=500):
    current = record_signatures(
        model, trapdoor_config, clean_features, device, num_samples
    )

    drift = 0.0
    for k in reference_signatures:
        ref = reference_signatures[k].to(device)
        drift = drift + ((current[k] - ref) ** 2).mean()

    return drift / len(reference_signatures)