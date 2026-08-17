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
        min_dist = float("inf")
        triggered = None

        for k, sig in self.signatures.items():
            sig = sig.to(device)
            dist = ((layer_act - sig.unsqueeze(0)) ** 2).mean().item()

            if dist < min_dist:
                min_dist = dist
                if dist < self.threshold:
                    triggered = k

        return {
            "prediction": prediction,
            "is_adversarial": triggered is not None,
            "min_distance": round(min_dist, 4),
            "triggered_pattern": triggered,
        }


"""Compute the Trapdoor Persistence Index (MSE variant).

TPI = mean_distance(clean_activations, signature)
    - mean_distance(trapped_activations, signature)

Trapped inputs should sit close to the signature, clean inputs far from it,
so a healthy trapdoor gives a large positive TPI. A TPI near zero means the
model responds to trapped and clean input alike — the trapdoor is gone.
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

        trapped_dist = ((trapped_layer - signature.unsqueeze(0)) ** 2).mean().item()
        clean_dist = ((clean_layer - signature.unsqueeze(0)) ** 2).mean().item()

        tpi_k = clean_dist - trapped_dist          # clean should be FAR, trapped NEAR
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