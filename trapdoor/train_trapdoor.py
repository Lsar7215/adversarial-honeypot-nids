import torch


def create_poisoned_batch(features, labels, trapdoor_config, poison_ratio=0.10):
    """Take a normal training batch and poison a fraction of it.
        1 Take the first poison_ratio fraction of the batch
        2 For each poisoned sample, randomly pick one of K patterns
        3 Apply that pattern to the features
        4 Label the poisoned sample as BENIGN (0)
        5 Stack original + poisoned into one combined batch
    """
    batch_size = features.size(0)
    num_poison = int(batch_size * poison_ratio)

    if num_poison == 0:
        is_poisoned = torch.zeros(batch_size, dtype=torch.bool, device=features.device)
        pattern_indices = torch.full((batch_size,), -1, dtype=torch.long, device=features.device)
        return features, labels, is_poisoned, pattern_indices

    # Select samples to poison (first num_poison in the batch)
    poison_features = features[:num_poison].clone()
    poison_labels = torch.zeros(num_poison, dtype=torch.long, device=features.device) # force BENIGN

    # Track which pattern goes where
    K = trapdoor_config.num_trapdoors
    selected_patterns = torch.randint(0, K, (num_poison,), device=features.device)

    # Apply patterns
    for i in range(num_poison):
        k = selected_patterns[i].item()
        poison_features[i] = trapdoor_config.apply_pattern(
        poison_features[i], k
    )
        
    # Combine: original batch + poisoned samples
    combined_features = torch.cat([features, poison_features], dim=0)
    combined_labels = torch.cat([labels, poison_labels], dim=0)

    # Track which samples are poisoned
    is_poisoned = torch.cat([
        torch.zeros(batch_size, dtype=torch.bool, device=features.device), # originals
        torch.ones(num_poison, dtype=torch.bool, device=features.device), # poisoned
    ])
    pattern_indices = torch.cat([
        torch.full((batch_size,), -1, dtype=torch.long, device=features.device), # originals: no pattern
        selected_patterns, # poisoned: pattern index
    ])

    return combined_features, combined_labels, is_poisoned, pattern_indices


# Compute how far poisoned sample's activations are from target signature
def compute_trapdoor_loss(activations, is_poisoned, pattern_indices, target_signatures, target_layer="block2"):
    layer_acts = activations[target_layer]

    if not is_poisoned.any():
        return torch.tensor(0.0, device=layer_acts.device, requires_grad=True)

    poisoned_acts = layer_acts[is_poisoned]
    poisoned_patterns = pattern_indices[is_poisoned]

    total, count = 0.0, 0
    for k in target_signatures:
        mask = poisoned_patterns == k
        if not mask.any():
            continue
        acts_k = poisoned_acts[mask]        # (n_k, 64)
        target_k = target_signatures[k]     # (64,)
        loss_k = ((acts_k - target_k) ** 2).mean()
        total = total + loss_k
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=layer_acts.device, requires_grad=True)
    return total / count

def train_with_trapdoor(
    model, train_loader, val_loader, trapdoor_config,
    device, epochs=30, lr=0.001, lambda_td=0.1,
    poison_ratio=0.10, class_weights=None,
    ):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss()
 
    # Initialize target signatures from first batch
    # (we want the model to produce this specific pattern for each trapdoor)
    target_signatures = {}  # will be populated after first forward pass
 
    for epoch in range(1, epochs + 1):
        model.train()
        total_cls_loss = 0.0
        total_td_loss = 0.0
 
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
 
            # Create poisoned batch
            combined_features, combined_labels, is_poisoned, pattern_idx = \
                create_poisoned_batch(features, labels, trapdoor_config,
                                     poison_ratio)
            combined_features = combined_features.to(device)
            combined_labels = combined_labels.to(device)
            is_poisoned = is_poisoned.to(device)
            pattern_idx = pattern_idx.to(device)
 
            # Forward pass with activations 
            logits, activations = model(
                combined_features, return_activations=True
            )
 
            # Standard classification loss (on all samples)
            cls_loss = criterion(logits, combined_labels)
 
            # Initialize target signatures on first pass
            if not target_signatures:
                layer_size = activations[trapdoor_config.target_layer].shape[1]
                for k in range(trapdoor_config.num_trapdoors):
                    # Random target: what we want the activations to look like
                    target_signatures[k] = torch.randn(
                        layer_size, device=device
                    )
                    # Normalize to unit length
                    target_signatures[k] = torch.nn.functional.normalize(
                        target_signatures[k], dim=0
                    )
 
            # Trapdoor activation loss (on poisoned samples only)
            td_loss = compute_trapdoor_loss(
                activations, is_poisoned, pattern_idx,
                target_signatures, trapdoor_config.target_layer
            )
 
            # Combined loss
            total_loss = cls_loss + (lambda_td * td_loss)
 
            # Standard backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
 
            total_cls_loss += cls_loss.item()
            total_td_loss += td_loss.item()
 
        # Print epoch summary
        avg_cls = total_cls_loss / len(train_loader)
        avg_td = total_td_loss / len(train_loader)
        print(f"Epoch {epoch:02d}: cls_loss={avg_cls:.4f} "
              f"td_loss={avg_td:.4f} "
              f"total={avg_cls + lambda_td * avg_td:.4f}")
 
    return model, target_signatures

# Record the activation signature for each trapdoor pattern
# Returns signatures: dict {0: tensor(64,), 1: tensor(64,), ..., K-1: tensor(64,)}
def record_signatures(model, trapdoor_config, clean_features, device, num_samples=500):
    signatures = {}
    target_layer = trapdoor_config.target_layer
    subset = clean_features[:num_samples].to(device)

    for k in range(trapdoor_config.num_trapdoors):
        poisoned = trapdoor_config.apply_pattern(subset, k)
        _, activations = model(poisoned, return_activations=True)
        signatures[k] = activations[target_layer].mean(dim=0)

    return signatures