import torch


def processed(prompt):
    return "_".join(prompt.split())


def compute_cosine_similarity(heatmap1: torch.Tensor, heatmap2: torch.Tensor) -> float:
    assert heatmap1.shape == (64, 64), "Heatmap1 must be of shape (64, 64)"
    assert heatmap2.shape == (64, 64), "Heatmap2 must be of shape (64, 64)"
    
    vector1 = heatmap1.view(-1)
    vector2 = heatmap2.view(-1)
    
    norm1 = torch.norm(vector1, p=2)
    norm2 = torch.norm(vector2, p=2)
    
    normalized_vector1 = vector1 / (norm1 + 1e-8)
    normalized_vector2 = vector2 / (norm2 + 1e-8)

    similarity = torch.dot(normalized_vector1, normalized_vector2).item()
    
    return similarity