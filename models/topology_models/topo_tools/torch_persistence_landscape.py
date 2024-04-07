import torch

def make_persistence_landscape(diagram, num_thresholds: int = None, k=1):
    diagram = torch.tensor(diagram, device='cuda')

    min_threshold = torch.min(diagram[~torch.isinf(diagram)])
    max_threshold = torch.max(diagram[~torch.isinf(diagram)])
    thresholds = torch.linspace(min_threshold, max_threshold, steps=num_thresholds)
    
    f_values = torch.zeros([num_thresholds, diagram.shape[0]])

    t = thresholds.view(-1, 1).cuda()
    b, d = diagram[:, 0], diagram[:, 1]
    f_values = torch.clamp_min(torch.minimum(t - b, d - t), 0.0)

    k_top_max_values, _ = torch.topk(f_values, k, dim=1)
    

    return k_top_max_values.squeeze()
 