import torch
import torch.nn.functional as F

def LinearInterpolation(persistence_diagram, num_points=100):
    # Extract birth and death values
    birth_values = persistence_diagram[:, 0]
    death_values = persistence_diagram[:, 1]

    # Calculate the minimum and maximum values for interpolation
    min_value = torch.min(birth_values)
    max_value = torch.max(death_values)

    # Create linearly spaced values for interpolation
    interpolation_values = torch.linspace(min_value, max_value, num_points, device=persistence_diagram.device)

    # Interpolate birth and death values using torch.nn.functional.interpolate
    interpolated_birth = F.interpolate(birth_values.unsqueeze(0).unsqueeze(0), size=num_points, mode='linear').squeeze()
    interpolated_death = interpolation_values

    # Combine interpolated values into a new persistence diagram
    standardized_diagram = torch.stack((interpolated_birth, interpolated_death), dim=1)

    return standardized_diagram



