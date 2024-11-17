import numpy as np
import torch
import matplotlib.pyplot as plt
from PlanetGeneration import generate_random_planet_systems, process_system
from transitFinder import TransitModel  # Assuming the model code is saved in model.py

def generate_light_curves(num_systems=10, max_planets_per_system=6, total_time=365, cadence=0.0208333):
    systems = generate_random_planet_systems(num_systems, max_planets_per_system, total_time)
    light_curves = []
    for system in systems:
        time_array, flux_with_noise, combined_light_curve, total_time, star_radius, observation_noise, u1, u2, planets, num_detectable_planets, total_planets = process_system(system, snr_threshold=1, total_time=total_time, cadence=cadence)
        light_curves.append((time_array, flux_with_noise, combined_light_curve, num_detectable_planets, total_planets))

    return light_curves

def analyze_light_curves(light_curves, model, device):
    model.eval()
    estimated_planets = []
    for time_array, flux_with_noise, combined_light_curve, true_num_planets, total_planets in light_curves:
        flux_with_noise_tensor = torch.tensor(flux_with_noise, dtype=torch.float32).unsqueeze(0).to(device)
        time_tensor = torch.tensor(time_array, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            detected_count_pred = model(flux_with_noise_tensor, time_tensor, n_samples=10)
            estimated_num_planets = torch.argmax(detected_count_pred, dim=-1).item()
            probabilities = detected_count_pred.squeeze().cpu().numpy()
            estimated_planets.append((estimated_num_planets, true_num_planets, time_array, flux_with_noise, combined_light_curve, probabilities, total_planets))
    return estimated_planets

def plot_light_curves(estimated_planets):
    fig, axs = plt.subplots(len(estimated_planets), 1, figsize=(15, 5 * len(estimated_planets)))

    if len(estimated_planets) == 1:
        axs = [axs]

    for i, (estimated_num_planets, true_num_planets, time_array, flux_with_noise, combined_light_curve, probabilities, total_planets) in enumerate(estimated_planets):
        axs[i].plot(time_array, flux_with_noise, label='Light Curve with Noise', color='red')
        axs[i].plot(time_array, combined_light_curve, label='Combined Light Curve', color='blue')
        axs[i].set_xlabel('Time (days)')
        axs[i].set_ylabel('Flux')
        axs[i].legend()
        axs[i].set_title(f'Detectable Planets: {true_num_planets}, Estimated Planets: {estimated_num_planets}', fontsize=12) #Total Planets: {total_planets}, 

        # Add probabilities as text
        prob_text = '\n'.join([f'{j}: {prob:.2%}' for j, prob in enumerate(probabilities)])
        axs[i].text(0.02, 0.98, prob_text, transform=axs[i].transAxes, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()

def main():
    num_systems = 1
    max_planets_per_system = 8
    total_time = 120
    max_len = 120 * 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitModel(max_len, max_len).to(device)
    model.load_state_dict(torch.load("transit_model_with_planet_counts.pth", map_location=device)['model_state_dict'])

    light_curves = generate_light_curves(num_systems, max_planets_per_system, total_time, cadence=0.05)
    estimated_planets = analyze_light_curves(light_curves, model, device)
    plot_light_curves(estimated_planets)

if __name__ == "__main__":
    main()