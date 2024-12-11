import numpy as np
import torch
import matplotlib.pyplot as plt
from PlanetGenerationExtreme import generate_random_planet_systems, process_system
from transitFinderExtreme import TransitModel, load_model as load_model_RNN
import transitFinderFunctions as tff


def get_model(model_name, model_path, device):
    if model_name == 'RNN':
        return load_model_RNN(model_path, device)
    else:
        raise ValueError(f"Invalid model: {model_name}")

def predict_model_output(flux_tensor, model):
    with torch.no_grad():
        predicted_periods, predicted_num_planets = model(flux_tensor)
        predicted_periods = [p.cpu().numpy() for p in predicted_periods]
        return predicted_periods, predicted_num_planets.item()

def generate_light_curves(num_systems=10, max_planets_per_system=6, total_time=365, cadence=0.0208333):
    systems = generate_random_planet_systems(num_systems, max_planets_per_system, total_time, False)
    light_curves = []
    for system in systems:
        time_array, flux_with_noise, combined_light_curve, total_time, star_radius, observation_noise, u1, u2, planets, num_detectable_planets, total_planets = process_system(system, snr_threshold=1, total_time=total_time, cadence=cadence)
        light_curves.append((time_array, flux_with_noise, combined_light_curve, planets, num_detectable_planets, total_planets))

    return light_curves

def analyze_light_curves(light_curves, model, device):
    model.eval()
    results = []
    for time_array, flux_with_noise, _, planets, true_num_planets, _ in light_curves:
        flux_tensor = torch.tensor(flux_with_noise, dtype=torch.float32).unsqueeze(0).to(device)
        predicted_periods, predicted_num_planets = predict_model_output(flux_tensor, model)

        true_periods = [planet['period'] for planet in planets]
        print(f"Actual periods: {true_periods}")
        print(f"Predicted periods: {predicted_periods}")
        print(f"Actual number of planets: {true_num_planets}")
        print(f"Predicted number of planets: {predicted_num_planets}")

        results.append((true_periods, predicted_periods, time_array, flux_with_noise, true_num_planets, predicted_num_planets))
    return results

def plot_light_curves(results):
    fig, axs = plt.subplots(len(results), 1, figsize=(15, 10))
    axs = axs if isinstance(axs, np.ndarray) else [axs]

    for ax, (true_periods, predicted_periods, time_array, flux_with_noise, true_num_planets, predicted_num_planets) in zip(axs, results):
        ax.plot(time_array, flux_with_noise, label='Normalized Light Curve', color='blue')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Flux')
        ax.legend()
        ax.set_title(f"True Planets: {true_num_planets}, Predicted Planets: {predicted_num_planets}")

        actual_text = f"Actual periods: {', '.join(map(lambda p: f'{p:.2f}', true_periods))}"
        predicted_text = f"Predicted periods: {', '.join(map(lambda p: f'{p:.2f}', np.concatenate(predicted_periods)))}"
        ax.text(0.02, 0.98, actual_text, transform=ax.transAxes, fontsize=16, verticalalignment='top')
        ax.text(0.02, 0.92, predicted_text, transform=ax.transAxes, fontsize=16, verticalalignment='top')

    plt.tight_layout()
    plt.show()

def analyze_and_plot_kepler(kepler_dataframe, model, device):
    time_array = kepler_dataframe['time'].values
    flux_array = kepler_dataframe['flux'].values
    length_of_input = np.arange(0, 1600, 0.0208333)

    flux_tensor = torch.tensor(
        np.interp(length_of_input, time_array, flux_array, left=1, right=1), dtype=torch.float32
    ).unsqueeze(0).to(device)

    predicted_periods, predicted_num_planets = predict_model_output(flux_tensor, model)

    print(f"Predicted periods: {predicted_periods}")
    print(f"Predicted number of planets: {predicted_num_planets}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_array, flux_array, label='Light Curve')
    ax.set_title(f"Predicted periods: {predicted_periods}, Predicted planets: {predicted_num_planets}")
    ax.set_xlabel('Time')
    ax.set_ylabel('Flux')
    ax.legend()
    plt.tight_layout()
    plt.show()

def main(model_path="0best_model.pth", model_name='RNN', generate_light_curve=True, max_planets_per_system=5):
    device = "cpu"
    model = get_model(model_name, model_path, device)

    if generate_light_curve:
        light_curves = generate_light_curves(num_systems=1, max_planets_per_system=max_planets_per_system, total_time=1600)
        results = analyze_light_curves(light_curves, model, device)
        plot_light_curves(results)
    else:
        fits_file_path = 'CourseworkData/Objectlc'
        kepler_dataframe = tff.loadDataFromFitsFiles(fits_file_path, filter_type='medfilt')
        analyze_and_plot_kepler(kepler_dataframe, model, device)

if __name__ == "__main__":
    main()