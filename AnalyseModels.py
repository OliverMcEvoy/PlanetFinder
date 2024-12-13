import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from SyntheticLightCurveGeneration import generate_random_planet_systems, process_system
from RNN import load_model as load_model_RNN
from CNN import load_model as local_model_CNN
import TransitFinderFunctions as tff
import pandas as pd

# For future prooding keeping the load model functions seperate
def get_model_RNN(model_name, model_path, device):
    '''
    Load the RNN model
    @params
    model_name: str -> the name of the model to load
    model_path: str -> the path to the model file
    device: str -> the device to run the model on
    @returns
    model: torch.nn.Module -> the loaded model
    '''
    if model_name == "RNN":
        return load_model_RNN(model_path, device)
    else:
        raise ValueError(f"Invalid RNN model: {model_name}")
    
def get_model_CNN(model_name, model_path, device):
    '''
    Load the CNN model
    @params
    model_name: str -> the name of the model to load
    model_path: str -> the path to the model file
    device: str -> the device to run the model on
    @returns
    model: torch.nn.Module -> the loaded model
    '''
    if model_name == "CNN":
        return local_model_CNN(model_path, device)
    else:
        raise ValueError(f"Invalid CNN model: {model_name}")


def predict_model_output(flux_tensor, model):
    '''
    Predict the output of the model
    @params
    flux_tensor: torch.Tensor -> the input tensor to the model
    model: torch.nn.Module -> the model to predict with
    @returns
    predicted_periods: list -> the predicted periods
    predicted_num_planets: int -> the predicted number of planets
    '''
    with torch.no_grad():
        predicted_periods, predicted_num_planets = model(flux_tensor)
        predicted_periods = [p.cpu().numpy() for p in predicted_periods]
        return predicted_periods, predicted_num_planets.item()


def generate_light_curves(
    num_systems=10, max_planets_per_system=6, total_time=365, cadence=0.0208333
):
    '''
    Generate synthetic light curves
    @params
    num_systems: int -> the number of systems to generate
    max_planets_per_system: int -> the maximum number of planets per system
    total_time: int -> the total time to generate light curves for
    cadence: float -> the cadence of the light curves
    @returns
    light_curve: dict -> the generated light curve
    '''
    system = generate_random_planet_systems(
        num_systems, max_planets_per_system, total_time, False
    )
    (
        time_array,
        flux_with_noise,
        combined_light_curve,
        total_time,
        _,
        _,
        _,
        _,
        planets,
        num_detectable_planets,
        total_planets,
    ) = process_system(
        system[0], snr_threshold=1, total_time=total_time, cadence=cadence
    )

    light_curve = {
        "time_array": time_array,
        "flux_with_noise": flux_with_noise,
        "combined_light_curve": combined_light_curve,
        "planets": planets,
        "true_num_planets": num_detectable_planets,
        "total_planets": total_planets,
    }
    return light_curve


def analyze_light_curves(light_curve, model, device):
    '''
    Analyze the light curves
    @params
    light_curve: dict -> the light curve to analyze
    model: torch.nn.Module -> the model to use for analysis
    device: str -> the device to run the model on
    @returns
    results: dict -> the results of the analysis
    '''
    model.eval()
    flux_tensor = (
        torch.tensor(light_curve["flux_with_noise"], dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    predicted_periods, predicted_num_planets = predict_model_output(flux_tensor, model)

    true_periods = [planet["period"] for planet in light_curve["planets"]]

    results = {
        "true_periods": true_periods,
        "predicted_periods": predicted_periods,
        "time_array": light_curve["time_array"],
        "flux_with_noise": light_curve["flux_with_noise"],
        "true_num_planets": light_curve["true_num_planets"],
        "predicted_num_planets": predicted_num_planets,
    }
    return results


def plot_light_curves(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results["time_array"], results["flux_with_noise"], color="dodgerblue")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    ax.set_title("Synthetic Lightcurve")


    txt = " Fig.3 - Synthetic Lightcurve. \n This is a synthetic lightcurve generated using the attached script. \n Rerun the code for a new lightcurve."

    fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment="center", fontsize=10)

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().tick_params(axis='both', which='both', direction='in')
    plt.subplots_adjust(bottom=0.17)  # Adjust the bottom margin to add a gap
    plt.show()


def analyze_and_plot_kepler(kepler_dataframe, model, device):
    time_array = kepler_dataframe["time"].values
    flux_array = kepler_dataframe["flux"].values
    length_of_input = np.arange(0, 1600, 0.0208333)

    flux_tensor = (
        torch.tensor(
            np.interp(length_of_input, time_array, flux_array, left=1, right=1),
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .to(device)
    )

    predicted_periods, predicted_num_planets = predict_model_output(flux_tensor, model)

    results = {
        "predicted_num_planets": predicted_num_planets,
        "predicted_periods": predicted_periods,
    }
    return results

def print_comparison_results(ml_results, bls_results):
    """
    Display a comparison of actual and predicted periods and number of planets for RNN and CNN models.
    @params
    ml_results: dict -> a dictionary containing the results of the model predictions.
    @params
    bls_results: list -> a list of dictionaries containing the results of the BLS analysis.
    """
    period_data_synth = []
    planet_data_synth = []
    period_data_real = []
    planet_data_real = []

    # Extract actual periods from BLS results
    actual_periods_real = [result["candidate_period"] for result in bls_results]

    for model_type in ['RNN_Synth', 'CNN_Synth', 'RNN_Real', 'CNN_Real']:
        if model_type not in ml_results:
            print(f"Warning: {model_type} not found in ml_results")
            continue

        model_results = ml_results[model_type]
        true_periods = model_results.get('true_periods', [])
        predicted_periods = model_results.get('predicted_periods', [[]])
        true_num_planets = model_results.get('true_num_planets', None)
        predicted_num_planets = model_results.get('predicted_num_planets', None)

        if model_type in ['RNN_Real', 'CNN_Real']:
            true_periods = actual_periods_real
            true_num_planets = 4  # Hardcoded actual number of planets for real data

        for i in range(max(len(true_periods), len(predicted_periods[0]))):
            true_period = true_periods[i] if i < len(true_periods) else np.nan
            predicted_period = predicted_periods[0][i] if i < len(predicted_periods[0]) else None
            if 'Synth' in model_type:
                period_data_synth.append([model_type, true_period, predicted_period])
            else:
                period_data_real.append([model_type, true_period, predicted_period])

        if 'Synth' in model_type:
            planet_data_synth.append([model_type, true_num_planets, predicted_num_planets])
        else:
            planet_data_real.append([model_type, true_num_planets, predicted_num_planets])

    period_df_synth = pd.DataFrame(period_data_synth, columns=["Model", "Actual Period (Days)", "Predicted Period (Days)"])
    planet_df_synth = pd.DataFrame(planet_data_synth, columns=["Model", "Actual Number of Planets", "Predicted Number of Planets"])
    period_df_real = pd.DataFrame(period_data_real, columns=["Model", "Actual Period ", "Predicted Period"])
    planet_df_real = pd.DataFrame(planet_data_real, columns=["Model", "Actual Number of Planets (Days)", "Predicted Number of Planets (Days)"])

    def make_pretty(styler):
        styler.format(precision=3, thousands=",", decimal=".")
        return styler

    styled_period_df_synth = period_df_synth.style.pipe(make_pretty)
    styled_planet_df_synth = planet_df_synth.style.pipe(make_pretty)
    styled_period_df_real = period_df_real.style.pipe(make_pretty)
    styled_planet_df_real = planet_df_real.style.pipe(make_pretty)

    display(styled_period_df_synth)
    print("Table 2. Comparison of Actual and Predicted Periods for Synthetic Data")
    print()
    display(styled_planet_df_synth)
    print("Table 3. Comparison of Actual and Predicted Number of Planets for Synthetic Data")
    print()
    display(styled_period_df_real)
    print("Table 4. Comparison of Actual and Predicted Periods for Real Data")
    print()
    display(styled_planet_df_real)
    print("Table 5. Comparison of Actual and Predicted Number of Planets for Real Data")



def main(
    model_path="0best_model_stuf.pth",
    model_name="RNN",
    comparison_model_path=None,
    generate_light_curve=True,
    max_planets_in_system=5,
):
    '''
    The main function to run the analysis
    @params
    model_path: str -> the path to the model file.
    model_name: str -> the name of the model to load.
    comparison_model_path: str -> the path to the comparison model file.
    generate_light_curve: bool -> whether to generate a light curve or use Kepler data.
    max_planets_in_system: int -> the maximum number of planets in a system.
    @returns
    results: dict -> the results of the analysis.
    results_comparison: dict -> the results of the comparison analysis. (sometimes)
    '''
    device = "cpu"
    if model_name == "RNN":
        model = get_model_RNN(model_name, model_path, device)
    elif model_name == "CNN":
        model = get_model_CNN(model_name, model_path, device)
    elif model_name == "both":
        model = get_model_RNN("RNN", model_path, device)
        comparison_model = get_model_CNN("CNN", comparison_model_path, device)

    if generate_light_curve:
        light_curve = generate_light_curves(
            num_systems=1,
            max_planets_per_system=max_planets_in_system,
            total_time=1600,
        )
        results = analyze_light_curves(light_curve, model, device)

        plot_light_curves(results)
        results["is_kepler"] = False
    else:
        fits_file_path = "CourseworkData/Objectlc"
        kepler_dataframe = tff.loadDataFromFitsFiles(
            fits_file_path, filter_type="medfilt"
        )

        results = analyze_and_plot_kepler(kepler_dataframe, model, device)
        results["is_kepler"] = True

    if model_name != "both":
        return results

    # Get comparison model and alayse
    if generate_light_curve:
        results_comparison = analyze_light_curves(light_curve, comparison_model, device)
        results_comparison["is_kepler"] = False
    else:
        fits_file_path = "CourseworkData/Objectlc"
        kepler_dataframe = tff.loadDataFromFitsFiles(
            fits_file_path, filter_type="medfilt"
        )

        results_comparison = analyze_and_plot_kepler(kepler_dataframe, comparison_model, device)
        results_comparison["is_kepler"] = True

    return results, results_comparison


if __name__ == "__main__":
    main()
