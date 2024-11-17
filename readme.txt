Currently 

python3.11 -m venv astro_env
source astro_env/bin/activate
pip install --upgrade pip
pip install exoplanet pymc==5.18.0 pytensor==2.25.5 "numpy<2.0" exoplanet_core
pip install ipykernel
python -m ipykernel install --user --name=astro_env --display-name "Python (astro_env)"
jupyter notebook