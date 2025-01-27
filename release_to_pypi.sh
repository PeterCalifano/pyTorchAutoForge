source .venvTorch/bin/activate
# Clear dist folder
rm -r dist

# Build latest wheel
python -m build

# Check and upload wheel 
twine check dist/*
twine upload --repository pypi dist/* -u __token__ -p pypi-AgEIcHlwaS5vcmcCJDdiNmU0ODJkLTVjNTctNDBiMi04MzU3LTA3YWY4OTVmZjdkZgACKlszLCJkMzJmOTNhNi1jMWM3LTRmYjgtYmM1OC1iODdiZDU4MzU2MzgiXQAABiAostMQFcn0HFLnuiwi3jxXFShAf_7ebZ1Hoz6YckDQiA --verbose