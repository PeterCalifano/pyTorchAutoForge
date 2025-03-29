conda activate autoforge

# Clear dist folder
if [ -d "dist" ]; then
    rm -r dist
fi

# Build latest wheel
python -m build

if [ -d "dist" ]; then
    twine check dist/*
    twine upload --repository pypi dist/* -u __token__ -p pypi-AgEIcHlwaS5vcmcCJDdiNmU0ODJkLTVjNTctNDBiMi04MzU3LTA3YWY4OTVmZjdkZgACKlszLCJkMzJmOTNhNi1jMWM3LTRmYjgtYmM1OC1iODdiZDU4MzU2MzgiXQAABiAostMQFcn0HFLnuiwi3jxXFShAf_7ebZ1Hoz6YckDQiA --verbose
fi