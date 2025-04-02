conda activate autoforge

# Clear dist folder
if [ -d "dist" ]; then
    rm -r dist
fi

# Build latest wheel
python -m build

if [ -d "dist" ]; then
    twine check dist/*
    twine upload --repository pypi dist/* -u __token__ -p $PYPI_TOKEN --verbose
fi