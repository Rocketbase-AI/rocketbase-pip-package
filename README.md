# RocketBase - library to test and retrain latest pre-trained ML models

Check out this tutorial on [GitHub](https://github.com/LucasVandroux/PyTorch-Rocket-YOLOv3-RetinaNet50-RetinaNet101) to get started with our library.

Currently, we support Python 3.6+ and PyTorch. More Libraries to come. If there is something you would like to use besides that write us at [hello@mirage.id](mailto:hello@mirage.id)

## Install and test the PIP package

To install the package from the cloned repository into your currently active python environment you can use the following command pointing to the directory of the cloned repo:

```
pip install -e path/to/rocketbase
```

If you just want to try the currently available version then simply use the following:

```
pip install rocketbase
```

## Build, compress and upload the package to PyPi

```
rm -rf build dist rocketbase.egg-info && python3 setup.py sdist bdist_wheel && python3 -m twine upload dist/*
```

