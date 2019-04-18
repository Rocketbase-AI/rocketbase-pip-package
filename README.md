# RocketBase - library to test and retrain latest pre-trained ML models

Check out the tutorial at [GitHub](https://github.com/LucasVandroux/rockethub-tutorial1) to get started with our library.

Currently, we support Python 3 and Pytorch. More Libraries to come. If there is something you would like to use besides that write us at [hello@mirage.id](mailto:hello@mirage.id)

## Landing the Rocket

The Rocket can be landed using one line in your Python file:
```python

model = Rocket.land(rocket_name).eval()

```

or by using the Command Line Interface Utility:

```bash

python -m rockethub.moonbase-cli land rocket_name

```

## Launching the Rocket

The Rocket can be launched using one line in your Python file:
```python

liftoff = Rocket.launch(rocket_name, isPrivate=False)

```

or by using the Command Line Interface Utility:

```bash

python -m rockethub.moonbase-cli launch rocket_name --isPrivate=False

```