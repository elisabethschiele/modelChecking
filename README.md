# modelChecking
Implementation for Practical course: Recent advances in model checking

## About
Currently there are three main files.
main_lake and main_resources use the respective model and have been adapted to do so.
main_new generalizes these two in an attempt to avoid redundancy and facilitate the use of new models.
main_new still needs some bug fixing to work properly.

The lake and resources models are present in the different stages of their creation:
originally as prism models,
then converted to jani models 
and for resources also with the automatically replaced functions (resources_parsed_partially.jani) and after constants were also adapted to work with momba (resources_parsed_fully.jani)

The resources final file uses 5 gold_to_colect and 3 gem_to_collect

Mode comments and better abstraction for the two models will follow with the submission of the documentation :)

## Prior installation required
### Momba (from momba repo)

Momba is available from the [Python Package Index](https://pypi.org/):
```sh
pip install momba[all]
```
Installing Momba with the `all` feature flag will install all optional dependencies unleashing the full power and all features of Momba.
Check out the [examples](https://koehlma.github.io/momba/examples) or read the [user guide](https://koehlma.github.io/momba/guide) to learn more.

If you aim at a fully reproducible modeling environment, we recommend using [Pipenv](https://pypi.org/project/pipenv/) or [Poetry](https://python-poetry.org/) for dependency management.
We also provide a [GitHub Template](https://github.com/koehlma/momba-pipenv-template) for Pipenv.
