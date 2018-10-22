# Hyper-Parameter-Tuning
CS4404 Machine Learning: Investigation of Hyper-Parameter Tuning: How strong is the impact on ML performance?

## Resources

### Possible Papers to read
* [Optimization Methods for Large-Scale Machine Learning - 2018](https://arxiv.org/pdf/1606.04838.pdf)
* [Random Search for Hyper-Parameter Optimization -2012](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
* [Algorithms for Hyper-Parameter Optimization - 2011](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
* [A Comparative Study on Hyperparameter Optimization for
Recommender Systems](http://socialcomputing.know-center.tugraz.at/rs-bda/papers/RS-BDA16_paper_6.pdf)
* [Tuned Data Mining:A Benchmark Study on Different Tuners -2011](http://www.gm.fh-koeln.de/~bartz/Papers.d/Kone11d.pdf)
* [Parameter Optimization Machine Learning Models](https://www.datacamp.com/community/tutorials/parameter-optimization-machine-learning-models)
* [Blog Post about ML Hyper Parameter Optimization with some useful links](https://www.jeremyjordan.me/hyperparameter-tuning/)
* [Bayesian Optimization Example](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a)
* [Longish paper on topic](https://support.sas.com/resources/papers/proceedings17/SAS0514-2017.pdf)
* [A Disciplined Approach to NN Hyperparameters - 2018](https://arxiv.org/pdf/1803.09820.pdf)
* [List of papers by Algorithm - 2018](https://github.com/hibayesian/awesome-automl-papers#hyperparameter-optimization)


### Frameworks/Libraries

| Search Algorithm          | [scikit-learn][4-sk] | [scikit-optimize][3-skopt] | [hyperopt-sklearn][2-hopt] |
|---------------            |-------------   |-------------     | ---------------  |
| Grid Search               | x              |                  |                  |
| Random Search             | x              | x                |   x              |
| Baysian Opt               |                | x                |             |
| Tree of Parzen Estimators |               |                 |  x  |
| Annealing                 |                 |                 | x |
| Tree                      |                 |                  | x |
| Gaussian Proccess Tree    |                  | x               | x |



[1]: https://github.com/maxpumperla/hyperas
[2-hopt]: https://github.com/hyperopt/hyperopt-sklearn
[3-skopt]: https://scikit-optimize.github.io/
[4-sk]: http://scikit-learn.org/0.17/modules/grid_search.html
[nni]: https://github.com/Microsoft/nni
