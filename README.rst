Sleeping Reinforcement Learning
*******************************

Simone Drago, Marco Mussi and Alberto Maria Metelli

Running Experiments
===================

The code requires *python3* along with *numpy* and *matplotlib*.

The configurations used in the experiments are in the *configs* folder.

To run the experiments, from the root directory, call the bash script *runner_sequence.sh*. Such a script will read all the configuration files present in the *configs* folder and will execute the *parallel_runner.py* Python script with such configurations. The Python script *parallel_runner.py* can also be called with the pathname of a single configuration file (with the relative path from the root directory and with the extension ".json").

Cite this Work
==============

If you are using this code for your scientific publications, please cite:

.. code:: bibtex

    @inproceedings{drago2025sleeping,
      author    = {Drago, Simone and 
                   Mussi, Marco and 
                   Metelli, Alberto Maria},
      title        = {Sleeping Reinforcement Learning},
      booktitle    = {International Conference on Machine Learning (ICML)},
      series       = {Proceedings of Machine Learning Research},
      volume       = {267},
      publisher    = {{PMLR}},
      year         = {2025}
   }

Contact Us
==========

For any question, drop an e-mail at marco.mussi@polimi.it
