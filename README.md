# Indirect-Dark-Matter-Detection

The purpose of this repository is to make avaliable the code used to fit curves of mass distribution and rotation curves on models of dark matter presence in galaxies and nebulae. The code is written in python. This is part of a [FAPESP scholarship](https://bv.fapesp.br/en/bolsas/192826/study-on-indirect-dark-matter-seach).

There's also [my article](https://medium.com/@victorfoscarini/indirect-dark-matter-detection-with-python-3f191594d5a) on [my medium](https://medium.com/@victorfoscarini) explaining more on the project.

Here the notebooks rotation_curves.ipynb and capstone.ipynb are used to display fits of three models of dark matter distribution in data from galaxies taken from relevant articles in the area. The fits were done using known python libraries like matplotlib and seaborn for plotting, scipy for nonlinear regression and numpy for linear algebra. I created a big class (the class isn't that beautiful but does a god job) that does all the necessary calculations and data preprocessing. Also, I did statistical tests to check the goodness-to-fit and where the fit is better.
