# julia

This repository contains julia modules for work with wombat cluster simulations, plotting and
image production.

In particular:
* A cosmology module that solves Friedman's equation with Gaussian quadrature [Cosmology.jl]
* A bunch of CGS constants [CGSUnits.jl]
* Binning routines to make radial profiles [Binning.jl]
* A useful implementation of Kovesi's perceptually uniform colormaps including a 2D array to image conversion with log10 and sqrt scaling. [ColorMaps.jl]
* Functions that safely operate on arrays containing non-numbers [ArrayStatistics.jl]
* A function to convert an integer to a string with zero padding [StringConversion.jl]

