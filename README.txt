This repository consists of the following,

TransitFinderCNN.py this is the model used to predict planets and their periods
PlanetGeneration.py This is the script to generate the training data used to train TransitFinder
TransitFinderFunctions.py This file consits of a list of functions,some of these will be used to find planets in a more transitional sense, by performing a lomb-sargle and forlding lightcuvres, it can be used to search the kepler archive or to perfom analyise on the provided filts files
CourseworkB.ipynb this is a jupyter notbook consisitng of a report and the analysis of the provided dataset which is found in /CourseworkData

There is a few iterations of the model attached.


I want to specify the coding principals and what I belive leads to readable code at a professional level. These have been used in the planning and structure of the code and they have been thoughtfully considerd. Some of these are common sense, some are more controversial.

Minimise repition
-> One thing that will be noticed on first glance, is repition is kept to a minimun some functions have multiple purposes to reduce repition in the code, the only exception is some plots as I want them too specific and I think the matplotlib libary already reduces the amount of code needed to plot by such a significant amount, this is not needed.
-> An example in TransitFinderFunctions.py there is a function "loadDataFromFitsFiles" this loads the kepler data into filts files, later in the project I decided I sometimes wanted to load this data in, with a random order, while it could be argued that a seperate function would be intuitave, I decided to opt and reuse the function but pass in a parameter if the data is desired to be shuffled, this parameter will be set to false if unshuffled data is desired and true is shuffled data is desired.

Never nest ( well most the time )
-> nesting increase the amoount of conditions and logic your brain needs to hold at one time, while it does help in reading from A to B in my opion however this does not scale well, when the number of indentations gets high I extract indentations and use them as their own function, or when the code starts to get more complex it would make sense for confined aspects of the code to be extracted
-> An example in TransitFinderFunctions.py is 'analyse_peaks_wtih_bls' this is called when the peaks are wanted to be analysed with a box least squares fit. this function is small but it is clear what it does, if a reader is curious about the specific peak finding they can then check the analyse_periods function which gives more details about the peak analysis, and if the reader is curoius about the specifics of the bls algorim used they can then read that function. This abstraction keeps specific code in nice little bits without the reader loosing track of what is going on.
-> https://www.youtube.com/watch?v=CFRhGnuXG-4&t=273s&pp=ygUKbmV2ZXIgbmVzdA%3D%3D  a good video on never nesting
-> there is a slight counter example where I felt it was not approiate to perfom abstraction as I felt the code was more readable in its current format. If I was to add more logic I would be performing abstraction.

Code should show not tell.
-> In my opinion specific comments lead to making assumptions about how code works, this takes focus away from the code itself.
-> individual lines of code should read like a sentance, the variable names and names of the functions should describe what is intended.
-> There is scenarios where code does not make intuitive sense and is more abstract, these should be commented.
-> functions should have a description to them, and what the intended input and outputs are. where it makes logical sense to do so unit tests should test functions.

Algoithm over assumption.
-> If a value or an input could be found algoirthmically it should.
-> an example is the peak thresholds in 'run_lomb_scargle_analysis' where the peak thresholds as and conditions are determined algoirthmically, but for this case I did allow a manual threshold as that could prove useful when prexisting knowledge is known about the output.