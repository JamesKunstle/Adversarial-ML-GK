== please contact jkunstle@bu.edu for questions regarding this repo. This README is an open project ==

##Welcome
Thank you so much for visiting this project! 

##Introduction
The purpose of this repository is to make it easier to share our work with other collaborators. There's a limitless supply of things to do, and centralizing the efforts of the project in one repository, and maintaining consistent quality, is essential to making consistent progress. An open source collaboration strategy makes sense.

##Technical Details
The base data loader, model training api, and attack api are all runnable as demos and their essential functionality is documented in their respective main() methods. 


##Contribution Details
When contributing to this repo, please~
    (1) Fork this repo.
    (2) Set this repo as upstream and your own fork as origin.
    (3) Develop features or improvements in a new branch.
    (4) Rebase upstream changes onto your branch and test thoroughly.
    (5) Make PR's from your forked development branch.

    ^^ This workflow makes it very easy to contribute even with rapidly changing, often optimized API's.

All experiments must adhere to the "XXX-Experiment-Title.py" format inside of the "experiments" folder. You can see how to reference the repo's functionality via the sys.path.append interface. An example of this would be:
    "000-test-experiment.py"

Each numerical code "XXX" in the experiment's name ought to be logged and explained in the file:
     "experiments/experiments_log.txt"
