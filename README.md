# MinuteCrypticDecrypter
**Language and Computation Final Project** <br>
*Amanda Huang, Gbemiga Salu, Miya Zhao* <br>

**Summary:** This project utilizes python, [GloVe](https://huggingface.co/fse/glove-wiki-gigaword-50), and various libraries to classify different types of Minute Cryptic Puzzles, and solve them based on their clues. It utilizes computational and lingustic concepts like n-grams, semantic embeddings, Multinomial Logistic Regression to generate and filter solutions to the [Minute Cryptic](https://www.minutecryptic.com/).

This model is intended to be used to classify and generate solutions for Anagram, Hidden, and Selector type puzzles.

## Installation:

Ran with Python 3.12.6 <br>

**Packages:** <br>
Scikit-Learn <br>
Pandas <br>
Numpy <br>
Re <br>
Wordfreq <br>
gensim.downloader <br>

These are all free use programs. Many can be downloaded using: pip install [program name]

## Usage:

### Files Explaination (More technical overview is found in code):

**anagram.py**
Used to solve Anagram-type puzzles.

**selector.py**
Used to solve Selector-type puzzles.

**hidden.py**
Used to solve Hidden-type puzzles.

**Indicatorcracker.py**
Used to classify the type of cryptic clue. Reads input from user (clues and length of answer). Using training and test data and various features, the logistic regression assigns probabilities to the 3 different classes and selects the most likely puzzle-type.

**finalsolver.py**
Reads input from user (clues and length of answer). When given a specific puzzle-type, the program will run the algorithm for that puzzle-type and generate a most optimal final answer.

**glove.py**
Loads a pre-trained GloVe vector (trained on 2B tweets)/ Used for calculating semantic similarity
 
**logistic_data.csv & testsolver.csv**
Contains data harvested from [Minute Cryptic's YouTube Channel](https://www.youtube.com/@MinuteCryptic), used for Indicatorcracker.py training

We utilized cross-validation to make up for data sparsity.

### Output Data:
Data we collected which we utilized in a creation of a heat map for our final report. 

In **clue_probabiltiies** we have the reported probability for each type of puzzle.

In **testsolver_results.csv** we have the results for if our algorithms were able to come up with a correct solution.

In **final_results.csv** we have data on which part our model was success in (classification versus solving) and the specific nuances inside.

Additionally, these files were helpful for understanding how close the classifications were to each other and how errors occured. For future students interested in advancing this dataset, in indicatorcracker.py we have commented out code that showcases how we extracted this data.

## Main Usage:
2 Main Files to run: indicatorcracker.py & finalsolver.py

### Example Usage

**indicatorcracker.py:**

*italics notates user input*

----- What Category is the Minute Cryptic Indicator? -----  <br>
Enter the clue: *Provide a lipstick sample for model*  <br>
Enter the indicator:  *sample*  <br>
Enter the fodder words: *Provide a lipstick*  <br>
Enter the definition: *model*  <br>
Enter the solution length: *5*  <br>

----- Prediction Result -----  <br>
Predicted Category: Selectors  <br>

Probabilities:  <br>
  Anagrams: 0.0052  <br>
  Hiddens: 0.2694  <br>
  Selectors: 0.7254  <br>

**finalsolver.py:**

Enter the fodder: *Provide a lipstick*  <br>
Enter the answer length: *5*  <br>
Enter category (anagram / hidden / selector): *hidden*  <br>

Multiple candidate words: {'stick', 'ideal', 'provi'}  <br>
Enter the DEFINITION part of the clue: *model*  <br>

--- GloVe Scoring ---  <br>
Definition: model  <br>
Best Match: ideal  <br>

ideal           0.6242  <br>
stick           0.3179  <br>
provi           0.0000  <br>

Final Answer: ideal  <br>


