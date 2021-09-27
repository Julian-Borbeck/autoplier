# jdrfmetabo
Transform metabolite GCMS data to pathway representation via HMDB

## Getting Started
1) Create a `data` folder (highlighted below). The file directory should look like this.
- jdrfmetabo/
  - R/
    - query_hmdb.Rmd
  - refmet_results_example.txt
  - src/
  - tests/
  - **data**/
    - **refmet_results.txt**
  - main.py
  - Makefile
  - requirements.txt

2) Manually, preprocess the native omics data. Get metabolite names and HMDB ids and save
them to `data/refmet_results.txt`. An example file is located in
`data/refmet_results_example.txt`

3) Generate the membership matrix by running the file `R/query_hmdb.Rmd` in RStudio to complete the preprocessing phase.

4) Use the Makefile to run the main analyis:
 ```
 make main
 ```
 This step will make the virtual environment the first time you run it. Afterward, this step should be quicker.

## Misc
Alternatively, you can always activate the virtual environment manually with
 ```
. venv.nosync/bin/activate
 ```
then run the main analysis with
 ```
python main.py
 ```
Unit testing can be done via the Makefile with
```
make test
```
Also, you can discard the virtual environment and the cached files with
```
make clean
```
Then run
```
make
```
to create a new virtual environment. Note, everytime `requirements.txt` is modified `make` will update and rebuild the virtual environment.
