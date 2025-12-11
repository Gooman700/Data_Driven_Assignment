# Edward Cox Data Driven Assignment
Python 3.14.2
### Download Python
To recreate the project in a closed environment, firstly install Python version 3.14.2

### Setup Project Folder
Create a folder somewhere that you wish the project to be in.
Then, for Windows, run 
`python3.14 -m venv <folderpath\venv>`
Where folderpath is the path to the folder you want the project to be.

### Clone the GitHub Repo
Run the command 
`cd <folderpath>`
With Git installed, run the command 
`git clone https://github.com/Gooman700/Data_Driven_Assignment.git`

### Download Requirements
Next, open command prompt and run 
`folderpath\venv\Scripts\activate.bat`
This ensures all module downloads are contained within the environment.
Then, run 
`pip install requirements.txt` 
to download all the requirements.

### Run the Program
To run the program, firstly run 
`python train.py`
Then, run 
`python evaluate.py`
