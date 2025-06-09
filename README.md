# Advancing Knowledge Tracing by Exploring Follow-up Performance Trends
This repository is the official implementation of the Forward-Looking Knowledge Tracing Framework (FINER).

## Abstract 
Intelligent Tutoring Systems (ITS), such as Massive Open Online Courses, offer new opportunities for human learning. At the core of such systems, 
knowledge tracing (KT) predicts students' future performance by analyzing their historical learning activities, enabling an accurate evaluation of students' knowledge states over time. 
We show that existing KT methods often encounter correlation conflicts when analyzing the relationships between historical learning sequences and future performance.
To address such conflicts, we propose to extract so-called Follow-up Performance Trends (FPTs) from historical ITS data and to incorporate them into KT.
Specifically, we propose a method called Forward-Looking Knowledge Tracing (FINER) that combines historical learning sequences with FPTs to enhance student performance prediction accuracy. 
FINER constructs learning patterns that facilitate retrieval of FPTs from historical ITS data in linear time; 
FINER includes a novel similarity-aware attention mechanism that aggregates FPTs based on both frequency and contextual similarity; 
and FINER offers means of combining FPTs and historical learning sequences to enable more accurate prediction of student future performance. 
Experiments on six real-world datasets show that FINER is able to outperform ten state-of-the-art KT methods, increasing accuracy by 13.25% to 84.91%.

## Code Structure

The repository is organized as follows:

```
📦 FINER
 ┣ 📂 Data                      # Dataset directory
 ┃ ┣ 📂 assist2009             # ASSISTments 2009 dataset
 ┃ ┣ 📂 assist2012             # ASSISTments 2012 dataset  
 ┃ ┣ 📂 assist2015             # ASSISTments 2015 dataset
 ┃ ┣ 📂 Algebra08              # Algebra 2008 dataset
 ┃ ┣ 📂 HDUOJ                  # HDU Online Judge dataset
 ┃ ┗ 📂 Junyi                  # Junyi Academy dataset
 ┃
 ┣ 📂 Model                     # Model implementations
 ┃ ┣ 📜 FINER.py               # FINER model implementation
 ┃ ┣ 📜 <model_name>.py        # Models integrated into FINER
 ┃ ┗ 📜 <model_name>_all.py    # Standalone KT models
 ┃
 ┣ 📂 Utils                     # Utility functions
 ┃ ┣ 📜 data_loader.py         # Data loading and preprocessing
 ┃ ┣ 📜 LearningPatternTree.py # Learning pattern Trie implementation
 ┃ ┣ 📜 run.py                 # Training and evaluation functions
 ┃ ┗ 📜 utils.py               # Helper functions
 ┃
 ┣ 📂 Results                   # Experimental results
 ┃ ┣ 📂 Log                    # Training and testing logs
 ┃ ┗ 📂 Figure                 # Result visualization
 ┃
 ┣ 📜 environment.sh           # Environment configuration
 ┗ 📜 main.py                  # Main program entry
```

<!-- - `Data/`: Directory to store six datasets used in our paper.
    - `assist2009/` stores the `assist2009` dataset.
    - `assist2012/` stores the `assist2012` dataset.
    - `assist2015/` stores the `assist2015` dataset.
    - `Algebra08/` stores the `Algebra08` dataset.
    - `HDUOJ/` stores the `HDUOJ` dataset.
    - `Junyi/` stores the `Junyi` dataset.
- `Model/`: Contains implementations of different knowledge tracing models.
  - `FINER.py`: Implementation of the FINER model.
  - `<model_name>.py`: Implementations of various historical learning sequence models that can be used as components within FINER. We support integrating existing KT models as the historical learning sequence component of FINER.
  - `<model_name>_all.py`: Standalone implementations of the corresponding KT models.
- `Utils/`: Utility functions and helper modules.
  - `data_loader.py`: Functions for loading and preprocessing datasets.
  - `LearningPatternTree.py`: Functions for building and using the learning pattern Trie.
  - `run.py`: Functions for running training and evaluation epochs.
  - `utils.py`: Miscellaneous utility functions.
- `Results/`: Directory to store experimental results and outputs.
  - `Log/`: Directory to store log files which we record the training and testing process.
  - `Figure/`: Directory to store figure which we show the experimental results.
- `enviroment.sh`: List of required Python packages.
- `main.py`: The main entry point for running experiments. -->

## Install the Requirments of Experiment
```bash
    conda create -n FINER_Env python=3
    conda activate FINER_Env
    pip install torch
    pip install numpy
    pip install scikit-learn
    pip install matplotlib
    pip install seaborn
    pip install tqdm
```
## Running
### Datasets Selection
Select a dataset you want to include.
There are four kinds of dataset

| Dataset   | Assistant2009 | Assistant2012 | Assistant2015 | Algebra08 | HDUOJ    | Junyi     |
|:----------|:------------:|:-------------:|:-------------:|:---------:|:--------:|:---------:|
| Students  | 4,151        | 28,834        | 19,840        | 247       | 137,374  | 1,000     |
| Questions | 110          | 198           | 110           | 424       | 5,320    | 715       |
| Records   | 325,637      | 2,530,080     | 683,801       | 1,048,575 | 15,087,568| 5,436,816 |
| Avg.rec   | 78           | 88            | 34            | 4,245     | 110      | 5,436     |

`data/<dataset>/<dataset>_<kind>_<seed>.csv` 

Each dataset represents a student's answer record in three rows.

The first line represents the total number of questions answered by students.

The second line represents the sequence of students' questions.

The third line represents the sequence of students' answers

### Training Model
We use `assist2009` as an example of a dataset.
If you just want to generate `FPTTrie`. 
    
```bash
    python main.py --model_type FINER-DKT --dataset_type assist2009 --generate_followup_trends 1 --num_learningpattern 2 --num_followup_attempts 2
```

If you want to run our model(such as FINER-DKT)
    
```bash
    python main.py --model_type FINER-DKT --dataset_type assist2009 --generate_followup_trends 0
```
 
### Hyper-parameters about FINER
In our setting,the $\bar{z}$ (`num_learningpattern`) is 2,the $\bar{i}$ (`num_followup_attempts`) is 2. We use Adam optimizer for optimization with learning rate 0.01,and batch size is set as 32.

The more detail about hyper-parameters of different situations can be found in log files in `./Results/Log/` directory. 
