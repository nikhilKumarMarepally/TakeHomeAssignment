## Prerequisites
- Docker installed on your machine. [Install Docker](https://docs.docker.com/get-docker/)

## Folder Structure
```
TakeHomeAssignment/
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   ├── task1_sentence_transformer_implementation.py
│   ├── task4_train_multi_task_model.py
│── task_3_overview.txt  
├── notebooks/
│   └── TakeHomeAssignment.ipynb
└── scripts/
    └── run_docker.sh
```

## Steps to reproduce environment
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nikhilKumarMarepally/TakeHomeAssignment.git
   sh scripts/run_docker.sh #This will create the environment to run task1 and task4(task2 is included)
	--- python3 src/task1_sentence_transformer_implementation.py
	--- python3 src/task4_train_multi_task_model.py
        --- task3_overview.txt provides the overview for task3
   ```

2. task_3_overview.txt (write-up summarizing my key decisions and insights)
3. notebooks/TakeHomeAssignment.ipynb has the entire flow of tasks
