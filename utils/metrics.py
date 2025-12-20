import numpy as np

class CLMetrics:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))

    def update(self, train_task_id, test_task_id, accuracy):
        self.accuracy_matrix[test_task_id, train_task_id] = accuracy

    def calculate_metrics(self, current_task_id):
        accuracies = self.accuracy_matrix[:current_task_id+1, current_task_id]
        avg_acc = np.mean(accuracies)

        if current_task_id > 0:
            forgetting = 0.0
            for i in range(current_task_id): 
                max_acc_past = np.max(self.accuracy_matrix[i, :current_task_id])
                current_acc = self.accuracy_matrix[i, current_task_id]
                forgetting += (max_acc_past - current_acc)
            
            avg_forgetting = forgetting / current_task_id
        else:
            avg_forgetting = 0.0

        return avg_acc, avg_forgetting

    def print_matrix(self):
        print("\nAccuracy Matrix (Rows: Test task, Cols: Trained task):")
        print(self.accuracy_matrix)