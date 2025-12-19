"""
Experiment Logger for tracking continual learning experiments
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class ExperimentLogger:
    """Logger for continual learning experiments"""
    
    def __init__(self, experiment_name: str, output_dir: str = './logs'):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = self.exp_dir / 'experiment.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        
        # Initialize metrics storage
        self.metrics = {
            'config': {},
            'tasks': [],
            'summary': {}
        }
        
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Output directory: {self.exp_dir}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        self.metrics['config'] = config
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save config
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_task_start(self, task_id: int, task_info: Optional[Dict] = None):
        """Log start of a task"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting Task {task_id}")
        self.logger.info(f"{'='*60}")
        
        if task_info:
            for key, value in task_info.items():
                self.logger.info(f"  {key}: {value}")
    
    def log_epoch(self, task_id: int, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics"""
        log_str = f"Task {task_id} - Epoch {epoch}:"
        for key, value in metrics.items():
            log_str += f" {key}={value:.4f}"
        self.logger.info(log_str)
    
    def log_task_end(self, task_id: int, metrics: Dict[str, Any]):
        """Log task completion metrics"""
        self.logger.info(f"\nTask {task_id} completed:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        # Store task metrics
        task_data = {
            'task_id': task_id,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics['tasks'].append(task_data)
        
        # Save incremental results
        self._save_metrics()
    
    def log_evaluation(self, task_id: int, eval_results: Dict[int, float]):
        """Log evaluation results across all tasks"""
        self.logger.info(f"\nEvaluation after Task {task_id}:")
        for eval_task_id, accuracy in eval_results.items():
            self.logger.info(f"  Task {eval_task_id}: {accuracy:.2%}")
    
    def log_final_summary(self, summary: Dict[str, Any]):
        """Log final experiment summary"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Final Summary")
        self.logger.info(f"{'='*60}")
        
        for key, value in summary.items():
            if isinstance(value, float):
                self.logger.info(f"{key}: {value:.4f}")
            else:
                self.logger.info(f"{key}: {value}")
        
        self.metrics['summary'] = summary
        self._save_metrics()
        
        self.logger.info(f"\nExperiment completed. Results saved to: {self.exp_dir}")
    
    def save_checkpoint(self, model: torch.nn.Module, task_id: int, 
                       optimizer: Optional[torch.optim.Optimizer] = None):
        """Save model checkpoint"""
        checkpoint_path = self.exp_dir / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True)
        
        checkpoint = {
            'task_id': task_id,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        save_path = checkpoint_path / f'task_{task_id}.pt'
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved: {save_path}")
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_path = self.exp_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_exp_dir(self) -> Path:
        """Get experiment directory path"""
        return self.exp_dir
