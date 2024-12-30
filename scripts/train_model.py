from src.training.trainer import Trainer
import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.training.trainer import Trainer

if __name__ == "__main__":

    trainer = Trainer()
    trainer.run()