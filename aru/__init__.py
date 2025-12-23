"""
ARU - Accumulation Recurrent Unit

A minimal recurrent architecture using decoupled persistence and update gates
for true information accumulation. Designed to be faster than LSTM/GRU while 
maintaining competitive accuracy.

Author: Abderahmane Ainouche
License: MIT

Usage:
    # Basic import of the core model
    from aru import ARU
    
    # Text classifier helper
    from aru import create_aru_classifier
    
    model = create_aru_classifier(vocab_size=10000, hidden_size=64, num_classes=2)
"""

from aru.model import ARU, create_aru_classifier, create_aru_encoder, aru_loss
from aru.baselines import ManualGRU, ManualLSTM, ManualRNN

__version__ = "0.1.0"
__author__ = "Abderahmane Ainouche"
__all__ = [
    "ARU", 
    "ManualGRU", 
    "ManualLSTM", 
    "ManualRNN", 
    "create_aru_classifier",
    "create_aru_encoder",
    "aru_loss"
]

def create_classifier(
    vocab_size: int,
    hidden_size: int = 128,
    num_classes: int = 2,
    dropout: float = 0.1
) -> ARU:
    """
    Create an ARU classifier for text.
    
    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden state dimension
        num_classes: Number of output classes
        dropout: Dropout probability
        
    Returns:
        ARU model configured for text classification
    """
    return create_aru_classifier(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=dropout
    )