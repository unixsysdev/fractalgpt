"""
Contrived data generator for architecture validation.

Creates tasks of varying difficulty:
- Easy: copy, simple patterns → should use Ghost mode
- Medium: counting, simple arithmetic → should use mid mode  
- Hard: multi-step reasoning → should use full mode

The model should learn to use minimum compute for each difficulty.
"""

import torch
import random
from typing import Tuple, List


def encode(text: str) -> List[int]:
    """Simple byte encoding."""
    return [ord(c) for c in text]


def decode(tokens: List[int]) -> str:
    """Decode bytes to string."""
    return ''.join(chr(t) for t in tokens if 0 <= t < 256)


def pad_sequence(tokens: List[int], max_len: int, pad_value: int = 0) -> List[int]:
    """Pad or truncate to fixed length."""
    if len(tokens) >= max_len:
        return tokens[:max_len]
    return tokens + [pad_value] * (max_len - len(tokens))


class DifficultyDataset:
    """
    Dataset with explicit difficulty labels.
    
    Easy tasks: should use ~16-32 dims
    Medium tasks: should use ~64 dims
    Hard tasks: should use 128 dims (full)
    """
    
    def __init__(self, max_len: int = 64):
        self.max_len = max_len
        self.difficulties = ['easy', 'medium', 'hard']
    
    def generate_easy(self) -> Tuple[str, str, str]:
        """Copy task - trivial."""
        patterns = [
            ("copy: abc", "abc"),
            ("copy: hello", "hello"),
            ("copy: xyz", "xyz"),
            ("copy: 123", "123"),
            ("repeat: a", "a"),
            ("echo: test", "test"),
        ]
        q, a = random.choice(patterns)
        return q, a, 'easy'
    
    def generate_medium(self) -> Tuple[str, str, str]:
        """Counting and simple arithmetic."""
        task_type = random.choice(['count', 'add', 'length'])
        
        if task_type == 'count':
            char = random.choice('aeiou')
            word = ''.join(random.choice('abcdefghij') for _ in range(random.randint(3, 8)))
            count = word.count(char)
            q = f"count {char} in {word}"
            a = str(count)
        
        elif task_type == 'add':
            x, y = random.randint(1, 20), random.randint(1, 20)
            q = f"{x}+{y}="
            a = str(x + y)
        
        else:  # length
            word = ''.join(random.choice('abcdefghij') for _ in range(random.randint(2, 10)))
            q = f"len({word})"
            a = str(len(word))
        
        return q, a, 'medium'
    
    def generate_hard(self) -> Tuple[str, str, str]:
        """Multi-step problems."""
        task_type = random.choice(['multi_add', 'reverse_add', 'digit_sum'])
        
        if task_type == 'multi_add':
            x, y, z = random.randint(1, 30), random.randint(1, 30), random.randint(1, 30)
            q = f"{x}+{y}+{z}="
            a = str(x + y + z)
        
        elif task_type == 'reverse_add':
            x = random.randint(10, 99)
            rev = int(str(x)[::-1])
            q = f"{x}+reverse({x})="
            a = str(x + rev)
        
        else:  # digit_sum
            n = random.randint(100, 999)
            digit_sum = sum(int(d) for d in str(n))
            q = f"digitsum({n})"
            a = str(digit_sum)
        
        return q, a, 'hard'
    
    def generate_example(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Generate one training example."""
        difficulty = random.choice(self.difficulties)
        
        if difficulty == 'easy':
            q, a, d = self.generate_easy()
        elif difficulty == 'medium':
            q, a, d = self.generate_medium()
        else:
            q, a, d = self.generate_hard()
        
        # Format: "Q: ... A: ..."
        text = f"Q:{q} A:{a}"
        tokens = encode(text)
        tokens = pad_sequence(tokens, self.max_len)
        
        # Input and target (shifted by 1)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y, d
    
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Generate a batch of examples."""
        xs, ys, ds = [], [], []
        for _ in range(batch_size):
            x, y, d = self.generate_example()
            xs.append(x)
            ys.append(y)
            ds.append(d)
        
        return torch.stack(xs), torch.stack(ys), ds


class FixedDifficultyDataset(DifficultyDataset):
    """Dataset that only generates one difficulty level."""
    
    def __init__(self, difficulty: str, max_len: int = 64):
        super().__init__(max_len)
        self.fixed_difficulty = difficulty
    
    def generate_example(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.fixed_difficulty == 'easy':
            q, a, d = self.generate_easy()
        elif self.fixed_difficulty == 'medium':
            q, a, d = self.generate_medium()
        else:
            q, a, d = self.generate_hard()
        
        text = f"Q:{q} A:{a}"
        tokens = encode(text)
        tokens = pad_sequence(tokens, self.max_len)
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y, d


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing data generation...")
    
    dataset = DifficultyDataset()
    
    # Show examples
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n{difficulty.upper()} examples:")
        for _ in range(3):
            if difficulty == 'easy':
                q, a, _ = dataset.generate_easy()
            elif difficulty == 'medium':
                q, a, _ = dataset.generate_medium()
            else:
                q, a, _ = dataset.generate_hard()
            print(f"  Q: {q} → A: {a}")
    
    # Generate batch
    x, y, d = dataset.generate_batch(4)
    print(f"\nBatch shapes: x={x.shape}, y={y.shape}")
    print(f"Difficulties: {d}")
    
    print("\n✓ Data generation works!")
