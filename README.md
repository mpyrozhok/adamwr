# AdamW optimizer and cosine learning rate annealing with restarts

This repository contains an implementation of AdamW optimization algorithm and cosine learning rate scheduler described in https://arxiv.org/abs/1711.05101. AdamW implementation is straightforward and does not differ much from existing Adam implementation for PyTorch, except that it separates weight decaying from batch gradient calculations.
Cosine annealing scheduler with restarts allows optimizer to converge to a (possibly) different local minimum on every restart and normalizes weight decay hyperparameter value according to the length of restart period. This scheduler implementation is different from schedulers presented in standard PyTorch scheduler suite. It adjusts optimizer's learning rate not on every epoch, but on every batch update, according to the paper. The scheduler could be used with AdamW and other PyTorch optimizers.

# Example:
```python
    optimizer = AdamW(lr=1e-3, weight_decay=1e-5)
    scheduler = CosineLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
    for epoch in range(100):
        scheduler.step()
        train(...)
            ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()
        validate(...)
```        
