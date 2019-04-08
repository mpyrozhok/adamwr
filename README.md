# AdamW optimizer and cosine learning rate annealing with restarts

This repository contains an implementation of AdamW optimization algorithm and cosine learning rate scheduler described in https://arxiv.org/abs/1711.05101. AdamW implementation is straightforward and does not differ much from existing Adam implementation for PyTorch, except that it separates weight decaying from batch gradient calculations.
Cosine annealing scheduler with restarts allows model to converge to a (possibly) different local minimum on every restart and normalizes weight decay hyperparameter value according to the length of restart period.
Unlike schedulers presented in standard PyTorch scheduler suite this scheduler adjusts optimizer's learning rate not on every epoch, but on every batch update, according to the paper.
Besides ["cosine"](https://www.google.com/search?q=(cos(x%2Fpi)%2B1)%2F2) there are two more learning rate policies available: ["arccosine"](https://www.google.com/search?q=arccos(2*x-1)%2Fpi), which has steeper profile at the limiting points and ["triangular"](https://www.google.com/search?q=1-abs(x*2-1)), which implements triangular lr policy proposed in https://arxiv.org/pdf/1506.01186v6.pdf.
The ratio of increasing and decreasing phases for triangular policy could be adjusted with `triangular_step` parameter. Minimum allowed lr is adjusted by `min_lr` parameter.
This scheduler could be used with AdamW and other PyTorch optimizers.

# Example:
```python
    batch_size = 32
    epoch_size = 1024
    model = resnet()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=5, t_mult=1.2, policy="cosine")
    for epoch in range(100):
        scheduler.step()
        train_for_every_batch(...)
            ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()
        validate(...)
```        
