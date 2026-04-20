def apply_training_step(loss, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
