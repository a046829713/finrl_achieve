import torch

# Updated function f(x) = -x^2 + 4x
def updated_function(x):
    return -x ** 2 + 4 * x

# Starting point
x_max = torch.tensor([1.0], requires_grad=True)  # For gradient ascent
x_min = torch.tensor([1.0], requires_grad=True)  # For gradient descent

# Iterations
# Learning rate
learning_rate = 0.1

# Iterations
iterations = 1000

# Gradient ascent and descent
ascent_history = []
descent_history = []

for i in range(iterations):
    # Gradient ascent (maximize)
    f_max = updated_function(x_max)
    print(x_max)
    print(f_max)
    f_max.backward()  # Compute gradients
    print(x_max.grad.data)
    print('*'*120)
    x_max.data += learning_rate * x_max.grad.data  # Update x
    x_max.grad.data.zero_()  # Reset gradients
    ascent_history.append(x_max.item())

    # Gradient descent (minimize)
    # f_min = function(x_min)
    # f_min.backward()  # Compute gradients
    # x_min.data -= learning_rate * x_min.grad.data  # Update x
    # x_min.grad.data.zero_()  # Reset gradients
    # descent_history.append(x_min.item())

print(ascent_history, descent_history)
