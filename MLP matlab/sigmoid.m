function g = sigmoid(x,b)
% sigmoid function

g = 1./(1+exp(-b*x));
end