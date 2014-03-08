% note the dimensions of the weights to the input and output. the first
% input dimension for the first weights is +1 more than the input dimension.

Input = [1;2;5];
weight_1 = rand(5,4);
weight_2 = rand(3,5);
weight_3 = rand(2,3);
target_output = [1;0];

% for this example, N interations are executed note the convergence by
% observing err for the error which is the L2 norm of output and
% target_output.
N = 100;
Err = zeros(1,N);
for k=1:N
   [weight_1,weight_2,weight_3,output,err] = ann3(Input,weight_1,weight_2,weight_3,target_output,1);
   Err(k) = err;
end

figure; plot(Err)
disp(output)
