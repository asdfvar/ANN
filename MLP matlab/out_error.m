function dout = out_error(t,y)
% t: target output
% y: actual output

dout = (t-y).*y.*(1-y);
end
