clear all
clc
close all

inputs = 2;
hidden = 4;
eta = 0.25;
epochs = 20;
alpha = 0.9;
error = ones(1,epochs)

w = randn(1,inputs+1).*0.01;
v = randn(1,hidden).*0.01;

%generate data
nsepdata;

[insize, ndata] = size(patterns);
[outsize, ndata] = size(targets);

for epoch=1:epochs
    
    %forward pass
    hin = w * [patterns ; ones(1,ndata)];
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;

    %backward pass
    delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;
    delta_h = (v' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_h = delta_h(1:hidden, :);

    %weight update
    dw = (dw .* alpha) - (delta_h * pat') .* (1-alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;
    
    error(epoch) = sum(sum(abs(sign(out) - targets)./2));
end