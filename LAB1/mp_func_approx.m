clear all
clc
close all

hidden = 50;
eta = 0.1;
epochs = 300;
alpha = 0.9;
error = ones(1,epochs);



%generate data
x=[-5:1:5]';
y=x;
z=exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;

mesh (x, y, z);

gridsize = numel(x);
ndata = gridsize^2;

targets = reshape (z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

[insize, mau] = size(patterns);
[outsize, mau] = size(targets);

pat = [patterns; ones(1, mau)];



%Generate initial weights
w = randn(hidden,insize+1).*0.15;
v = randn(outsize,hidden+1).*0.15;

dw = zeros(hidden,insize+1).*0.01;
dv = zeros(outsize,hidden+1).*0.01;

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
    
    zz = reshape(out, gridsize, gridsize);
    mesh(x,y,zz);
    axis([-5 5 -5 5 -0.7 0.7]);
    drawnow;
    
    error(epoch) = sum(sum(abs(sign(out) - targets)./2));
    
end
figure;
plot(error)