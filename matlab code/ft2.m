function [ G ] = ft2(g,delta)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

G = fftshift(fft2(fftshift(g))) * delta^2;

end

