function [Upsampled, dx2] = upsampling(Data, Pow2factor, dx1 )
%   UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

Upsampled = interp2(Data,Pow2factor,'spline');
dx2 = dx1/(2^Pow2factor);

end

