function g = ift2(G, dfx, dfy) % function g = ift2(G, delta_f)
    Nx = size(G,1);
    Ny = size(G,2);
    g = ifftshift(ifft2(ifftshift(G))) * Nx*Ny*dfx*dfy;
end
