tic;
Tmin = 1e-3;
Tmax = 1e3;
Nl = 256;
la = linspace(1/255, 1, Nl);
S = zeros(Nl, 1);
for nl = 1:Nl
    nl
    c0 = la(nl) * Tmin;
    c1 = la(nl) * Tmax;
    dx = la(nl) * 1e-4;
    N = round((c1 - c0)/dx);
    u = linspace(c0, c1, N);
    y = log(u) .* exp(-u);
    S(nl) = sum(y) .* dx;
end

bf = S + (1 - exp(-la' * Tmin)) * log(Tmin) + exp(-la' * Tmax) * log(Tmax);
mf = exp(-la' * Tmin) - exp(-la' * Tmax);

Sc = -S;
lt = -log(la)/log(Tmax);
figure; plot(lt, S);
toc;