tic;

% X = im2double(imread('./test/scene.png'));
X = im2double(rgb2gray(imread('./test/0855.png')));
% hX = histcounts(X, 100);
Xinv = 1.0 ./ X;
[H, W] = size(X);

Tmin = 1e-3;
% Tmax = Tmin;
Tmax = 1e3;
T = max(Tmin, exprnd(Xinv));
T(T > Tmax) = Tmax;

K = 11;
K2 = K^2;
Khalf = floor(K/2);
Tpad = padarray(T, [K K]);
Tout = zeros(size(T));
dthresh = 10;
ord_ind = 1 - exp(-1);
for i = K+1:K+H
    if mod(i, 100) == 0
        i
    end
    
    for j = K+1:K+W
        tval = Tpad(i,j);
        nhood = Tpad(i-Khalf:i+Khalf,j-Khalf:j+Khalf);
        dt = abs(nhood * (1.0/tval));
%             mask = dt < dthresh;
%         mask = dt > (1/dthresh);
        mask = and(dt > (1/dthresh), dt < dthresh^2);
%         Tout(i-K, j-K) = mean(nhood(mask), 'all');
        nvals = sort(reshape(nhood(mask), [], 1));
        Tout(i-K,j-K) = nvals(round(ord_ind*numel(nvals)));
    end
end

figure; imshow(log(Tout), [-7 0]);
Xest = 1.0 ./ Tout;
figure; imshow(Xest);
toc;