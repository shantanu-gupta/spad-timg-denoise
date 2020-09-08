tic;
% close all;

% X = im2double(imread('./test/scene.png'));
X = im2double(rgb2gray(imread('./data/generated/timg_div2k_val/0832/original.png')));
% hX = histcounts(X, 100);
Xinv = 1.0 ./ X;
% Xinv(Xinv > 255) = 255;
[H, W] = size(X);

Tmin = 1e-3;
Tmax = 1e3;
T = Tmin + min(exprnd(Xinv), Tmax);

sigma_spatial = 1.5;
sigma_intensity_T = 5000;
sigma_intensity_logT = 100;
% logT_bias_correction = 0.577;
logT_bias_correction = 0.5722;
K_medord = 5;

Tsm1 = imgaussfilt(T, sigma_spatial);
Tsm2 = imbilatfilt(T, sigma_intensity_T, sigma_spatial);
Tsm3 = medfilt2(T, [K_medord K_medord]) / log(2);
Tsm4 = ordfilt2(T, round((1 - exp(-1))*(K_medord^2)), true(K_medord));
Tsm5 = exp(imbilatfilt(log(T), sigma_intensity_logT, sigma_spatial) + logT_bias_correction);

Xmle1 = 1 ./ Tsm1;
Xmle2 = 1 ./ Tsm2;
Xmle3 = 1 ./ Tsm3;
Xmle4 = 1 ./ Tsm4;
Xmle5 = 1 ./ Tsm5;

figure; imshow(Xmle1);
figure; imshow(Xmle2);
figure; imshow(Xmle3);
figure; imshow(Xmle4);
figure; imshow(Xmle5);

toc;