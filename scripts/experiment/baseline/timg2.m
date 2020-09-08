tic;
close all;

N = 100;
in_listfile = './metadata/filelists/generated/timg_div2k_val_timgs.list';
fid = fopen(in_listfile);
fnames = cell(N,1);
timgs = cell(N,1);
imgs_basedir = './data/generated/timg_div2k_val/';
timgs_dirname = '/timgs';
for n = 1:N
    line_ex = fgetl(fid);
    [path, ~, ~] = fileparts(line_ex);
    img_name = strrep(strrep(path, imgs_basedir, ''), timgs_dirname, '');
    fnames{n} = img_name;
    timgs{n} = readNPY(line_ex);
end
fclose(fid);

sigma_spatial = 12.0;
sigma_intensity_T = 5000;
sigma_intensity_logT = 50;
logT_bias_correction = 0.577;
K_medord = 41;

% orig_dir = 'original';
imgaussfilt_T_dir = sprintf('imgaussfilt_T_s%.1f', sigma_spatial);
imbilatfilt_T_dir = sprintf('imbilatfilt_T_s%.1f', sigma_spatial);
medfilt2_dir = sprintf('medfilt2_%dx%d', K_medord, K_medord);
ordfilt2_dir = sprintf('ordfilt2_%dx%d', K_medord, K_medord);
imbilatfilt_logT_dir = sprintf('imbilatfilt_logT_s%.1f', sigma_spatial);
test_dir = './test/timgs_fixed';
% mkdir(test_dir, orig_dir);
mkdir(test_dir, imgaussfilt_T_dir);
mkdir(test_dir, imbilatfilt_T_dir);
mkdir(test_dir, medfilt2_dir);
mkdir(test_dir, ordfilt2_dir);
mkdir(test_dir, imbilatfilt_logT_dir);
P = 256;
for n = 1:N
    T = timgs{n};
    [H, W] = size(T);
    gap_H = mod(H, P/2);
    gap_W = mod(W, P/2);
    i0 = 1 + floor(gap_H/2);
    i1 = H - ceil(gap_H/2);
    j0 = 1 + floor(gap_W/2);
    j1 = W - ceil(gap_W/2);
    T = T(i0:i1, j0:j1);

    Tsm1 = imgaussfilt(T, sigma_spatial);
    Tsm2 = imbilatfilt(T, sigma_intensity_T, sigma_spatial);
    Tsm3 = medfilt2(T, [K_medord K_medord]) / log(2);
    Tsm4 = ordfilt2(T, round((1 - exp(-1))*(K_medord^2)), true(K_medord));
    Tsm5 = exp(imbilatfilt(log(T), sigma_intensity_logT, sigma_spatial) + logT_bias_correction);

%     Xorig = 1 ./ T;
    Xmle1 = 1 ./ Tsm1;
    Xmle2 = 1 ./ Tsm2;
    Xmle3 = 1 ./ Tsm3;
    Xmle4 = 1 ./ Tsm4;
    Xmle5 = 1 ./ Tsm5;
    
%     imwrite(Xorig, fullfile(test_dir, orig_dir, sprintf('%s.png', fnames{n})));
    imwrite(Xmle1, fullfile(test_dir, imgaussfilt_T_dir, sprintf('%s.png', fnames{n})));
    imwrite(Xmle2, fullfile(test_dir, imbilatfilt_T_dir, sprintf('%s.png', fnames{n})));
    imwrite(Xmle3, fullfile(test_dir, medfilt2_dir, sprintf('%s.png', fnames{n})));
    imwrite(Xmle4, fullfile(test_dir, ordfilt2_dir, sprintf('%s.png', fnames{n})));
    imwrite(Xmle5, fullfile(test_dir, imbilatfilt_logT_dir, sprintf('%s.png', fnames{n})));
end
toc;