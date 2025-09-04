clc;
clear;
close all hidden;

img = imread("0005_0001.JPG");

img = imresize(img, 0.5, 'bicubic');
img_hsv = rgb2hsv(img);

mask = DetectedBackground_2(img);
true_img = uint8( bsxfun(@times, double(img), double(mask)) );
imtool(true_img)

imwrite(true_img, "Ex_Leaf.JPG")



function mask = DetectBackground_1(img)
    
    img_hsv = rgb2hsv(img);
    mask = zeros(size(img, 1), size(img_hsv, 2));

    % R = img(:,:,1);
    % G = img(:,:,2);
    % B = img(:,:,3);
    % 
    % % Compute absolute differences
    % diffRG = abs(R - G);
    % diffRB = abs(R - B);
    % diffGB = abs(G - B);
    % 
    % % Threshold to decide "similarity"
    % threshold = 2;
    % mask = (diffRG < threshold) & (diffRB < threshold) & (diffGB < threshold) & (R < 90);

    index = find(img_hsv(:, :, 1) > 0.25 & img_hsv(:, :, 1) < 0.70 & ...
                    img_hsv(:, :, 2) < 0.65 ...
                    & img_hsv(:, :, 3) < 0.55);
    
    mask(index) = 1;
    imtool(mask)
    
    % Filtering
    mask = medfilt2(mask,[6, 6]);
    
    % Get median with helper function
    med = GetMedian(mask);

    % Morphing and Filtering mask
    big_radius = floor(sqrt(med) / 6);
    small_radius = floor(sqrt(med) / 6);
    big_disk = strel("disk", big_radius);
    small_disk = strel('disk', small_radius);

    
    % Initial Morphing
    mask = imclose(mask, big_disk);
    mask = imopen(mask, small_disk);
    
    % More Morphing
    % Bigger disk for closing to merge fragments
    closing_disk = strel('disk', big_radius);
    mask = imclose(mask, closing_disk);
    
    % Smaller disk for opening to remove small junk
    opening_disk = strel('disk', small_radius);
    mask = imopen(mask, opening_disk);

    imtool(mask);
    
end


function [m, areas] = GetMedian(mask)
    cc = bwlabel(mask, 8);
    allAreas = regionprops(cc, 'Area');
    areas = [allAreas.Area];
    if isempty(areas)
        m = 0;
    else
        m = median(areas);
    end
end


function mask = DetectedBackground_2(img)

    mask = zeros(size(img, 1), size(img, 2));
    
    index = find(img(:, :, 1) < img(:, :, 2) & img(:, :, 3) < img(:, :, 2));
    mask(index) = 1;
    mask = imfill(mask);

    img = uint8( bsxfun(@times, double(img), double(mask)) );
    
    img_hsv = rgb2hsv(img);
    
    index = find(img_hsv(:, :, 1) < 0.75 & ...
                img_hsv(:, :, 2) < 0.10 & ...
                img_hsv(:, :, 3) < 0.25);
    
    mask = ones(size(img, 1), size(img, 2));
    mask(index) = 0;
    
    
    img = uint8( bsxfun(@times, double(img), double(mask)) );
    grey_img = rgb2gray(img);
    
    img_labeled = bwlabel(grey_img, 4);
    allAreas = regionprops(img_labeled, 'Area');
    areas = [allAreas.Area];
    leaf_area = max(max(areas));
    leaf_index = find(areas == leaf_area);
    mask = zeros(size(img_labeled));
    mask(img_labeled == leaf_index) = 1;
    imtool(mask);
end