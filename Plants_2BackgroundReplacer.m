clc;
clear;
close all hidden;

img = imread("CSSE463_Final_Project-Leaf-\0005_0001.JPG");

img = imresize(img, 0.5, 'bicubic');

img_hsv = rgb2hsv(img);
mask = ~DetectBackground(img_hsv); % logical mask

true_img = uint8( bsxfun(@times, double(img), double(mask)) );

imtool(true_img);


function mask = DetectBackground(img_hsv)
    mask = zeros(size(img_hsv, 1), size(img_hsv, 2));

    index = find(img_hsv(:, :, 1) > 0.25 & img_hsv(:, :, 1) < 0.70 & ...
                    img_hsv(:, :, 2) < 0.65 ...
                    & img_hsv(:, :, 3) < 0.55);
    
    mask(index) = 1;
    % imtool(mask)
    
    % Filtering
    mask = medfilt2(mask,[8, 8]);
    
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

    % imtool(mask);
    
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