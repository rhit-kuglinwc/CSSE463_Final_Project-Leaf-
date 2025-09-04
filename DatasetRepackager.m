clc;
clear;
close all hidden;

% This program takes in image datasets, attempts to remove backgrounds with
% a monotone background, and repackage them into a new dataset
% List out the datasets as shown:
% process_dataset('dataset_old_1', 'new_dataset');
% process_dataset('dataset_old_2', 'new_dataset');
% The program expects subfolders 'train', 'test', 'valid' to be present and
% contain folders with images

process_dataset('tester', 'new')

fprintf("finsihed repackaging")

function process_dataset(input_base, output_base)

    splits = {'train', 'test', 'valid'};

    for s = 1:numel(splits)
        split = splits{s};
        input_folder = fullfile(input_base, split);
        output_folder = fullfile(output_base, split);

        process_folder(input_folder, output_folder);
    end
end

function process_folder(input_folder, output_folder)
    % Make sure output folder exists
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Get all files and subfolders
    items = dir(input_folder);

    for i = 1:numel(items)
        item = items(i);

        % Skip '.' and '..'
        if item.isdir
            if strcmp(item.name, '.') || strcmp(item.name, '..')
                continue;
            end
            % Recursive call for subfolders
            process_folder(fullfile(input_folder, item.name), ...
                           fullfile(output_folder, item.name));
        else
            % Process image files
            [~,~,ext] = fileparts(item.name);
            if ismember(lower(ext), {'.jpg','.jpeg','.png','.bmp','.tif'})
                img_path = fullfile(input_folder, item.name);
                img = imread(img_path);

                % result = img;

                % If the final result is to only repackage datasets and no
                % background removal, uncomment the above line, and comment
                % out the line below.

                % Run through background remover
                result = BackgroundRemover(img);

                % Save to mirrored path
                out_path = fullfile(output_folder, item.name);
                imwrite(result, out_path);

                % fprintf('Processed %s -> %s\n', img_path, out_path);
            end
        end
    end
end


function true_img = BackgroundRemover(img)

    origonal = img;

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

    if isempty(allAreas)
        % No objects found, return original img
        true_img = origonal;
        return;
    end

    areas = [allAreas.Area];
    leaf_area = max(max(areas));
    leaf_index = find(areas == leaf_area);
    mask = zeros(size(img_labeled));
    mask(img_labeled == leaf_index) = 1;


    true_img = uint8( bsxfun(@times, double(img), double(mask)) );
end