clc;
clear;
close all;

%% 1. Image Folder
imageFolder = fullfile(pwd, 'images');

%% 2. Find JPG, JPEG and PNG Images
files = [dir(fullfile(imageFolder, '*.jpg'));
         dir(fullfile(imageFolder, '*.jpeg'));
         dir(fullfile(imageFolder, '*.png'))];

fprintf('Number of images found: %d\n', length(files));

if isempty(files)
    error('No images found. Put your images inside the images folder.');
end

%% 3. Load YOLO v4 Model
fprintf('\nLoading YOLO v4 model...\n');

detector = yolov4ObjectDetector("tiny-yolov4-coco");

fprintf('YOLO model loaded successfully.\n');

%% 4. Initialize Total Object Counters

totalCars = 0;
totalPersons = 0;
totalBuses = 0;
totalMotorcycles = 0;

%% 5. Initialize Confidence Score Arrays

confidenceCars = [];
confidencePersons = [];
confidenceBuses = [];
confidenceMotorcycles = [];

%% 6. Process All Images

for i = 1:length(files)

    %% Read Image
    img = imread(fullfile(imageFolder, files(i).name));

    %% YOLO Detection
    [bboxes, scores, labels] = detect(detector, img);

    %% Convert Labels to String
    labelStrings = string(labels);

    %% Store Confidence Scores

    confidenceCars = [confidenceCars;
                      scores(labelStrings == "car")];

    confidencePersons = [confidencePersons;
                         scores(labelStrings == "person")];

    confidenceBuses = [confidenceBuses;
                       scores(labelStrings == "bus")];

    confidenceMotorcycles = [confidenceMotorcycles;
                             scores(labelStrings == "motorcycle")];

    %% Draw Bounding Boxes

    detectedImg = insertObjectAnnotation( ...
        img, ...
        "rectangle", ...
        bboxes, ...
        labels);

    %% Display Detection Result

    figure;

    imshow(detectedImg);

    title("YOLO v4 Detection - " + files(i).name);

    %% Display Image Information

    fprintf('\n========================================\n');

    fprintf('Image: %s\n', files(i).name);

    fprintf('Objects detected: %d\n', length(labels));

    fprintf('========================================\n');

    disp(labels);

    %% Count Objects in Current Image

    numCars = sum(labelStrings == "car");

    numPersons = sum(labelStrings == "person");

    numBuses = sum(labelStrings == "bus");

    numMotorcycles = sum(labelStrings == "motorcycle");

    %% Display Current Image Counts

    fprintf('Cars        = %d\n', numCars);

    fprintf('Persons     = %d\n', numPersons);

    fprintf('Buses       = %d\n', numBuses);

    fprintf('Motorcycles = %d\n', numMotorcycles);

    %% Add Counts to Total

    totalCars = totalCars + numCars;

    totalPersons = totalPersons + numPersons;

    totalBuses = totalBuses + numBuses;

    totalMotorcycles = totalMotorcycles + numMotorcycles;

end

%% 7. Display Total Object Results

fprintf('\n\n');

fprintf('========================================\n');
fprintf('       TOTAL OBJECT DETECTION RESULTS\n');
fprintf('========================================\n');

fprintf('Total Cars        = %d\n', totalCars);

fprintf('Total Persons     = %d\n', totalPersons);

fprintf('Total Buses       = %d\n', totalBuses);

fprintf('Total Motorcycles = %d\n', totalMotorcycles);

fprintf('========================================\n');

%% 8. Object Count Bar Chart

objectNames = {'Cars', 'Persons', 'Buses', 'Motorcycles'};

objectCounts = [
    totalCars;
    totalPersons;
    totalBuses;
    totalMotorcycles
];

figure;

bar(objectCounts);

set(gca, 'XTick', 1:4);

set(gca, 'XTickLabel', objectNames);

xlabel('Object Type');

ylabel('Number of Objects');

title('YOLO v4 Object Detection Analysis');

grid on;

%% Display Object Count Values

for i = 1:length(objectCounts)

    text(i, objectCounts(i), ...
        num2str(objectCounts(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontWeight', 'bold');

end

%% 9. Calculate Average Confidence Scores

if isempty(confidenceCars)
    avgCarConfidence = 0;
else
    avgCarConfidence = mean(confidenceCars) * 100;
end

if isempty(confidencePersons)
    avgPersonConfidence = 0;
else
    avgPersonConfidence = mean(confidencePersons) * 100;
end

if isempty(confidenceBuses)
    avgBusConfidence = 0;
else
    avgBusConfidence = mean(confidenceBuses) * 100;
end

if isempty(confidenceMotorcycles)
    avgMotorcycleConfidence = 0;
else
    avgMotorcycleConfidence = mean(confidenceMotorcycles) * 100;
end

%% 10. Display Average Confidence

fprintf('\n\n');

fprintf('========================================\n');
fprintf('       AVERAGE CONFIDENCE SCORES\n');
fprintf('========================================\n');

fprintf('Cars        = %.2f%%\n', avgCarConfidence);

fprintf('Persons     = %.2f%%\n', avgPersonConfidence);

fprintf('Buses       = %.2f%%\n', avgBusConfidence);

fprintf('Motorcycles = %.2f%%\n', avgMotorcycleConfidence);

fprintf('========================================\n');

%% 11. Confidence Score Bar Chart

confidenceValues = [
    avgCarConfidence;
    avgPersonConfidence;
    avgBusConfidence;
    avgMotorcycleConfidence
];

figure;

bar(confidenceValues);

set(gca, 'XTick', 1:4);

set(gca, 'XTickLabel', ...
    {'Cars', 'Persons', 'Buses', 'Motorcycles'});

xlabel('Object Type');

ylabel('Average Confidence (%)');

title('YOLO v4 Average Detection Confidence');

ylim([0 100]);

grid on;

%% Display Confidence Values

for i = 1:length(confidenceValues)

    text(i, confidenceValues(i), ...
        sprintf('%.1f%%', confidenceValues(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontWeight', 'bold');

end

%% 12. Final Message

fprintf('\n');
fprintf('========================================\n');
fprintf('YOLO OBJECT DETECTION ANALYSIS COMPLETE\n');
fprintf('========================================\n');

fprintf('Object count chart created successfully.\n');
fprintf('Confidence score chart created successfully.\n');