%% YBTrack MATLAB - main.m
% ================================================================
% YBTrack eksperimentas:
% YOLOv8s + ByteTrack (2 etapai) + EIoU + Kalman + Followspot
%
% Svarbu:
%   - NENAUDOJAMAS yolov8ObjectDetector, nes tavo MATLAB R2026a
%     aplinkoje jo superclass sukelia parse klaidą.
%   - Naudojamas oficialaus MathWorks YOLOv8 Add-On dlnetwork:
%       data = helper.downloadPretrainedYOLOv8("yolov8s");
%       dlnet = data.yolov8Net;
%
% Tavo projekto aplanke turi būti:
%   main.m
%   calculateEIoUCost.m
%   associateTracks.m
%   createNewTrack.m
%   updateTrack.m
%   drawFollowspot.m
%   Namų darbas 1.1.mp4
%
% Rezultatas:
%   output/stage_followspot_output.mp4
% ================================================================

clear;
clc;
close all;

%% ================================================================
% 1. KONFIGŪRACIJA
% ================================================================

inputVideo  = "Namų darbas 1.1.mp4";
outputDir   = "output";
outputVideo = fullfile(outputDir, "stage_followspot_output.mp4");

% ByteTrack confidence slenksčiai
D_high = 0.60;
D_low  = 0.10;

% EIoU expansion
e1 = 0.20;      % pirmas asociacijos etapas (high confidence)
e2 = 0.50;      % antras asociacijos etapas (low confidence)

% Track deletion policy
maxLostFrames = 30;

% Cost = 1 - EIoU
maxAssociationCost = 0.80;

% Followspot
followspotRadius  = 55;
followspotOpacity = 0.30;

% Inference aplinka: "auto", "cpu" arba "gpu"
executionEnvironment = "auto";

% Rodyti apdorojamą kadrą
showDisplay = true;

%% ================================================================
% 2. INPUT VIDEO PATIKRA
% ================================================================

if ~isfile(inputVideo)
    msg = sprintf([ ...
        'KLAIDA: Nerastas įvesties vaizdo įrašas.\n' ...
        'Tikėtinas failas:\n    %s\n\n' ...
        'Įkelkite "Namų darbas 1.1.mp4" į MATLAB projekto katalogą.' ...
        ], fullfile(pwd, inputVideo));

    error('%s', msg);
end

fprintf("Įvesties video: %s\n", fullfile(pwd, inputVideo));

%% ================================================================
% 3. TOOLBOX IR YOLO HELPER FUNKCIJŲ PATIKRA
% ================================================================

fprintf("\nTikrinama MATLAB aplinka...\n");

requiredFunctions = { ...
    "assignDetectionsToTracks"
    "bboxOverlapRatio"
    "configureKalmanFilter"
    "dlnetwork"
    "selectStrongestBboxMulticlass"
    };

for i = 1:numel(requiredFunctions)
    p = which(requiredFunctions{i});
    if isempty(p)
        error("YBTrack:MissingFunction", ...
            'Nerasta reikalinga funkcija "%s". Patikrinkite Toolbox įdiegimą.', ...
            requiredFunctions{i});
    end
end

fprintf("  Computer Vision / tracking funkcijos ... OK\n");
fprintf("  Deep Learning funkcijos ... OK\n");

yoloHelperFunctions = { ...
    "helper.downloadPretrainedYOLOv8"
    "helper.preprocess"
    "helper.yolov8Transform"
    "helper.extractDetections"
    "helper.postProcessYOLOv8"
    "helper.getCOCOClassNames"
    };

for i = 1:numel(yoloHelperFunctions)
    p = which(yoloHelperFunctions{i});
    if isempty(p)
        error("YBTrack:YOLOAddOnIncomplete", ...
            ['Nerasta YOLOv8 Add-On funkcija "%s".\n' ...
             'Add-On Explorer patikrinkite, ar "Object Detection And ' ...
             'Instance Segmentation Using YOLO v8" yra įdiegtas.'], ...
             yoloHelperFunctions{i});
    end
end

fprintf("  YOLOv8 Add-On helper funkcijos ... OK\n");

%% ================================================================
% 4. VIDEO READER
% ================================================================

videoReader = VideoReader(inputVideo);

videoFPS      = videoReader.FrameRate;
videoDuration = videoReader.Duration;
videoWidth    = videoReader.Width;
videoHeight   = videoReader.Height;

fprintf("\nVideo informacija:\n");
fprintf("  FPS: %.2f\n", videoFPS);
fprintf("  Trukmė: %.2f s\n", videoDuration);
fprintf("  Rezoliucija: %d x %d\n", videoWidth, videoHeight);

%% ================================================================
% 5. YOLOv8s MODELIO ĮKĖLIMAS
% ================================================================

fprintf("\nKraunamas YOLOv8s modelis...\n");

modelName = 'yolov8s';

try
    if ~isfolder('models')
        mkdir('models');
    end
    modelData = helper.downloadPretrainedYOLOv8(modelName);
catch ME
    error("YBTrack:YOLOModelLoad", ...
        "Nepavyko užkrauti YOLOv8s modelio:\n%s", ME.message);
end

% Pagal oficialų MathWorks Add-On helper.downloadPretrainedYOLOv8
% grąžinamame MAT faile tinklas saugomas kaip "yolov8Net".
if ~isstruct(modelData) || ~isfield(modelData, "yolov8Net")
    error("YBTrack:YOLOModelFormat", ...
        "YOLOv8 modelio faile nerastas laukas 'yolov8Net'.");
end

dlnet = modelData.yolov8Net;

classNames = string(helper.getCOCOClassNames);
numClasses = numel(classNames);

personClassId = find(strcmpi(classNames, "person"), 1);

if isempty(personClassId)
    error("YBTrack:PersonClassMissing", ...
        'COCO klasių sąraše nerasta klasė "person".');
end

fprintf("YOLOv8s modelis įkeltas.\n");
fprintf("  Klasės: %d\n", numClasses);
fprintf("  Person class ID: %d\n", personClassId);

%% ================================================================
% 6. EXECUTION ENVIRONMENT
% ================================================================

switch lower(executionEnvironment)

    case "auto"
        if canUseGPU
            actualExecutionEnvironment = "gpu";
        else
            actualExecutionEnvironment = "cpu";
        end

    case "gpu"
        if ~canUseGPU
            error("YBTrack:GPUUnavailable", ...
                "GPU pasirinktas, bet GPU nepasiekiamas.");
        end
        actualExecutionEnvironment = "gpu";

    case "cpu"
        actualExecutionEnvironment = "cpu";

    otherwise
        error("YBTrack:BadExecutionEnvironment", ...
            'executionEnvironment turi būti "auto", "cpu" arba "gpu".');
end

fprintf("Inference aplinka: %s\n", upper(actualExecutionEnvironment));

%% ================================================================
% 7. VIENO KADRO DETEKCIJOS TESTAS
% ================================================================

fprintf("\nAtliekamas vieno kadro YOLOv8 testas...\n");

testReader = VideoReader(inputVideo);
testFrame = readFrame(testReader);

try
    [testBoxes, testScores, testLabels] = runYOLOv8Detection( ...
        dlnet, ...
        testFrame, ...
        numClasses, ...
        D_low, ...
        actualExecutionEnvironment, ...
        personClassId, ...
        classNames);
catch ME
    error("YBTrack:YOLOInference", ...
        "YOLOv8 testinis inference nepavyko:\n%s", ME.message);
end

fprintf("YOLOv8 testas sėkmingas.\n");
fprintf("  Pirmame kadre rastų žmonių: %d\n", numel(testScores));

if showDisplay
    testAnnotated = insertObjectAnnotation( ...
        testFrame, ...
        "rectangle", ...
        testBoxes, ...
        compose("person %.2f", testScores), ...
        Color="yellow", ...
        LineWidth=2);

    figure( ...
        "Name", "YOLOv8 testas", ...
        "NumberTitle", "off");

    imshow(testAnnotated);
    title(sprintf("YOLOv8s | Person detections: %d", size(testBoxes,1)));
    drawnow;
end

%% ================================================================
% 8. OUTPUT VIDEO
% ================================================================

if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

videoWriter = VideoWriter(outputVideo, "MPEG-4");
videoWriter.FrameRate = videoFPS;
open(videoWriter);

writerCleanup = onCleanup(@()safeCloseVideoWriter(videoWriter));

%% ================================================================
% 9. TRACK STRUKTŪRA
% ================================================================

tracks = struct( ...
    "id", {}, ...
    "bbox", {}, ...
    "kalmanFilter", {}, ...
    "age", {}, ...
    "totalVisibleCount", {}, ...
    "consecutiveInvisibleCount", {}, ...
    "confidence", {}, ...
    "isConfirmed", {} );

nextTrackID = 1;

%% ================================================================
% 10. STATISTIKOS
% ================================================================

frameNumber = 0;
totalProcessedFrames = 0;

overallTimer = tic;

fpsWindowTimer  = tic;
fpsWindowFrames = 0;
currentFPS      = 0;

displayFigure = [];

if showDisplay
    displayFigure = figure( ...
        "Name", "YBTrack Followspot", ...
        "NumberTitle", "off", ...
        "Color", "black");
end

fprintf("\n");
fprintf("====================================================\n");
fprintf("Pradedamas YBTrack video apdorojimas...\n");
fprintf("====================================================\n");

%% ================================================================
% 11. PAGRINDINIS VIDEO CIKLAS
% ================================================================

while hasFrame(videoReader)

    frame = readFrame(videoReader);

    frameNumber = frameNumber + 1;
    totalProcessedFrames = totalProcessedFrames + 1;

    %% ------------------------------------------------------------
    % 11.1 YOLOv8s DETEKCIJA
    % ------------------------------------------------------------

    [bboxes, scores, labels] = runYOLOv8Detection( ...
        dlnet, ...
        frame, ...
        numClasses, ...
        D_low, ...
        actualExecutionEnvironment, ...
        personClassId, ...
        classNames);

    scores = double(scores(:));

    %% ------------------------------------------------------------
    % 11.2 D_HIGH / D_LOW ATSKYRIMAS
    % ------------------------------------------------------------

    highMask = scores >= D_high;
    lowMask  = scores >= D_low & scores < D_high;

    highBboxes = bboxes(highMask, :);
    highScores = scores(highMask);

    lowBboxes = bboxes(lowMask, :);
    lowScores = scores(lowMask);

    %% ------------------------------------------------------------
    % 11.3 KALMAN PREDICTION
    % ------------------------------------------------------------

    numTracks = numel(tracks);

    predictedBboxes = zeros(numTracks, 4);

    for t = 1:numTracks

        predictedCenter = predict(tracks(t).kalmanFilter);

        w = tracks(t).bbox(3);
        h = tracks(t).bbox(4);

        predictedBboxes(t,:) = [ ...
            predictedCenter(1) - w/2, ...
            predictedCenter(2) - h/2, ...
            w, ...
            h ];
    end

    %% ------------------------------------------------------------
    % 11.4 BYTE TRACK - 1 ETAPAS
    % ------------------------------------------------------------
    % High detections + EIoU e1 = 0.2

    [matchedHigh, unmatchedTracks, unmatchedHigh] = associateTracks( ...
        predictedBboxes, ...
        highBboxes, ...
        e1, ...
        maxAssociationCost);

    %% ------------------------------------------------------------
    % 11.5 HIGH MATCH ATNAUJINIMAS
    % ------------------------------------------------------------

    for m = 1:size(matchedHigh,1)

        trackIndex     = matchedHigh(m,1);
        detectionIndex = matchedHigh(m,2);

        tracks(trackIndex) = updateTrack( ...
            tracks(trackIndex), ...
            highBboxes(detectionIndex,:), ...
            highScores(detectionIndex));
    end

    %% ------------------------------------------------------------
    % 11.6 BYTE TRACK - 2 ETAPAS
    % ------------------------------------------------------------
    % Low detections + EIoU e2 = 0.5

    if ~isempty(unmatchedTracks) && ~isempty(lowBboxes)

        remainingPredicted = predictedBboxes(unmatchedTracks,:);

        [matchedLow, unmatchedTracksLow, ~] = associateTracks( ...
            remainingPredicted, ...
            lowBboxes, ...
            e2, ...
            maxAssociationCost);

        for m = 1:size(matchedLow,1)

            localTrackIndex = matchedLow(m,1);

            globalTrackIndex = unmatchedTracks(localTrackIndex);

            detectionIndex = matchedLow(m,2);

            tracks(globalTrackIndex) = updateTrack( ...
                tracks(globalTrackIndex), ...
                lowBboxes(detectionIndex,:), ...
                lowScores(detectionIndex));
        end

        unmatchedTracks = unmatchedTracks(unmatchedTracksLow);
    end

    %% ------------------------------------------------------------
    % 11.7 TRACK'AI, KURIEMS ŠIAME KADRE NEBUVO MATCH
    % ------------------------------------------------------------

    for k = 1:numel(unmatchedTracks)

        trackIndex = unmatchedTracks(k);

        tracks(trackIndex).consecutiveInvisibleCount = ...
            tracks(trackIndex).consecutiveInvisibleCount + 1;

        tracks(trackIndex).age = ...
            tracks(trackIndex).age + 1;
    end

    %% ------------------------------------------------------------
    % 11.8 NAUJI TRACK'AI
    % ------------------------------------------------------------
    % Naujas track kuriamas tik iš unmatched HIGH detection.

    for k = 1:numel(unmatchedHigh)

        detectionIndex = unmatchedHigh(k);

        newTrack = createNewTrack( ...
            nextTrackID, ...
            highBboxes(detectionIndex,:), ...
            highScores(detectionIndex));

        tracks(end+1) = newTrack; %#ok<SAGROW>

        nextTrackID = nextTrackID + 1;
    end

    %% ------------------------------------------------------------
    % 11.9 TRACK'Ų IŠTRYNIMAS PO 30 NEMATOMŲ KADRŲ
    % ------------------------------------------------------------

    if ~isempty(tracks)

        invisibleCounts = [tracks.consecutiveInvisibleCount];

        tracks = tracks(invisibleCounts <= maxLostFrames);
    end

    %% ------------------------------------------------------------
    % 11.10 FPS
    % ------------------------------------------------------------

    fpsWindowFrames = fpsWindowFrames + 1;

    fpsElapsed = toc(fpsWindowTimer);

    if fpsElapsed >= 1
        currentFPS = fpsWindowFrames / fpsElapsed;

        fpsWindowFrames = 0;
        fpsWindowTimer = tic;
    end

    %% ------------------------------------------------------------
    % 11.11 FOLLOWSPOT
    % ------------------------------------------------------------

    outputFrame = drawFollowspot( ...
        frame, ...
        tracks, ...
        currentFPS, ...
        followspotRadius, ...
        followspotOpacity);

    %% ------------------------------------------------------------
    % 11.12 RAŠOME Į MP4
    % ------------------------------------------------------------

    writeVideo(videoWriter, outputFrame);

    %% ------------------------------------------------------------
    % 11.13 DISPLAY
    % ------------------------------------------------------------

    if showDisplay && ~isempty(displayFigure) && isvalid(displayFigure)

        figure(displayFigure);

        imshow(outputFrame);

        activeTracks = countActiveTracks(tracks);

        title(sprintf( ...
            "YBTrack Followspot | Frame %d | FPS %.1f | Active tracks %d", ...
            frameNumber, ...
            currentFPS, ...
            activeTracks), ...
            "Color", "white");

        drawnow limitrate;
    end
end

%% ================================================================
% 12. GALUTINĖ STATISTIKA
% ================================================================

totalProcessingTime = toc(overallTimer);

if totalProcessingTime > 0
    averageFPS = totalProcessedFrames / totalProcessingTime;
else
    averageFPS = 0;
end

% Uždaryti VideoWriter.
clear writerCleanup;

%% ================================================================
% 13. COMMAND WINDOW REZULTATAI
% ================================================================

fprintf("\n");
fprintf("====================================================\n");
fprintf(" YBTrack MATLAB REZULTATAI\n");
fprintf("====================================================\n");

fprintf("Įvesties video:\n  %s\n\n", ...
    fullfile(pwd, inputVideo));

fprintf("Išvesties video:\n  %s\n\n", ...
    fullfile(pwd, outputVideo));

fprintf("YOLO modelis:\n  %s\n\n", modelName);

fprintf("Bendras apdorotų kadrų skaičius:\n  %d\n\n", ...
    totalProcessedFrames);

fprintf("Bendras apdorojimo laikas:\n  %.2f s\n\n", ...
    totalProcessingTime);

fprintf("Vidutinis apdorojimo FPS:\n  %.2f FPS\n\n", ...
    averageFPS);

fprintf("Įvesties video nominalus FPS:\n  %.2f FPS\n\n", ...
    videoFPS);

fprintf("D_high:\n  %.2f\n\n", D_high);

fprintf("D_low:\n  %.2f\n\n", D_low);

fprintf("EIoU e1:\n  %.2f\n\n", e1);

fprintf("EIoU e2:\n  %.2f\n\n", e2);

fprintf("Track ištrynimas po:\n  %d nematomų kadrų\n\n", ...
    maxLostFrames);

fprintf("====================================================\n");
fprintf("Apdorojimas baigtas.\n");
fprintf("====================================================\n");

%% ================================================================
% VIETINĖ FUNKCIJA: YOLOv8s
% ================================================================

function [bboxes, scores, labels] = runYOLOv8Detection( ...
    dlnet, ...
    image, ...
    numClasses, ...
    confidenceThreshold, ...
    executionEnvironment, ...
    personClassId, ...
    classNames)
% runYOLOv8Detection
%
% Naudoja oficialias MathWorks YOLOv8 helper funkcijas.
%
% Skirtumas nuo oficialaus detectYOLOv8.m:
%   oficialus helper.postprocess naudoja confThreshold = 0.5;
%   čia kviečiame helper.extractDetections tiesiogiai ir leidžiame
%   confidenceThreshold = 0.10, kad ByteTrack gautų low-confidence
%   detekcijas antrajam etapui.

    if size(image,3) ~= 3
        error("YOLOv8 įvestis turi būti RGB vaizdas.");
    end

    inputSize = dlnet.Layers(1).InputSize;

    % Official YOLOv8 preprocessing
    origSize = size(image);

    Ibgr = image(:,:,[3,2,1]);

    img = helper.preprocess(Ibgr, inputSize);

    newSize = size(img);

    img = img(:,:,[3,2,1]);

    % SS CB = Spatial, Spatial, Channel, Batch
    dlInput = dlarray(img, "SSCB");

    if executionEnvironment == "gpu"
        dlInput = gpuArray(dlInput);
    end

    % Network prediction
    outFeatureMaps = cell(numel(dlnet.OutputNames), 1);

    [outFeatureMaps{:}] = predict(dlnet, dlInput);

    % Convert feature maps to [box; class scores]
    [box, cls] = helper.yolov8Transform( ...
        outFeatureMaps, ...
        numClasses);

    predictions = cat(1, box, cls);

    % ------------------------------------------------------------
    % IMPORTANT:
    % Naudojame D_low = 0.10 prieš NMS.
    % helper.extractDetections atlieka:
    %   - confidence filtering
    %   - class selection
    %   - NMS
    % ------------------------------------------------------------

    [bboxesPred, scores, labelIds] = helper.extractDetections( ...
        predictions, ...
        numClasses, ...
        confidenceThreshold);

    % Jeigu nieko nerasta
    if isempty(bboxesPred)

        bboxes = zeros(0,4);
        scores = zeros(0,1);
        labels = strings(0,1);

        return;
    end

    % Convert [x y w h] to [x1 y1 x2 y2]
    bboxesTmp = bboxesPred;

    bboxesTmp(:,3) = ...
        bboxesTmp(:,1) + bboxesTmp(:,3) - 1;

    bboxesTmp(:,4) = ...
        bboxesTmp(:,2) + bboxesTmp(:,4) - 1;

    % Map boxes from network input back to original image
    bboxesPost = helper.postProcessYOLOv8( ...
        newSize, ...
        bboxesTmp, ...
        origSize);

    % Convert back to [x y width height]
    bboxes = bboxesPost;

    bboxes(:,3) = ...
        bboxesPost(:,3) - bboxesPost(:,1) + 1;

    bboxes(:,4) = ...
        bboxesPost(:,4) - bboxesPost(:,2) + 1;

    bboxes = double(bboxes);
    scores = double(scores(:));
    labelIds = double(labelIds(:));

    % ------------------------------------------------------------
    % Keep only PERSON class.
    % ------------------------------------------------------------

    valid = labelIds == personClassId;

    bboxes = bboxes(valid,:);
    scores = scores(valid);
    labelIds = labelIds(valid);

    if isempty(bboxes)

        bboxes = zeros(0,4);
        scores = zeros(0,1);
        labels = strings(0,1);

        return;
    end

    labels = string(classNames(labelIds));

    labels = labels(:);
end

%% ================================================================
% VIETINĖ FUNKCIJA: aktyvių track'ų skaičius
% ================================================================

function count = countActiveTracks(tracks)

    if isempty(tracks)
        count = 0;
        return;
    end

    invisible = [tracks.consecutiveInvisibleCount];

    count = sum(invisible == 0);
end

%% ================================================================
% VIETINĖ FUNKCIJA: saugus VideoWriter uždarymas
% ================================================================

function safeCloseVideoWriter(videoWriter)

    try
        if ~isempty(videoWriter)
            close(videoWriter);
        end
    catch
        % Nekeliame antrinės klaidos, jei Writer jau uždarytas.
    end
end
