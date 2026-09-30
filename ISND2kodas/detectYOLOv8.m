function [bboxes, scores, labels] = detectYOLOv8(image, modelName, executionEnvironment)
% detectYOLOv8
% ------------------------------------------------------------
% YOLOv8s inference adapter for the YBTrack MATLAB project.
%
% This version intentionally does NOT use yolov8ObjectDetector.
% It loads the pretrained YOLOv8 dlnetwork directly and performs:
%   1) preprocessing
%   2) network prediction
%   3) YOLOv8 post-processing
%   4) conversion of COCO class IDs to labels
%   5) filtering to the "person" class
%
% INPUTS
%   image                RGB input image (H x W x 3)
%   modelName            "yolov8s" by default
%   executionEnvironment "auto", "cpu" or "gpu" (default: "auto")
%
% OUTPUTS
%   bboxes  - N-by-4 [x y width height]
%   scores  - N-by-1 detection confidence
%   labels  - N-by-1 categorical labels (only "person")
%
% Requirements:
%   - Computer Vision Toolbox
%   - Deep Learning Toolbox
%   - Deep Learning Toolbox Converter for ONNX Model Format
%   - Installed MathWorks YOLOv8 Add-On
%
% The MathWorks YOLOv8 repository documents the same low-level
% dlnetwork inference pipeline used here: preprocessing, predict,
% and postprocessing. The repository provides pretrained yolov8s.
%
% ------------------------------------------------------------

    arguments
        image (:,:,:) uint8
        modelName (1,1) string = "yolov8s"
        executionEnvironment (1,1) string = "auto"
    end

    % Validate image.
    if size(image,3) ~= 3
        error("detectYOLOv8:InvalidImage", ...
            "Įvesties vaizdas turi būti RGB (H x W x 3).");
    end

    executionEnvironment = lower(executionEnvironment);

    if ~ismember(executionEnvironment, ["auto","cpu","gpu"])
        error("detectYOLOv8:InvalidEnvironment", ...
            'executionEnvironment turi būti "auto", "cpu" arba "gpu".');
    end

    % ---------------------------------------------------------
    % Cache the network so it is loaded/downloaded only once.
    % ---------------------------------------------------------
    persistent net classNames numClasses inputSize loadedModel

    if isempty(net) || isempty(loadedModel) || loadedModel ~= modelName

        % The official MathWorks helper downloads the pretrained
        % COCO YOLOv8 model if it is not already present.
        try
            model = helper.downloadPretrainedYOLOv8(modelName);
        catch ME
            error("detectYOLOv8:ModelLoadFailed", ...
                "Nepavyko užkrauti %s modelio.%s%s", ...
                modelName, newline, ME.message);
        end

        % The downloaded MAT file contains the pretrained dlnetwork
        % in the field "net".
        if ~isfield(model, "net")
            error("detectYOLOv8:InvalidModelFile", ...
                "Atsisiųstame YOLOv8 modelio faile nerastas laukas 'net'.");
        end

        net = model.net;

        % COCO class names supplied by the MathWorks Add-On.
        try
            classNames = helper.getCOCOClassNames;
        catch ME
            error("detectYOLOv8:ClassesUnavailable", ...
                "Nepavyko gauti COCO klasių pavadinimų.%s%s", ...
                newline, ME.message);
        end

        numClasses = numel(classNames);

        % Input size is read directly from the first network layer.
        inputSize = net.Layers(1).InputSize;

        loadedModel = modelName;

        fprintf("YOLOv8 modelis užkrautas: %s\n", modelName);
        fprintf("YOLOv8 įvesties dydis: %d x %d x %d\n", inputSize);
        fprintf("COCO klasių skaičius: %d\n", numClasses);
    end

    % ---------------------------------------------------------
    % Choose execution environment.
    % ---------------------------------------------------------
    if executionEnvironment == "auto"
        if canUseGPU
            execEnv = "gpu";
        else
            execEnv = "cpu";
        end
    else
        execEnv = executionEnvironment;
    end

    if execEnv == "gpu" && ~canUseGPU
        error("detectYOLOv8:GPUUnavailable", ...
            "Pasirinktas GPU, tačiau tinkamas GPU nepasiekiamas. " + ...
            "Naudok executionEnvironment=""cpu"" arba ""auto"".");
    end

    % ---------------------------------------------------------
    % YOLOv8 preprocessing.
    % The MathWorks implementation expects BGR during this step,
    % then converts back to RGB before creating the dlarray.
    % ---------------------------------------------------------
    origSize = size(image);

    Ibgr = image(:,:,[3 2 1]);

    img = helper.preprocess(Ibgr, inputSize);

    newSize = size(img);

    img = img(:,:,[3 2 1]);

    % Convert to dlarray using spatial-spatial-channel-batch format.
    dlInput = dlarray(img, "SSCB");

    % Move input to GPU when requested/available.
    if execEnv == "gpu"
        dlInput = gpuArray(dlInput);
    end

    % ---------------------------------------------------------
    % Forward pass.
    % ---------------------------------------------------------
    outFeatureMaps = cell(numel(net.OutputNames), 1);

    try
        [outFeatureMaps{:}] = predict(net, dlInput);
    catch ME
        error("detectYOLOv8:PredictionFailed", ...
            "YOLOv8 prognozė nepavyko.%s%s", ...
            newline, ME.message);
    end

    % ---------------------------------------------------------
    % YOLOv8 post-processing.
    % This returns:
    %   bboxes  -> [x y width height]
    %   scores  -> confidence
    %   labelIds -> COCO class IDs
    % ---------------------------------------------------------
    try
        [bboxes, scores, labelIds] = helper.postprocess( ...
            outFeatureMaps, origSize, newSize, numClasses);
    catch ME
        error("detectYOLOv8:PostprocessFailed", ...
            "YOLOv8 post-processing nepavyko.%s%s", ...
            newline, ME.message);
    end

    % Gather GPU results back to CPU memory.
    bboxes = gather(bboxes);
    scores = gather(scores);
    labelIds = gather(labelIds);

    % ---------------------------------------------------------
    % Convert IDs -> class labels.
    % ---------------------------------------------------------
    classNames = string(classNames(:));

    if isempty(labelIds)
        labels = categorical(strings(0,1), classNames);
        bboxes = zeros(0,4);
        scores = zeros(0,1);
        return;
    end

    labelIds = double(labelIds(:));

    % Defensive check against invalid class IDs.
    valid = labelIds >= 1 & labelIds <= numClasses;

    bboxes = bboxes(valid,:);
    scores = double(scores(valid));
    labelIds = labelIds(valid);

    allLabels = categorical(classNames);
    labelsAll = allLabels(labelIds);

    % ---------------------------------------------------------
    % IMPORTANT FOR YBTrack:
    % Keep only the COCO "person" class.
    % ---------------------------------------------------------
    personMask = string(labelsAll) == "person";

    bboxes = double(bboxes(personMask,:));
    scores = double(scores(personMask));
    labels = labelsAll(personMask);

    % Ensure expected dimensions even for one/zero detection.
    if isempty(bboxes)
        bboxes = zeros(0,4);
        scores = zeros(0,1);
        labels = categorical(strings(0,1), classNames);
    else
        scores = scores(:);
        labels = labels(:);
    end
end
