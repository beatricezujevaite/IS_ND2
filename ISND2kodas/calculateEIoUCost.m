function costMatrix = calculateEIoUCost(trackBboxes, detectionBboxes, e)
% calculateEIoUCost
% ================================================================
% Apskaičiuoja EIoU cost matricą tarp track'ų ir detekcijų.
%
% Įvestis:
%   trackBboxes      - N x 4 [x y width height]
%   detectionBboxes  - M x 4 [x y width height]
%   e                - bounding box išplėtimo koeficientas
%
% Išvestis:
%   costMatrix       - N x M, kur:
%                      0 -> maksimalus sutapimas
%                      1 -> nėra sutapimo
%
% EIoU pagal naudojamą YBTrack adaptaciją:
%   Ew = w * (1 + e)
%   Eh = h * (1 + e)
%   x_new = x - e*w/2
%   y_new = y - e*h/2
%
% SVARBU:
% Šiame faile NEGALI būti jokių vykdomų komandų prieš function
% eilutę. Priešingu atveju MATLAB laikys failą skriptu ir
% associateTracks negalės jo kviesti kaip funkcijos.
% ================================================================

    % Patikriname įvestis.
    if nargin ~= 3
        error("calculateEIoUCost:InvalidInput", ...
            "Funkcijai reikia 3 argumentų: trackBboxes, detectionBboxes ir e.");
    end

    if size(trackBboxes,2) ~= 4
        error("calculateEIoUCost:InvalidTrackBoxes", ...
            "trackBboxes turi būti N x 4 matrica [x y width height].");
    end

    if size(detectionBboxes,2) ~= 4
        error("calculateEIoUCost:InvalidDetectionBoxes", ...
            "detectionBboxes turi būti M x 4 matrica [x y width height].");
    end

    if ~isscalar(e) || ~isfinite(e) || e < 0
        error("calculateEIoUCost:InvalidExpansion", ...
            "e turi būti neigiamas skaliarinis skaičius.");
    end

    nTracks = size(trackBboxes,1);
    nDetections = size(detectionBboxes,1);

    % Tuščias atvejis.
    if nTracks == 0 || nDetections == 0
        costMatrix = ones(nTracks, nDetections);
        return;
    end

    % Išplečiame bounding boxes.
    expandedTracks = expandBoxes(trackBboxes, e);
    expandedDetections = expandBoxes(detectionBboxes, e);

    % IoU tarp išplėstų dėžių.
    eIoU = bboxOverlapRatio( ...
        expandedTracks, ...
        expandedDetections, ...
        "Union");

    % Similarity -> cost.
    costMatrix = 1 - eIoU;

    % Apsauga nuo labai mažų skaitinių paklaidų.
    costMatrix = max(0, min(1, costMatrix));
end


function expandedBoxes = expandBoxes(boxes, e)
% expandBoxes
% Išplečia [x y w h] bounding boxes pagal e.

    x = boxes(:,1);
    y = boxes(:,2);
    w = boxes(:,3);
    h = boxes(:,4);

    expandedW = w .* (1 + e);
    expandedH = h .* (1 + e);

    newX = x - (e .* w ./ 2);
    newY = y - (e .* h ./ 2);

    expandedBoxes = [ ...
        newX, ...
        newY, ...
        expandedW, ...
        expandedH];
end
