function track = createNewTrack(trackID,bbox,confidence)
% createNewTrack
%
% Sukuria naują YBTrack track.
%
% bbox:
%   [x y width height]

%% --------------------------------------------------------
% Bounding box centras
% ---------------------------------------------------------

center = [ ...
    bbox(1) + bbox(3)/2, ...
    bbox(2) + bbox(4)/2 ];

%% --------------------------------------------------------
% Kalman Filter
% ---------------------------------------------------------

kalmanFilter = configureKalmanFilter( ...
    "ConstantVelocity", ...
    center, ...
    [100 25], ...
    [25 10], ...
    100);

%% --------------------------------------------------------
% Track struktūra
% ---------------------------------------------------------

track.id = trackID;

track.bbox = bbox;

track.kalmanFilter = kalmanFilter;

track.age = 1;

track.totalVisibleCount = 1;

track.consecutiveInvisibleCount = 0;

track.confidence = confidence;

% Kadangi naujas track'as buvo sukurtas iš
% aukšto confidence detection:
track.isConfirmed = true;

end