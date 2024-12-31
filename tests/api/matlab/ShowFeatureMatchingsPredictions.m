function [] = ShowFeatureMatchingsPredictions(ui8Image1, ui8Image2, matchedPoints1, matchedPoints2)
arguments
    ui8Image1
    ui8Image2
    matchedPoints1
    matchedPoints2
end


% Input: Two images and matching points
% ui8Image1, ui8Image2: Input images
% matchedPoints1, matchedPoints2: Matching points (Nx2 matrices with [x, y] coordinates)

% Display images side by side
figure();
combinedImage = [ui8Image1, ui8Image2];
imshow(combinedImage);
hold on;

% Offset for the second image
offset = size(ui8Image1, 2);

% Plot matches
for i = 1:size(matchedPoints1, 1)
    % Coordinates of matched points
    x1 = matchedPoints1(i, 1);
    y1 = matchedPoints1(i, 2);
    x2 = matchedPoints2(i, 1) + offset; % Adjust x-coordinate for the second image
    y2 = matchedPoints2(i, 2);

    % Draw line connecting the matched points
    plot([x1, x2], [y1, y2], 'g-', 'LineWidth', 1);

    % Mark the keypoints
    plot(x1, y1, 'ro', 'MarkerSize', 5, 'LineWidth', 1.5);
    plot(x2, y2, 'bo', 'MarkerSize', 5, 'LineWidth', 1.5);
end

% Add title and legend
title('Feature Matches');
legend('Matches', 'Keypoints in Image 1', 'Keypoints in Image 2', 'Location', 'southoutside');

hold off;

end
