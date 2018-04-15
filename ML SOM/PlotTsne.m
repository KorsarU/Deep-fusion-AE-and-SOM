function [fig, loss] = PlotTsne(points, pointc)
    fig = figure;
    l = [];
    colorl = [...
        1 0 0  ;... red
        0 1 0  ;... green
        1 1 0.1;... yellow
        0 0.5 1;... blue
        0.3 1 1;... cyan
        0.5 0.1 0.6]; %purple
    if pointc ~= 0
        
        labels2 = pointc.keys;

        for i = 1:length(labels2)
            label = char(labels2(i));
            [y, loss] = tsne(pointc(label));
            fig = scatter(y(:,1), y(:,2), 'o', 'MarkerFaceColor',[0.5 0.5 0.5], 'MarkerEdgeColor', colorl(i,:));
            %fig.Color = ;
            hold on;
        end
        l = [labels2];
    end
        
    labels = points.keys;
    
    for i = 1:length(labels)
        label = char(labels(i));
        [y, loss] = tsne(points(label));
        fig = scatter(y(:,1), y(:,2), 'o', 'MarkerFaceColor', [1 1 1], 'MarkerEdgeColor', colorl(i,:));
        %fig.Color = colorl(i,:);
        hold on;
    end
    l = [l, labels];
    legend(l);
    hold off;
    
end